import logging
from typing import List, Dict
import tensorflow as tf
import pandas as pd
import os
import joblib

import numpy as np
from pydispatch import dispatcher
from sklearn.preprocessing import MinMaxScaler

import communication.pubsub.signals as signals
import util.config as cfg
from agent_components.demand.learning.data import sequence_for_usages
from communication.grpc_messages_pb2 import PBCustomerBootstrapData, PBSimEnd, PBTariffTransaction, PBTimeslotComplete, \
    PBTxType, PBWeatherReport, PBWeatherForecast, PBCompetition
from communication.pubsub.SignalConsumer import SignalConsumer
from agent_components.demand.modelsearch import listAvailableModels, listAvailableScalers
from agent_components.demand.modelutil import load_model_init

log = logging.getLogger(__name__)
UTILITY_MODELS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\past-saved-data\\model_utility2_ckpts"
UTILITY_SCALERS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\scalers\\utilityscalers2"
UTILITY_SINGLE_CSV_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\analysis_data\\single_csv\\utility2"

class TariffUtilityPredictor(SignalConsumer):
    """
    Predictor class that loads models and outputs
    1. Loads model and scalers per customer type - D
    2. Bootstrap Customer Data to prepopulate SeedUsageProfile of 24h - D
    3. Subscription of new tariff spec -> Publish utility prediction - D
    """

    def __init__(self, modelType="DNN"):
        super().__init__()
        self.customer_info = {}
        self.models = {}
        self.scalers = {}
        self.seed_usage_profile = {}
        self.prepared_customer_info_columns = {}
        self.y_vars = set(["Utility"])

        csv_path = ''
        for (modeltype, customerType, modelPath) in listAvailableModels(UTILITY_MODELS_FOLDER, modelType).values():
            single_csv_path = os.path.join(UTILITY_SINGLE_CSV_FOLDER, f"{customerType}.csv")
            csv_path = single_csv_path
            self.models[customerType] = load_model_init(modelPath, single_csv_path, list(self.y_vars))

        self.columnNames = [col for col in pd.read_csv(csv_path).columns if col not in self.y_vars]
        for customerType, customerScalers in listAvailableScalers(UTILITY_SCALERS_FOLDER, self.columnNames).items():
            self.scalers[customerType] = {}
            for (scaler_col, _customerType, scalerPath) in customerScalers.values():
                self.scalers[_customerType][scaler_col] = joblib.load(scalerPath)
        print("Finishing loading TariffUtilityPredictor models")

    def subscribe(self):
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.connect(self.handle_competition_event, signals.PB_COMPETITION)
        dispatcher.connect(self.handle_poss_tariff_spec, signals.POSS_TARIFF_SPEC)
        log.info("TariffUtilityPredictor is listening!")

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.disconnect(self.handle_competition_event, signals.PB_COMPETITION)
        dispatcher.disconnect(self.handle_poss_tariff_spec, signals.POSS_TARIFF_SPEC)

    def handle_poss_tariff_spec(self, sender, signal: str, msg: tuple):
        spec_id, customerName, spec_row = msg
        # Do prediction
        row = self._prepare_input_data(customerName, spec_row)
        row = self._transform_row_with_scalers(row, customerName)
        row = np.array([row])
        customerType = self._getCustomerType(customerName)
        utility = self.models[customerType].predict(row)[1][0][0]
        utility_data = (spec_id, customerName, utility)
        dispatcher.send(signal=signals.UTILITY_EST, msg=utility_data)

    def handle_competition_event(self, sender, signal: str, msg: PBCompetition):
        for customer in msg.customers:
            self.customer_info[customer.name] = customer
        self._generate_customer_info_profile()

    def handle_customer_bootstrap_data_event(self, sender, signal: str, msg: PBCustomerBootstrapData):
        customerName = msg.customerName
        self.seed_usage_profile[customerName] = msg.netUsage
        self._generate_customer_info_profile()

    def _generate_customer_info_profile(self):
        for customerName in self.seed_usage_profile:
            if customerName in self.customer_info:
                self.prepared_customer_info_columns[customerName] = self._prepare_customer_info_data(customerName)


    def _getCustomerType(self, customerName):
        return ''.join([i for i in customerName if i.isalpha()]).lower()

    def _transform_row_with_scalers(self, row, customerName):
        customerType = self._getCustomerType(customerName)
        for idx, col in enumerate(self.columnNames):
            row[idx] = self.scalers[customerType][col].transform(np.array([row[idx]]).reshape(1, -1))[0][0]
        return row

    def _prepare_input_data(self, customerName, spec_row):
        if customerName not in self.prepared_customer_info_columns:
            print(f"Unable to find customer info for utility estimation {customerName}")

        row = spec_row.copy()
        # for col in self.columnNames:
        #     if col in spec_dict:
        #         row.append(spec_dict[col])
        #     else:
        #         break
        row += self.prepared_customer_info_columns[customerName]
        return row

    def _prepare_customer_info_data(self, customerName):
        row = []
        customerInfo = self.customer_info[customerName]
        powerType = customerInfo.powerType.label
        customerClass = "SMALL"
        controllableKW = customerInfo.controllableKW
        upRegulationKW = customerInfo.upRegulationKW
        downRegulationKW = customerInfo.downRegulationKW
        storageCapacity = customerInfo.storageCapacity
        multiContracting = str(False)
        canNegotiate = str(False)

        if hasattr(customerInfo, "customerClass"):
            customerClass = customerInfo.customerClass
        if hasattr(customerInfo, "multiContracting"):
            multiContracting = customerInfo.multiContracting
        if hasattr(customerInfo, "canNegotiate"):
            canNegotiate = customerInfo.canNegotiate

        translation = 0
        for i, col in enumerate(self.columnNames):
            if "PowerType" in col:
                translation = i
                break

        idx = translation
        for col in self.columnNames[idx:]:
            col_split = col.split("-")
            if len(col_split) == 1:
                break
            col_prefix, col_suffix = col_split[0], col_split[1]
            if col_prefix == "PowerType":
                if col_suffix == powerType:
                    row.append(1)
                else:
                    row.append(0)
            elif col_prefix == "CustomerClass":
                if col_suffix == customerClass:
                    row.append(1)
                else:
                    row.append(0)
            else:
                break

        row += [controllableKW, upRegulationKW, downRegulationKW, storageCapacity]

        idx = translation + len(row)
        for col in self.columnNames[idx:]:
            col_split = col.split("-")
            if len(col_split) == 1:
                break
            col_prefix, col_suffix = col_split[0], col_split[1]
            if col_prefix == "MultiContracting":
                if col_suffix == multiContracting:
                    row.append(1)
                else:
                    row.append(0)
            elif col_prefix == "CanNegotiate":
                if col_suffix == canNegotiate:
                    row.append(1)
                else:
                    row.append(0)
            else:
                break

        row += self.seed_usage_profile[customerName]
        return row
