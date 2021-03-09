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
USAGE_PROFILE_MODELS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\past-saved-data\\model_customertype5_ckpts"
USAGE_PROFILE_SCALERS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\scalers\\scalers5"
USAGE_PROFILE_SINGLE_CSV_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\analysis_data\\single_csv\\cust_usage5"

class UsageProfilePredictor(SignalConsumer):
    """
    Predictor class that loads models and outputs
    1. Loads model and scalers per customer type - D
    2. Bootstrap Customer Data to produce InitialUsageProfile of 24h - D
    3. Publish InitialUsageProfile - D
    4. Subscription of new customers -> Add to existing customers - D
    5. Every ts -> Publish profile usage of next 24h for existing customers - D
    """

    def __init__(self, modelType="DNN"):
        super().__init__()
        self.customer_info = {}
        self.models = {}
        self.scalers = {}
        self.seed_usage_profile = {}
        self.initial_usage_profile = {}
        self.existing_customers = set([])
        self.seedWeather = [0] * 336
        self.seedWeatherInitialisedCount = 0
        self.seedWeatherColumns = []
        self.weatherPredictions = {}
        self.y_vars = set(["UsePower-{}".format(i) for i in range(7*24)])
        self.population = {}

        csv_path = ''
        for (modeltype, customerType, modelPath) in listAvailableModels(USAGE_PROFILE_MODELS_FOLDER, modelType).values():
            single_csv_path = os.path.join(USAGE_PROFILE_SINGLE_CSV_FOLDER, f"{customerType}.csv")
            csv_path = single_csv_path
            self.models[customerType] = load_model_init(modelPath, single_csv_path, list(self.y_vars))

        self.columnNames = [col for col in pd.read_csv(csv_path).columns if col not in self.y_vars]
        for customerType, customerScalers in listAvailableScalers(USAGE_PROFILE_SCALERS_FOLDER, self.columnNames).items():
            self.scalers[customerType] = {}
            for (scaler_col, _customerType, scalerPath) in customerScalers.values():
                self.scalers[_customerType][scaler_col] = joblib.load(scalerPath)
        print("Finishing loading UsageProfilePredictor models")

    def subscribe(self):
        """Subscribes this object to the events of interest to the estimator"""
        # dispatcher.connect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        # dispatcher.connect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        # dispatcher.connect(self.handle_sim_end, signals.PB_SIM_END)
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.connect(self.handle_weather_report_event, signals.PB_WEATHER_REPORT)
        dispatcher.connect(self.handle_weather_forecast_event, signals.PB_WEATHER_FORECAST)
        dispatcher.connect(self.handle_competition_event, signals.PB_COMPETITION)
        log.info("estimator is listening!")

    def unsubscribe(self):
        # dispatcher.disconnect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        # dispatcher.disconnect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        # dispatcher.disconnect(self.handle_sim_end, signals.PB_SIM_END)
        dispatcher.disconnect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.disconnect(self.handle_weather_report_event, signals.PB_WEATHER_REPORT)
        dispatcher.disconnect(self.handle_weather_forecast_event, signals.PB_WEATHER_FORECAST)
        dispatcher.disconnect(self.handle_competition_event, signals.PB_COMPETITION)

    def handle_competition_event(self, sender, signal: str, msg: PBCompetition):
        for customer in msg.customers:
            self.customer_info[customer.name] = customer
        self._generate_init_usage_profile()

    def handle_customer_bootstrap_data_event(self, sender, signal: str, msg: PBCustomerBootstrapData):
        customerName = msg.customerName
        self.seed_usage_profile[customerName] = msg.netUsage
        self._generate_init_usage_profile()

    def handle_weather_report_event(self, sender, signal:str, msg: PBWeatherReport):
        ts = msg.currentTimeslot.serialNumber
        if 24 <= ts <= 359:  # Check weather report is seed weather report
            self.seedWeather[ts - 24] = msg.temperature, msg.windSpeed, msg.windDirection, msg.cloudCover
            self.seedWeatherInitialisedCount += 1
            if self.seedWeatherInitialisedCount == 336:
                for weatherTuple in self.seedWeather:
                    self.seedWeatherColumns += list(weatherTuple)
                assert len(self.seedWeatherColumns) == 336 * 4
                self._generate_init_usage_profile()

    def handle_weather_forecast_event(self, sender, signal:str, msg: PBWeatherForecast):
        pred_weather_row = []
        for prediction in msg.predictions:
            pred_weather_row.append(prediction.temperature)
            pred_weather_row.append(prediction.windSpeed)
            pred_weather_row.append(prediction.windDirection)
            pred_weather_row.append(prediction.cloudCover)
        self.weatherPredictions[msg.currentTimeslot] = pred_weather_row
        if msg.currentTimeslot == 360:
            self._generate_init_usage_profile()
        else:
            self._generate_usage_profile_for_existing(msg.currentTimeslot)

    def _generate_init_usage_profile(self):
        for customerName in self.seed_usage_profile:
            if self.seedWeatherInitialisedCount == 336 and 360 in self.weatherPredictions and customerName in self.customer_info:
                row = self._prepare_input_data(customerName, 360)
                row = self._transform_row_with_scalers(row, customerName)
                row = np.array([row])
                customerType = self._getCustomerType(customerName)
                self.initial_usage_profile[customerName] = self.models[customerType].predict(row)[1][0]
                usage_profile_data = (customerName, 360, self.initial_usage_profile)
                dispatcher.send(signal=signals.COMP_USAGE_EST, msg=usage_profile_data)
                print(f"Customer InitialUsage Prepared: {customerName}")

    def _generate_usage_profile_for_existing(self, ts):
        for customerName in self.existing_customers:
            row = self._prepare_input_data(customerName, ts)
            row = self._transform_row_with_scalers(row, customerName)
            row = np.array([row])
            customerType = self._getCustomerType(customerName)
            usage_profile = self.models[customerType].predict(row)[1][0]
            usage_profile_data = (customerName, ts, usage_profile)
            dispatcher.send(signal=signals.COMP_USAGE_EST, msg=usage_profile_data)

    def _getCustomerType(self, customerName):
        return ''.join([i for i in customerName if i.isalpha()]).lower()

    def _transform_row_with_scalers(self, row, customerName):
        customerType = self._getCustomerType(customerName)
        for idx, col in enumerate(self.columnNames):
            row[idx] = self.scalers[customerType][col].transform(np.array([row[idx]]).reshape(1, -1))[0][0]
        return row

    def _prepare_input_data(self, customerName, ts):
        row = self.weatherPredictions[ts].copy()

        customerInfo = self.customer_info[customerName]
        powerType = customerInfo.powerType.label
        customerClass = "SMALL"
        controllableKW = customerInfo.controllableKW
        upRegulationKW = customerInfo.upRegulationKW
        downRegulationKW = customerInfo.downRegulationKW
        storageCapacity = customerInfo.storageCapacity
        multiContracting = str(False)
        canNegotiate = str(False)

        self.population[customerName] = customerInfo.population

        if hasattr(customerInfo, "customerClass"):
            customerClass = customerInfo.customerClass
        if hasattr(customerInfo, "multiContracting"):
            multiContracting = customerInfo.multiContracting
        if hasattr(customerInfo, "canNegotiate"):
            canNegotiate = customerInfo.canNegotiate

        idx = len(row)
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

        idx = len(row)
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
        row += self.seedWeatherColumns

        assert len(row) == len(self.columnNames)

        return row

        # PredWeather
        # CustomerInfo
            # PowerType
            # ControllableKW
            # UpRegulation
            # DownRegulation
            # StorageCapacity
            # MultiContracting
            # CanNegotiate
        # SeedUsage
        # SeedWeather

    def handle_tariff_transaction_event(self, sender, signal: str, msg: PBTariffTransaction):
        customerName = msg.customerInfo.name
        if msg.txType is PBTxType.Value("SIGNUP"):
            self.existing_customers.add(customerName)
        elif msg.txType is PBTxType.Value("WITHDRAW"):
            if customerName in self.existing_customers:
                self.existing_customers.remove(customerName)
            else:
                print(f"Missing customer {customerName}: Cannot withdraw subscription")

    # def handle_customer_change(self, msg: PBTariffTransaction):
    #     """
    #     whenever a SIGNUP or WITHDRAW happens, we need to adapt the customer counts in the estimator
    #     :param msg:
    #     :return:
    #     """
    #
    #     customer = msg.customerInfo.name
    #     c_count = msg.customerCount
    #
    #     #remember the population
    #     self.customer_populations[customer] = msg.customerInfo.population
    #
    #     if msg.txType is PBTxType.Value("WITHDRAW"):
    #         c_count *= -1
    #     if msg.customerInfo.name not in self.customer_counts:
    #         self.customer_counts[customer] = 0
    #
    #     self.customer_counts[customer] += c_count
    #
    #     if self.customer_counts[customer] == 0:
    #         del self.customer_counts[customer]

#     def handle_timeslot_complete(self, sender, signal: str, msg: PBTimeslotComplete):
#         """Triggers an estimation round for all customers"""
#         self.current_timeslot = msg.timeslotIndex + 1
#         # trigger learning on all customers for recently completed TS
#         self.process_customer_new_data()
#
    # def handle_sim_end(self, sender, signal: str, msg: PBSimEnd):
    #     # remove all data
    #     self.usages = {}
    #     self.predictions = {}
    #     self.current_timeslot = 0
#
#     def handle_usage(self, tx: PBTariffTransaction):
#         """Every new usage that is given to the estimator is handled here. It's first scaled to the population of the
#         customer and then stored """
#         customer_name = tx.customerInfo.name
#         kwh = self._convert_to_whole_population(tx.kWh, tx.customerInfo.name)
#         timeslot = tx.postedTimeslot
#         self._apply_usage(customer_name, kwh, timeslot)
#
#     def _convert_to_whole_population(self, usage, name):
#         """pass in the usage that was summed up for this timeslot. It'll be scaled up to the whole population of the
#         customer to estimate on """
#         part = self.customer_counts[name] / self.customer_populations[name]
#         usage = usage / part
#         return usage
#
#     def _convert_from_whole_population(self, customer_prediction: "CustomerPredictions"):
#         """Fixes the customerPredictions object to be appropriate for the actual number of ppl subscribed to us"""
#         for p in customer_prediction.predictions:
#             usage = customer_prediction.predictions[p]
#             name = customer_prediction.customer_name
#             part = self.customer_counts[name] / self.customer_populations[name]
#             usage = usage * part
#             customer_prediction.predictions[p] = usage
#         return customer_prediction
#
#
#     def _apply_usage(self, customer_name, kwh, timeslot):
#         if customer_name not in self.usages:
#             self.usages[customer_name] = {}
#         if timeslot not in self.usages[customer_name]:
#             self.usages[customer_name][timeslot] = 0
#         self.usages[customer_name][timeslot] += kwh
#
#
#     def process_customer_new_data(self):
#         """after the timeslot is completed, this triggers prediction and learning on all timeslots."""
#
#         log.info("starting prcessing of customer data after round")
#
#         now = self.current_timeslot
#         #the 24  TS BEFORE now are         TARGETS
#         #the 168 TS BEFORE the targets are INPUT
#         step1 = cfg.DEMAND_FORECAST_DISTANCE
#         step2 = step1 + cfg.DEMAND_ONE_WEEK
#         target_ts = np.arange(now-step1,now)
#         input_ts = np.arange(now-step2, now-step1)
#         pred_input_ts = np.arange(now-cfg.DEMAND_ONE_WEEK, now)
#         self._ensure_all_there(np.arange(now - step2, now))
#
#         # iterate over all customers
#         #make batches
#         X_ALL = []
#         Y_ALL = []
#         X_PRED_ALL = []
#         scalers = []
#         cust_row_map = []
#
#         #make data into batches that can be passed to the NN
#         current_customers_data = {customer: self.usages[customer] for customer in self.usages if customer in self.customer_counts}
#         log.info("predicting usage for {} customers".format(len(list(current_customers_data.values()))))
#         for c in current_customers_data.items():
#             scaler = self.scalers[c[0]]
#             #store order of customers
#             cust_row_map.append(c[0])
#             scalers.append(scaler)
#             X = np.array([c[1][i] for i in input_ts])
#             X = scaler.transform(X.reshape(-1,1)).flatten()
#             X_ALL.append(X)
#             Y = np.array([c[1][i] for i in target_ts])
#             Y = scaler.transform(Y.reshape(-1,1)).flatten()
#             Y_ALL.append(Y)
#             X_PRED = np.array([c[1][i] for i in pred_input_ts])
#             X_PRED = scaler.transform(X_PRED.reshape(-1,1)).flatten()
#             X_PRED_ALL.append(X_PRED)
#
#         X_ALL = np.array(X_ALL)
#         Y_ALL = np.array(Y_ALL)
#         X_PRED_ALL = np.array(X_PRED_ALL)
#
#         predictions_list:List[CustomerPredictions]= []
#         #if no customers subscribed yet
#         if len(X_ALL) == 0 or len(Y_ALL) == 0 or len(X_PRED_ALL) == 0:
#             dispatcher.send(signal=signals.COMP_USAGE_EST, msg=predictions_list)
#         else:
#             #predict on all in batch
#             with self.graph.as_default():
#                 preds = self.model.predict(X_PRED_ALL)
#             preds = [scalers[i].inverse_transform(d.reshape(-1,1)).flatten() for i,d in enumerate(preds)]
#             # and storing / unpacking all batched predictions
#             for i, p in enumerate(preds):
#                 p = p / 1000
#                 obj_ = CustomerPredictions(name=cust_row_map[i], predictions=p, first_ts=now)
#                 obj_ = self._convert_from_whole_population(obj_)
#                 self.store_predictions(obj_.customer_name, p)
#                 predictions_list.append(obj_)
#
#             for i in range(cfg.DEMAND_FORECAST_DISTANCE):
#                 log.info("Usage prediced: TIMESLOT {} -- USAGE {}".format(now+i,np.array(preds)[:,i].sum()))
#
#             #after all customers have been predicted, one message is spread
#             dispatcher.send(signal=signals.COMP_USAGE_EST, msg=predictions_list)
#
#             #learn on all in batch
#             log.info("starting learning on customer data")
#             with self.graph.as_default():
#                 self.model.fit(X_ALL, Y_ALL)
#             log.info("learning completed")
#
#
#     def store_predictions(self, customer_name: str, predictions: np.array):
#         """Stores all new predictions in the memory. This let's us compare predictions and real values later."""
#         for i, ts in enumerate(range(self.current_timeslot+1, self.current_timeslot+1+24)):
#             if customer_name not in self.predictions:
#                 self.predictions[customer_name] = {}
#             if ts not in self.predictions[customer_name]:
#                 self.predictions[customer_name][ts] = []
#             self.predictions[customer_name][ts].append(predictions[i])
#
#     def _ensure_all_there(self, tss:np.array):
#         """Ensures that all the usages are recorded for all customers for the given tss"""
#         for c in self.usages:
#             there = np.array(list(self.usages[c].keys()))
#             mask = np.in1d(tss, there, invert=True)
#             missing = tss[mask]
#             for ts in missing:
#                 #any missing set to 0
#                 self.usages[c][ts] = 0
#
#
#
#
#
# class CustomerPredictions:
#     """Holds a 24 hour set of predictions"""
#     def __init__(self, name, predictions, first_ts):
#         self.customer_name = name
#         self.first_ts = first_ts
#         tss = [i for i in range(first_ts, first_ts + len(predictions))]
#
#         #predictions. Set them in mWh here, even though PBTariffTransaction reports them in kWh
#         self.predictions:Dict[int, float] = {}
#         for i, ts in enumerate(tss):
#             self.predictions[ts] = predictions[i]


