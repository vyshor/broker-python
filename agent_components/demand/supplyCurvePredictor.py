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
LINEAR_A_MODELS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\past-saved-data\\model_linearA_ckpts"
LINEAR_B_MODELS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\past-saved-data\\model_linearB_ckpts"
ELBOW_MODELS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\past-saved-data\\model_elbow_ckpts"
LINEAR_SCALERS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\scalers\\linearscalers"
ELBOW_SCALERS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\scalers\\elbowscalers"
LINEAR_SINGLE_CSV_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\analysis_data\\single_csv\\linear"
ELBOW_SINGLE_CSV_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\analysis_data\\single_csv\\elbow"

class SupplyCurvePredictor(SignalConsumer):
    """
    Predictor class that loads models and outputs
    1. Loads model and scalers per customer type - D
    2. Subscription of weather Prediction Data - D
    3. Publish supply curve for future ts - D
    """

    def __init__(self, modelType="DNN"):
        super().__init__()
        self.customer_info = {}
        self.models_elbow = {}
        self.models_linearA = {}
        self.models_linearB = {}
        self.scalers_elbow = {}
        self.scalers_linear = {}
        self.y_vars_elbow = set(["ElbowQty-{}".format(i) for i in range(24)] + ["ElbowPrice-{}".format(i) for i in range(24)])
        self.y_vars_linearA = set(["LinearGradA-{}".format(i) for i in range(24)] + ["LinearInterceptA-{}".format(i) for i in range(24)])
        self.y_vars_linearB = set(["LinearGradB-{}".format(i) for i in range(24)] + ["LinearInterceptB-{}".format(i) for i in range(24)])
        self.population = {}
        self.seed_total_usage = np.array([0] * (14 * 24), dtype=np.float64)
        self.seedWeather = [0] * 336
        self.seedWeatherInitialisedCount = 0
        self.seedWeatherColumns = []

        csv_path = ''
        for (modeltype, customerType, modelPath) in listAvailableModels(ELBOW_MODELS_FOLDER, modelType).values():
            single_csv_path = os.path.join(ELBOW_SINGLE_CSV_FOLDER, f"{customerType}.csv")
            csv_path = single_csv_path
            self.models_elbow[customerType] = load_model_init(modelPath, single_csv_path, list(self.y_vars_elbow))

        self.columnNames = [col for col in pd.read_csv(csv_path).columns if col not in self.y_vars_elbow and col not in self.y_vars_linearA and col not in self.y_vars_linearB]
        self.predColumnNames_elbow = [col for col in pd.read_csv(csv_path).columns if col in self.y_vars_elbow]

        # for (modeltype, customerType, modelPath) in listAvailableModels(LINEAR_A_MODELS_FOLDER, modelType).values():
        #     single_csv_path = os.path.join(LINEAR_SINGLE_CSV_FOLDER, f"{customerType}.csv")
        #     csv_path = single_csv_path
        #     self.models_linearA[customerType] = load_model_init(modelPath, single_csv_path, list(self.y_vars_linearA), excluded_cols=list(self.y_vars_linearB))
        #
        # # self.columnNames_linearA = [col for col in pd.read_csv(csv_path).columns if col not in self.y_vars_linearA]
        # self.predColumnNames_linearA = [col for col in pd.read_csv(csv_path).columns if col in self.y_vars_linearA]
        #
        # for (modeltype, customerType, modelPath) in listAvailableModels(LINEAR_B_MODELS_FOLDER, modelType).values():
        #     single_csv_path = os.path.join(LINEAR_SINGLE_CSV_FOLDER, f"{customerType}.csv")
        #     csv_path = single_csv_path
        #     self.models_linearB[customerType] = load_model_init(modelPath, single_csv_path, list(self.y_vars_linearB), excluded_cols=list(self.y_vars_linearA))
        #
        # # self.columnNames_linearB = [col for col in pd.read_csv(csv_path).columns if col not in self.y_vars_linearB]
        # self.predColumnNames_linearB = [col for col in pd.read_csv(csv_path).columns if col in self.y_vars_linearB]

        for customerType, customerScalers in listAvailableScalers(ELBOW_SCALERS_FOLDER, self.columnNames).items():
            self.scalers_elbow[customerType] = {}
            for (scaler_col, _customerType, scalerPath) in customerScalers.values():
                self.scalers_elbow[_customerType][scaler_col] = joblib.load(scalerPath)

        # for customerType, customerScalers in listAvailableScalers(LINEAR_SCALERS_FOLDER, self.columnNames).items():
        #     self.scalers_linear[customerType] = {}
        #     for (scaler_col, _customerType, scalerPath) in customerScalers.values():
        #         self.scalers_linear[_customerType][scaler_col] = joblib.load(scalerPath)
        print("Finishing loading SupplyCurvePredictor models")

    def subscribe(self):
        dispatcher.connect(self.handle_weather_forecast_event, signals.PB_WEATHER_FORECAST)
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.connect(self.handle_weather_report_event, signals.PB_WEATHER_REPORT)
        log.info("SupplyCurvePredictor is listening!")

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_weather_forecast_event, signals.PB_WEATHER_FORECAST)
        dispatcher.disconnect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.disconnect(self.handle_weather_report_event, signals.PB_WEATHER_REPORT)

    def handle_customer_bootstrap_data_event(self, sender, signal: str, msg: PBCustomerBootstrapData):
        self.seed_total_usage += np.array(msg.netUsage)

    def handle_weather_report_event(self, sender, signal:str, msg: PBWeatherReport):
        ts = msg.currentTimeslot.serialNumber
        if 24 <= ts <= 359:  # Check weather report is seed weather report
            self.seedWeather[ts - 24] = msg.temperature, msg.windSpeed, msg.windDirection, msg.cloudCover
            self.seedWeatherInitialisedCount += 1
            if self.seedWeatherInitialisedCount == 336:
                for weatherTuple in self.seedWeather:
                    self.seedWeatherColumns += list(weatherTuple)
                assert len(self.seedWeatherColumns) == 336 * 4

    def handle_weather_forecast_event(self, sender, signal:str, msg: PBWeatherForecast):
        pred_weather_row = []
        for prediction in msg.predictions:
            pred_weather_row.append(prediction.temperature)
            pred_weather_row.append(prediction.windSpeed)
            pred_weather_row.append(prediction.windDirection)
            pred_weather_row.append(prediction.cloudCover)

        # Do prediction
        for future_ts in range(1, 25):
            row = self._prepare_input_data(msg.currentTimeslot, pred_weather_row)
            row = self._transform_row_with_scalers(row, future_ts, self.scalers_elbow)
            row = np.array([row])

            for model_dict, predColumnNames, signalOutput in [(self.models_elbow, self.predColumnNames_elbow, signals.ELBOW_EST)]:
            # for model_dict, predColumnNames, signalOutput in [(self.models_elbow, self.predColumnNames_elbow, signals.ELBOW_EST), (self.models_linearA, self.predColumnNames_linearA, signals.LINEAR_A_EST), (self.models_linearB, self.predColumnNames_linearB, signals.LINEAR_B_EST)]:
                model_outputs = model_dict[str(future_ts)].predict(row)[1][0]
                output_dict = {}

                for col_idx, col in enumerate(predColumnNames):
                    output_dict[col] = model_outputs[col_idx]

                output_msg_data = (future_ts, msg.currentTimeslot, output_dict)
                dispatcher.send(signal=signalOutput, msg=output_msg_data)
            # print(f"Supply Est Prepared: Future Timeslot {future_ts}: Current Timeslot: {msg.currentTimeslot}")


    def _getCustomerType(self, customerName):
        return ''.join([i for i in customerName if i.isalpha()]).lower()

    def _transform_row_with_scalers(self, row, future_ts, scalers):
        for idx, col in enumerate(self.columnNames):
            row[idx] = scalers[str(future_ts)][col].transform(np.array([row[idx]]).reshape(1, -1))[0][0]
        return row

    def _prepare_input_data(self, ts, pred_weather_row):
        rotation_idx = (ts - 360) % 336
        row = [ts]
        print(f"Length of row1: {len(row)}")
        row += pred_weather_row
        print(f"Length of row2: {len(row)}")
        row += list(self.seed_total_usage[rotation_idx:]) + list(self.seed_total_usage[:rotation_idx])
        print(f"Length of row3: {len(row)}")
        row += self.seedWeatherColumns[rotation_idx:] + self.seedWeatherColumns[:rotation_idx]
        print(f"Length of row: {len(row)}")
        return row

