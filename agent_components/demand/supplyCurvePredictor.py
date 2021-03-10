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
SUPPLY_MODELS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\past-saved-data\\model_supply4_ckpts"
SUPPLY_SCALERS_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\scalers\\supplycurvescalers4"
SUPPLY_SINGLE_CSV_FOLDER = "D:\\Users\\X\\Desktop\\FYP\\codes\\analysis_data\\single_csv\\supply_curve4"

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
        self.models = {}
        self.scalers = {}
        self.y_vars = set(["BsplineCoeff-{}".format(i) for i in range(100, 1600, 100)] + ["BsplineCoeff-0-{}".format(i) for i in range(3)] + ["BsplineCoeff-1600-{}".format(i) for i in range(3)])
        self.population = {}

        csv_path = ''
        for (modeltype, customerType, modelPath) in listAvailableModels(SUPPLY_MODELS_FOLDER, modelType).values():
            single_csv_path = os.path.join(SUPPLY_SINGLE_CSV_FOLDER, f"{customerType}.csv")
            csv_path = single_csv_path
            self.models[customerType] = load_model_init(modelPath, single_csv_path, list(self.y_vars))

        self.columnNames = [col for col in pd.read_csv(csv_path).columns if col not in self.y_vars]
        self.predColumnNames = [col for col in pd.read_csv(csv_path).columns if col in self.y_vars]
        for customerType, customerScalers in listAvailableScalers(SUPPLY_SCALERS_FOLDER, self.columnNames).items():
            self.scalers[customerType] = {}
            for (scaler_col, _customerType, scalerPath) in customerScalers.values():
                self.scalers[_customerType][scaler_col] = joblib.load(scalerPath)
        print("Finishing loading SupplyCurvePredictor models")

    def subscribe(self):
        dispatcher.connect(self.handle_weather_forecast_event, signals.PB_WEATHER_FORECAST)
        log.info("SupplyCurvePredictor is listening!")

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_weather_forecast_event, signals.PB_WEATHER_FORECAST)

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
            row = self._transform_row_with_scalers(row, future_ts)
            row = np.array([row])
            bsplineCoefficients = self.models[str(future_ts)].predict(row)[1][0]
            bspline_dict = {}
            for col_idx, col in enumerate(self.predColumnNames):
                bspline_dict[col] = bsplineCoefficients[col_idx]
            bsplineCoefficients_data = (future_ts, msg.currentTimeslot, bspline_dict)
            dispatcher.send(signal=signals.SUPPLY_EST, msg=bsplineCoefficients_data)
            print(f"Supply Est Prepared: Future Timeslot {future_ts}: Current Timeslot: {msg.currentTimeslot}")


    def _getCustomerType(self, customerName):
        return ''.join([i for i in customerName if i.isalpha()]).lower()

    def _transform_row_with_scalers(self, row, future_ts):
        for idx, col in enumerate(self.columnNames):
            row[idx] = self.scalers[str(future_ts)][col].transform(np.array([row[idx]]).reshape(1, -1))[0][0]
        return row

    def _prepare_input_data(self, ts, pred_weather_row):
        row = [ts]
        row += pred_weather_row
        return row

