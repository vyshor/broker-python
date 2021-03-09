import logging
from typing import List, Dict
import tensorflow as tf

import numpy as np
from pydispatch import dispatcher
from sklearn.preprocessing import MinMaxScaler

import communication.pubsub.signals as signals
import util.config as cfg
from agent_components.demand.learning.data import sequence_for_usages
from communication.grpc_messages_pb2 import PBCustomerBootstrapData, PBSimEnd, PBTariffTransaction, PBTimeslotComplete, \
    PBTxType, PBWeatherReport, PBCompetition, PBWeatherForecast
from communication.pubsub.SignalConsumer import SignalConsumer

log = logging.getLogger(__name__)

class ReadConsumer(SignalConsumer):
    def __init__(self):
        super().__init__()
        self.customer_info = {}
        self.scalers = {}  # scalers can also be looked up via customer name. They scale the customer data
        self.usages: Dict[int, Dict[int,float]] = {}  # map of maps. first map --> customer_name, second map --> timeslot ID
        self.customer_counts = {} #map that stores the number of customers per customer_name
        self.customer_populations = {}
        self.updated = set()
        self.current_timeslot = 0
        self.predictions = {}  # customers -> timeslots -> 24x predictions
        self.weatherPredictions = {}
        log.info("Initialising ReadConsumer")

    def subscribe(self):
        """Subscribes this object to the events of interest to the estimator"""
        dispatcher.connect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.connect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.connect(self.handle_sim_end, signals.PB_SIM_END)
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.connect(self.handle_weather_report_event, signals.PB_WEATHER_REPORT)
        dispatcher.connect(self.handle_weather_forecast_event, signals.PB_WEATHER_FORECAST)
        dispatcher.connect(self.handle_competition_event, signals.PB_COMPETITION)
        log.info("estimator is listening!")

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.disconnect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.disconnect(self.handle_sim_end, signals.PB_SIM_END)
        dispatcher.disconnect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.disconnect(self.handle_weather_report_event, signals.PB_WEATHER_REPORT)
        dispatcher.disconnect(self.handle_weather_forecast_event, signals.PB_WEATHER_FORECAST)
        dispatcher.disconnect(self.handle_competition_event, signals.PB_COMPETITION)

    def handle_competition_event(self, sender, signal: str, msg: PBCompetition):
        return
        print(msg)
        for customer in msg.customers:
            print(customer.powerType.label)
            print(customer.controllableKW)
            print(customer.storageCapacity)
            if hasattr(customer, "multiContracting"):
                print(customer.multiContracting)

    def handle_weather_forecast_event(self, sender, signal:str, msg: PBWeatherForecast):
        return
        # self.weatherPredictions[msg.currentTimeslot] = []
        # for prediction in msg.predictions:
        #     self.weatherPredictions[msg.currentTimeslot].append(
        #         (prediction.temperature, prediction.windSpeed, prediction.windDirection, prediction.cloudCover)
        #     )
        # print(self.weatherPredictions)
    def handle_weather_report_event(self, sender, signal: str, msg: PBWeatherReport):
        return
        print(msg)

    def handle_tariff_transaction_event(self, sender, signal: str, msg: PBTariffTransaction):
        return
        print(msg)

    def handle_customer_change(self, msg: PBTariffTransaction):
        return
        print(msg)

    def handle_customer_bootstrap_data_event(self, sender, signal: str, msg: PBCustomerBootstrapData):
        # return
        name = msg.customerName
        print(name)

    def handle_timeslot_complete(self, sender, signal: str, msg: PBTimeslotComplete):
        print(msg)

    def handle_sim_end(self, sender, signal: str, msg: PBSimEnd):
        print(msg)

    def handle_usage(self, tx: PBTariffTransaction):
        return
        print(tx)
