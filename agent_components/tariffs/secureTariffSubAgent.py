from google.protobuf.json_format import MessageToJson, Parse
from pydispatch import dispatcher
import numpy as np
import datetime
from collections import deque

import util.config as cfg
from communication.grpc_messages_pb2 import PBTariffSpecification, PBTariffRevoke, PBCustomerBootstrapData, PBCompetition, PBTimeslotComplete, PBCustomerInfo
from communication.powertac_communication_server import submit_service
from communication.pubsub import signals
from communication.pubsub.SignalConsumer import SignalConsumer
from util import id_generator
from scipy.interpolate import BSpline


import logging
log = logging.getLogger(__name__)

class SecureTariffSubAgent(SignalConsumer):
    def __init__(self):
        super().__init__()
        self.initial_usage_profile = {}
        self.updated_usage_profile = {}
        self.current_datetime = None
        self.total_consumption_week_profile = np.array([0] * 168)
        self.current_ts = 360
        self.bspline = {}
        self.supply_cols = ["BsplineCoeff-0-{}".format(i) for i in range(3)] + ["BsplineCoeff-{}".format(i) for i in range(100, 1600, 100)] + ["BsplineCoeff-1600-{}".format(i) for i in range(3)]
        self.supply_knots = np.array([0] * 3 + [i for i in range(100, 1600, 100)] + [1600] * 3)
        self.next_24h_total_usage = None
        self.sold_for_next_24h = deque([0] * 24)
        self.est_remaining_usage_for_next_24h = None
        self.number_of_customers = -1
        self.customer_bootstrapped = 0
        self.current_min_price = 9999999999
        self.current_max_price = 0
        self.tariff_spec_tries = 1
        self.customerTypes2customers = {}
        self.spec_info = {}
        self.HARD_LIMIT_FOR_QTY_DEMANDED = 1500
        self.qty_demanded_upperbound_factor = 0.9
        self.qty_demanded_lowerbound_factor = 0.5

    def subscribe(self):
        # dispatcher.connect(self.handle_tariff_spec, signals.PB_TARIFF_SPECIFICATION)
        # dispatcher.connect(self.handle_tariff_revoke, signals.PB_TARIFF_REVOKE)
        dispatcher.connect(self.handle_competition_event, signals.PB_COMPETITION)
        dispatcher.connect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.connect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.connect(self.handle_usage_profile, signals.COMP_USAGE_EST)
        dispatcher.connect(self.handle_utility_estimation, signals.UTILITY_EST)
        dispatcher.connect(self.handle_supply_estimation, signals.SUPPLY_EST)
        log.info("tariff publisher is listening")

    def unsubscribe(self):
        # dispatcher.disconnect(self.handle_tariff_spec, signals.PB_TARIFF_SPECIFICATION)
        # dispatcher.disconnect(self.handle_tariff_revoke, signals.PB_TARIFF_REVOKE)
        dispatcher.disconnect(self.handle_competition_event, signals.PB_COMPETITION)
        dispatcher.disconnect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.disconnect(self.handle_customer_bootstrap_data_event, signals.PB_CUSTOMER_BOOTSTRAP_DATA)
        dispatcher.disconnect(self.handle_usage_profile, signals.COMP_USAGE_EST)
        dispatcher.disconnect(self.handle_utility_estimation, signals.UTILITY_EST)
        dispatcher.disconnect(self.handle_supply_estimation, signals.SUPPLY_EST)

    def handle_competition_event(self, sender, signal: str, msg: PBCompetition):
        customers = msg.customers
        self.current_datetime = datetime.datetime.fromtimestamp(msg.simulationBaseTime/1000.0) + datetime.timedelta(days=15)
        self.number_of_customers = len(customers)
        for customer in customers:
            customerPowerType = customer.powerType.label
            if customerPowerType not in self.customerTypes2customers:
                self.customerTypes2customers[customerPowerType] = set([])
            self.customerTypes2customers[customerPowerType].add(customer.name)

    def handle_customer_bootstrap_data_event(self, sender, signal: str, msg: PBCustomerBootstrapData):
        two_weeks_ago = self.current_datetime - datetime.timedelta(days=14)
        datetime_idx = (two_weeks_ago.weekday() * 24) + two_weeks_ago.hour
        two_weeks_usage = np.array(msg.netUsage)
        for i in range(168):
            week_usage_idx = (datetime_idx + i) % 168
            self.total_consumption_week_profile[week_usage_idx] += np.mean(two_weeks_usage[i::168])
        self.customer_bootstrapped += 1
        if self.number_of_customers == self.customer_bootstrapped:
            current_datetime_idx = (self.current_datetime.weekday() * 24) + self.current_datetime.hour
            adjusted_next_24h_total_usage = self.total_consumption_week_profile[current_datetime_idx+1:current_datetime_idx+25] * np.array([i/24 for i in range(1, 25)])
            self.next_24h_total_usage = deque(adjusted_next_24h_total_usage * -1)
            self.est_remaining_usage_for_next_24h = np.array(self.next_24h_total_usage) - np.array(self.sold_for_next_24h)
            self._update_min_max_price()

    def _update_min_max_price(self):
        if self.current_ts not in self.bspline or len(self.bspline[self.current_ts]) != 24:
            return
        for ts_idx, remaining_usage in enumerate(self.est_remaining_usage_for_next_24h):
            est_qty_demanded = remaining_usage / (ts_idx+1)
            print(f"Future timeslot: {ts_idx}")
            print(f"Est Qty Demanded: {est_qty_demanded}")
            est_upper_price = self.bspline[self.current_ts][ts_idx+1](min(self.qty_demanded_upperbound_factor * est_qty_demanded, self.HARD_LIMIT_FOR_QTY_DEMANDED))
            est_lower_price = self.bspline[self.current_ts][ts_idx+1](min(self.qty_demanded_lowerbound_factor * est_qty_demanded, self.HARD_LIMIT_FOR_QTY_DEMANDED))
            print(f"Est Upper Price: {est_upper_price}")
            print(f"Est Lower Price: {est_lower_price}")
            self.current_min_price = min(est_lower_price, self.current_min_price)
            self.current_max_price = max(est_upper_price, self.current_max_price)
        self._produce_tariff_spec()

    def _produce_tariff_spec(self):
        for powerType, customers in self.customerTypes2customers.items():
            for _ in range(self.tariff_spec_tries):
                proposed_rates = np.random.uniform(self.current_min_price, self.current_max_price, size=168) * -1
                minDuration = 0
                signUpPayment = 0
                earlyWithdrawPayment = 0
                periodicPayment = 0
                spec_row = list(proposed_rates) + [minDuration, signUpPayment, earlyWithdrawPayment, periodicPayment]
                for customerName in customers:
                    spec_id = id_generator.create_id()
                    spec_tup = spec_id, customerName, spec_row
                    self.spec_info[spec_id] = spec_tup
                    dispatcher.send(signal=signals.POSS_TARIFF_SPEC, msg=spec_tup)

    def handle_usage_profile(self, sender, signal: str, msg: tuple):
        customerName, predicted_ts, usage_profile = msg
        if predicted_ts == 360:
            self.initial_usage_profile[customerName] = usage_profile
            # print(f"Received Initial Usage Profile: {customerName}")
        else:
            self.updated_usage_profile[customerName] = usage_profile
            print(f"Received Updated Usage Profile: {customerName} | Timeslot: {predicted_ts}")

    def handle_utility_estimation(self, sender, signal: str, msg: tuple):
        spec_id, customerName, utility = msg
        print(f"Received Utility Est: {utility} | Spec_id: {spec_id}")

    def handle_supply_estimation(self, sender, signal: str, msg: tuple):
        future_ts_translation, ts, bspline_dict = msg
        if ts not in self.bspline:
            self.bspline[ts] = {}
        coeffs = []
        for col in self.supply_cols:
            coeffs.append(bspline_dict[col])
        self.bspline[ts][future_ts_translation] = BSpline(self.supply_knots, np.array(coeffs), 2)
        self._update_min_max_price()

    def handle_timeslot_complete(self, sender, signal: str, msg: PBTimeslotComplete):
        self.current_datetime += datetime.timedelta(hours=1)
        future_24h_datetime_idx = ((self.current_datetime.weekday() * 24) + self.current_datetime.hour + 24) % 168
        self.current_ts = msg.timeslotIndex + 1
        self.next_24h_total_usage.popleft()
        self.next_24h_total_usage.append(self.total_consumption_week_profile[future_24h_datetime_idx] * -1)
        self.sold_for_next_24h.popleft()
        self.sold_for_next_24h.append(0)
        self.est_remaining_usage_for_next_24h = np.array(self.next_24h_total_usage) - np.array(self.sold_for_next_24h)
        print(f"New timeslot: {self.current_ts}")

# def handle_tariff_spec(self, sender, signal: str, msg: PBTariffSpecification):
    #     """Handling incoming specs. Let's just clone the babies!"""
    #     #if from our idol
    #     if msg.broker == cfg.TARIFF_CLONE_COMPETITOR_AGENT:
    #         spec = self.make_spec_mine(msg)
    #         self.clones[msg.id] = spec.id
    #         #and send it to the server as if it was ours
    #         log.info("cloning tariff !")
    #         dispatcher.send(signals.OUT_PB_TARIFF_SPECIFICATION, msg=spec)
    #
    # def make_spec_mine(self, msg)-> PBTariffSpecification:
    #     # let's clone this
    #     mine = Parse(MessageToJson(msg), PBTariffSpecification())
    #     #set the broker
    #     mine.broker = cfg.ME
    #     #set new ID
    #     mine.id = id_generator.create_id()
    #     # also set the IDs of the rates
    #     for r in mine.rates:
    #         r.tariffId = mine.id
    #     for r in mine.regulationRates:
    #         r.tariffId = mine.id
    #     return mine
    #
    #
    # def handle_tariff_revoke(self, sender, signal: str, msg: PBTariffRevoke):
    #     """if our idol revokes, let's revoke too"""
    #     if msg.broker == cfg.TARIFF_CLONE_COMPETITOR_AGENT:
    #         if msg.tariffId in self.clones:
    #             #have cloned this tariff
    #             other = msg.tariffId
    #             msg.broker = cfg.ME
    #             msg.tariffId = self.clones[other]
    #             dispatcher.send(signals.OUT_PB_TARIFF_REVOKE, msg=msg)
    #             del self.clones[other]

