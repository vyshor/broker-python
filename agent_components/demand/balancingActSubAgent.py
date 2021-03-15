from google.protobuf.json_format import MessageToJson, Parse
from pydispatch import dispatcher
from typing import Iterable
import numpy as np
import datetime
from collections import deque
import operator

import util.config as cfg
from communication.grpc_messages_pb2 import PBTariffSpecification, PBTariffRevoke, PBOrder, PBTariffTransaction, PBTxType, PBRate, PBCompetition, PBTimeslotComplete, PBMarketTransaction, PBClearedTrade
from communication.powertac_communication_server import submit_service
from communication.pubsub import signals
from communication.pubsub.SignalConsumer import SignalConsumer
from util import id_generator


import logging
log = logging.getLogger(__name__)

class BalancingActSubAgent(SignalConsumer):
    '''
    Role of BalancingActSubAgent
    1. Subscribe to customer subscription - D
    2. Subscribe to customer predicted demand - D
    3. End of each ts, based on customer predicted demand, ensure sufficient load for them, otherwise curb usage
    '''
    def __init__(self):
        super().__init__()
        self.predicted_usage_profile = {}
        self.obtained_load = {}
        self.subscribed_rate = {}
        self.average_rate = {}
        self.current_datetime = None
        self.current_ts = 360
        self.has_spare_load = True
        self.spare_load = np.array([0] * 24)
        self.elbow_cols = ["ElbowPrice-{}".format(i) for i in range(24)] + ["ElbowQty-{}".format(i) for i in range(24)]
        print("BalancingActSubAgent is ready")

    def subscribe(self):
        dispatcher.connect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.connect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.connect(self.handle_existing_customer_usage, signals.EXISTING_USAGE_EST)
        dispatcher.connect(self.handle_competition_event, signals.PB_COMPETITION)
        dispatcher.connect(self.handle_elbow_estimation, signals.ELBOW_EST)
        dispatcher.connect(self.handle_completed_sales, signals.PB_MARKET_TRANSACTION)
        log.info("BalancingActSubAgent is listening")

    def unsubscribe(self):
        dispatcher.disconnect(self.handle_tariff_transaction_event, signals.PB_TARIFF_TRANSACTION)
        dispatcher.disconnect(self.handle_timeslot_complete, signals.PB_TIMESLOT_COMPLETE)
        dispatcher.disconnect(self.handle_existing_customer_usage, signals.EXISTING_USAGE_EST)
        dispatcher.disconnect(self.handle_competition_event, signals.PB_COMPETITION)
        dispatcher.disconnect(self.handle_elbow_estimation, signals.ELBOW_EST)
        dispatcher.disconnect(self.handle_completed_sales, signals.PB_MARKET_TRANSACTION)

    def handle_competition_event(self, sender, signal: str, msg: PBCompetition):
        self.current_datetime = datetime.datetime.fromtimestamp(msg.simulationBaseTime/1000.0) + datetime.timedelta(days=15)

    def handle_existing_customer_usage(self, sender, signal: str, msg: tuple):
        customerName, ts, usage_profile = msg
        if customerName not in self.predicted_usage_profile:
            self.predicted_usage_profile[customerName] = {}
        self.predicted_usage_profile[customerName][ts] = usage_profile

    def handle_tariff_transaction_event(self, sender, signal: str, msg: PBTariffTransaction):
        customerName = msg.customerInfo.name
        if msg.txType is PBTxType.Value("SIGNUP"):
            # 1. Set obtain load
            # 2. Extract rate information
            self.subscribed_rate[customerName] = self._extract_rates_from_spec(msg.tariffSpec.rates)
            self.average_rate[customerName] = np.mean(self.subscribed_rate[customerName])
            self.obtained_load[customerName] = deque([0] * 24)
            print(f"Customer {customerName} has subscribed")
        elif msg.txType is PBTxType.Value("WITHDRAW"):
            if customerName in self.obtained_load:
                # 1. Redistribute the reserved load to next most expensive customer
                # 2. Clear the data
                del self.average_rate[customerName]
                self.spare_load += np.array(self.obtained_load[customerName])
                self.has_spare_load = True
                del self.obtained_load[customerName]
            else:
                print(f"Missing customer {customerName}: Cannot withdraw subscription")

    def _first_timeslot_bid_strat(self, ts, future_ts_translation, prices, qtys):
        target_ts = ts + future_ts_translation
        self.send_order(target_ts, qtys[0], -1 * prices[0])

    def _24h_timeslot_bid_strat(self, ts, future_ts_translation, prices, qtys):
        target_ts = ts + future_ts_translation
        self.send_order(target_ts, qtys[0], -1 * prices[0])

    def handle_completed_sales(self,  sender, signal: str, msg: PBMarketTransaction):
        ts = msg.timeslot
        mWh = msg.mWh
        price = msg.price
        print(f"{ts} Bought load {mWh} at {price}")
        if ts > self.current_ts:
            self.spare_load[ts - self.current_ts] += mWh
            self.has_spare_load = True

    def handle_elbow_estimation(self, sender, signal: str, msg: tuple):
        future_ts_translation, ts, elbow_dict = msg
        # if ts not in self.elbow:
        #     self.elbow[ts] = {}
        price_qty = []
        for col in self.elbow_cols:
            price_qty.append(elbow_dict[col])
        prices, qtys = np.array(price_qty[:24]), np.array(price_qty[24:])
        if ts == 360:
            self._first_timeslot_bid_strat(ts, future_ts_translation, prices, qtys)
        elif future_ts_translation == 24:
            self._24h_timeslot_bid_strat(ts, future_ts_translation, prices, qtys)

    def _extract_rates_from_spec(self, rates: Iterable[PBRate]):
        return [rate.minValue for rate in rates]

    def send_order(self, ts, mWh, limitPrice):
        order = PBOrder(broker=cfg.ME, timeslot=ts, mWh=mWh, limitPrice=limitPrice)
        print(f"Offer TS: {ts} MWh: {mWh} Price: {limitPrice}")
        dispatcher.send(signals.OUT_PB_ORDER, msg=order)

    def _reconcilation_of_demand(self, ts):
        customer_demand = {}
        for customerName in self.obtained_load.keys():
            if customerName in self.predicted_usage_profile:
                pred_for_next_24h = None
                for i in range(5):
                    if ts-i in self.predicted_usage_profile[customerName]:
                        pred_for_next_24h = self.predicted_usage_profile[customerName][i:24+i]
                        break
                if pred_for_next_24h is None:
                    continue
                customer_demand[customerName] = pred_for_next_24h

        net_surplus = []

        if self.has_spare_load and len(self.obtained_load) > 0:
            print(f"Excess load of {self.obtained_load}")
            next_highest_rate_customer = max(self.average_rate.items(), key=operator.itemgetter(1))[0]
            for i in range(24):
                self.obtained_load[next_highest_rate_customer][i] += self.spare_load[i]
                self.spare_load[i] = 0
            self.has_spare_load = False

        for future_ts in range(0, 24):
            surplus_customers = []
            deficit_customers = []
            total_surplus = 0
            total_deficit = 0

            for customerName, demand_24 in customer_demand.items():
                datetime_idx = (self.current_datetime.weekday() * 24) + self.current_datetime.hour
                charged_rate = self.subscribed_rate[customerName][datetime_idx] * 1000 # Subscribe_rate is price per KWh , while usage is in price per MWh
                surplus = self.obtained_load[customerName][future_ts] - demand_24[future_ts]
                if surplus > 0:
                    total_surplus += surplus
                    surplus_customers.append((surplus, charged_rate, customerName))
                else:
                    total_deficit -= surplus
                    deficit_customers.append((charged_rate, surplus, customerName))

            net_surplus.append(total_surplus + total_deficit)

            surplus_customers.sort(reverse=True)
            deficit_customers.sort(reverse=True)
            for excess_amount, _, surplusCustomerName in surplus_customers:
                if total_deficit < -0.0001:
                    for deficit_idx, (_, deficit_amount, deficitCustomerName) in deficit_customers:
                        if excess_amount > 0.0001:
                            give_amount = min(excess_amount, abs(deficit_amount))
                            deficit_customers[deficit_idx][1] = deficit_amount + give_amount
                            total_surplus -= give_amount
                            total_deficit += give_amount

                            # Update the original allocation
                            self.obtained_load[surplusCustomerName][future_ts] -= give_amount
                            self.obtained_load[deficitCustomerName][future_ts] += give_amount
                        else:
                            break
                else:
                    break

            if future_ts == 0:
                # If there is no deficit - For surplus, just keep until last ts and sell
                if total_surplus + total_deficit > 0:
                    sell_amount = total_surplus + total_deficit
                    self.send_order(self.current_ts+future_ts+1, -1 * sell_amount, None) # Neg amount, Positive Rate
                # If there is deficit, it is too late and normally too expensive to buy, thus necessary to curb usage
                else:
                    # TODO: Exercise Economic Control
                    pass
            elif future_ts != 23: # 24 TS is handled by wholesale bought purchase
                # Buy more for customers in deficit - As there is still time
                for (charged_rate, deficit_amount, deficitCustomerName) in deficit_customers:
                    self.send_order(self.current_ts+future_ts+1, -1 * deficit_amount, -1 * charged_rate)  # Positive amount, Neg Rate


    def handle_timeslot_complete(self, sender, signal: str, msg: PBTimeslotComplete):
        self.current_datetime = self.current_datetime + datetime.timedelta(hours=1)
        self.current_ts = msg.timeslotIndex + 1
        for customerName, load in self.obtained_load.items():
            load.pop_left()
            load.append(0)
        # Start periodic calculation for excess or deficit
        self._reconcilation_of_demand(self.current_ts)
