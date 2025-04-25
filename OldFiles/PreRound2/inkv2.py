from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState
import json
import numpy as np

class Trader:
    def resin_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "RAINFOREST_RESIN"
        LIMIT = 50
        FAIR_VALUE = 10000

        position = state.position.get(product, 0)
        orders: List[Order] = []

        buy_orders = sorted(order_depth.sell_orders.items())
        sell_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        for ask_price, ask_vol in buy_orders:
            if ask_price < FAIR_VALUE and position < LIMIT:
                vol = min(-ask_vol, LIMIT - position)
                orders.append(Order(product, ask_price, vol))
                position += vol

        for bid_price, bid_vol in sell_orders:
            if bid_price > FAIR_VALUE and position > -LIMIT:
                vol = min(bid_vol, position + LIMIT)
                orders.append(Order(product, bid_price, -vol))
                position -= vol

        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid

            if spread >= 2:
                midprice = int((best_bid + best_ask) / 2) 
                buy_price = midprice - 3
                sell_price = midprice + 3
                if midprice < 10000 - 1:
                    buy_price = midprice
                if midprice > 10000 + 1:
                    sell_price = midprice

                our_spread = abs(sell_price - buy_price)

                buy_vol = 14
                sell_vol = 14
                if buy_vol > LIMIT - position:
                    buy_vol = 7
                    if buy_vol > LIMIT - position:
                        buy_vol = LIMIT - position
                if sell_vol > position + LIMIT:
                    sell_vol = 7
                    if sell_vol > position + LIMIT:
                        sell_vol = LIMIT + position

                if buy_vol > 0:
                    orders.append(Order(product, buy_price, int(buy_vol)))
                if sell_vol > 0:
                    orders.append(Order(product, sell_price, -int(sell_vol)))

            THRESHOLD = 35
            MIN_TARGET_BUY = LIMIT - buy_vol
            MIN_TARGET_SELL = LIMIT - sell_vol
            MAX_DEVIATION = 0

            if abs(position) >= THRESHOLD:
                excess_buy = abs(position) - MIN_TARGET_SELL
                if position > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid >= FAIR_VALUE - MAX_DEVIATION:
                        sell_qty = min(excess_buy, best_bid_volume)
                        if sell_qty > 0:
                            orders.append(Order(product, best_bid, -sell_qty))
                            position -= sell_qty

                elif position < 0:
                    excess_sell = abs(position) - MIN_TARGET_BUY
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = -order_depth.sell_orders[best_ask]
                    if best_ask <= FAIR_VALUE + MAX_DEVIATION:
                        buy_qty = min(excess_sell, best_ask_volume)
                        if buy_qty > 0:
                            orders.append(Order(product, best_ask, buy_qty))
                            position += buy_qty

        return orders

    def kelp_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> List[Order]:
        product = "KELP"
        LIMIT = 50

        position = state.position.get(product, 0)
        orders: List[Order] = []

        buy_orders = sorted(order_depth.sell_orders.items())
        sell_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        kelp_data = trader_data.get("KELP", {})
        past_prices = kelp_data.get("past_prices", [])

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        mid_price = int(best_bid + best_ask) / 2
        spread = best_ask - best_bid

        bid_vol = sum([v for _, v in buy_orders])
        ask_vol = sum([-v for _, v in sell_orders])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

        past_prices.append(mid_price)

        if len(past_prices) < 5:
            FAIR_VALUE = mid_price
        else:
            coeffs = np.array([ 0.15251124,  0.15576791,  0.17976303,  0.21362462,  0.29735446,
        -0.00146693, -0.01186836])
            intercept = 1.9887312933824433
            input_features = list(past_prices[-5:]) + [spread, imbalance]
            FAIR_VALUE = np.dot(coeffs, input_features) + intercept

        print(f"{product}: predicted FV: {FAIR_VALUE}")

        for ask_price, ask_vol in buy_orders:
            if ask_price < FAIR_VALUE and position < LIMIT:
                vol = min(-ask_vol, LIMIT - position)
                orders.append(Order(product, ask_price, vol))
                position += vol

        for bid_price, bid_vol in sell_orders:
            if bid_price > FAIR_VALUE and position > -LIMIT:
                vol = min(bid_vol, position + LIMIT)
                orders.append(Order(product, bid_price, -vol))
                position -= vol

        if spread >= 2:
          buy_price = int(FAIR_VALUE - 2)
          sell_price = int(FAIR_VALUE + 2)
          if position < LIMIT:
              orders.append(Order(product, buy_price, min(25, LIMIT - position)))
          if position > -LIMIT:
              orders.append(Order(product, sell_price, -min(25, position + LIMIT)))

        self.kelp_data_out = {
            "KELP": {
                "past_prices": past_prices,
            }
        }

        return orders

    def ink_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict):
        product = "SQUID_INK"
        LIMIT = 50
        orders: List[Order] = []
        position = state.position.get(product, 0)

        ink_data = trader_data.get("INK", {})
        depletion_timestamps = ink_data.get("depletion_timestamps", [])
        normalized_time = state.timestamp / 100

        buy_orders = sorted(order_depth.sell_orders.items())
        sell_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_ask_volume = order_depth.sell_orders[best_ask] if best_ask else 0
        best_bid_volume = order_depth.buy_orders[best_bid] if best_bid else 0

        if best_bid_volume > 8:
            depletion_timestamps.append(normalized_time)

        intensity = sum(np.exp(-(normalized_time - t)) for t in depletion_timestamps if normalized_time - t >= 0)

        risk_threshold = 1.5

        if (position == 50 or intensity >= risk_threshold):
          for bid_price, bid_vol in sell_orders:
              if position > -LIMIT:
                  vol = bid_vol
                  orders.append(Order(product, bid_price, -vol))
                  print(f"{product} SHORTING {vol} @ {bid_price}")
                  position -= vol

        if position < LIMIT :
          for ask_price, ask_vol in buy_orders:
              if position < LIMIT:
                  vol = ask_vol
                  orders.append(Order(product, ask_price, vol))
                  print(f"{product} COVERING SHORT {vol} @ {ask_price}")
                  position += vol

        self.kelp_data_out["INK"] = {"depletion_timestamps": depletion_timestamps[-10:]}

        return orders

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        for product in state.order_depths:
            if product == "RAINFOREST_RESIN":
                result[product] = self.resin_strategy(state, state.order_depths[product])
            elif product == "KELP":
                result[product] = self.kelp_strategy(state, state.order_depths[product], trader_data)
            elif product == "SQUID_INK":
                result[product] = self.ink_strategy(state, state.order_depths[product], trader_data)

        traderData = json.dumps(self.kelp_data_out)
        return result, conversions, traderData
