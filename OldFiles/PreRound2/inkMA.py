from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState
import json
import numpy as np

class Trader:
    def resin_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "RAINFOREST_RESIN"
        FAIR_VALUE = 10000
        LIMIT = 50
        WINDOW_SIZE = 10
        MIN_LOCKOUTS = 2

        position = state.position.get(product, 0)
        orders: List[Order] = []

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        resin_data = trader_data.get("resin_lockout_data", {})
        lockout_window = resin_data.get("lockout", [])[-WINDOW_SIZE:]
        lockout_window.append(abs(position) == LIMIT)
        if len(lockout_window) > WINDOW_SIZE:
            lockout_window.pop(0)

        lockout_count = sum(lockout_window)

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        buy_vol = LIMIT - position
        sell_vol = LIMIT + position

        max_buy_price = FAIR_VALUE - 1 if position > LIMIT * 0.5 else FAIR_VALUE
        min_sell_price = FAIR_VALUE + 1 if position < -LIMIT * 0.5 else FAIR_VALUE

        for price, volume in sell_orders:
            if buy_vol > 0 and price <= max_buy_price:
                qty = min(buy_vol, -volume)
                orders.append(Order(product, price, qty))
                print(f"[TAKE] {product} {qty} @ {price}")
                buy_vol -= qty

        for price, volume in buy_orders:
            if sell_vol > 0 and price >= min_sell_price:
                qty = min(sell_vol, volume)
                orders.append(Order(product, price, -qty))
                print(f"[TAKE] {product} {-qty} @ {price}")
                sell_vol -= qty

        if lockout_count >= MIN_LOCKOUTS:
            print("Readjusting Liquidity Provision")
            price_delta = max(0, 2 - (lockout_count - MIN_LOCKOUTS))
            liq_buy_price = FAIR_VALUE - price_delta
            liq_sell_price = FAIR_VALUE + price_delta

            if buy_vol > 0:
                orders.append(Order(product, liq_buy_price, buy_vol // 2))
                print(f"[LIQ ADJ] {product} {buy_vol//2} @ {liq_buy_price}")
            if sell_vol > 0:
                orders.append(Order(product, liq_sell_price, -sell_vol // 2))
                print(f"[LIQ ADJ] {product} {-sell_vol//2} @ {liq_sell_price}")

        if buy_vol > 0:
            best_bid = max(buy_orders, key=lambda x: x[1])[0] if buy_orders else FAIR_VALUE - 1
            orders.append(Order(product, min(max_buy_price, best_bid + 1), buy_vol))
            print(f"[QUOTE] {product} {buy_vol} @ {min(max_buy_price, best_bid + 1)}")

        if sell_vol > 0:
            best_ask = min(sell_orders, key=lambda x: x[1])[0] if sell_orders else FAIR_VALUE + 1
            orders.append(Order(product, max(min_sell_price, best_ask - 1), -sell_vol))
            print(f"[QUOTE] {product} {-sell_vol} @ {max(min_sell_price, best_ask - 1)}")

        self.resin_lockout_data = {"resin_lockout_data": lockout_window}
        return orders

    def kelp_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> List[Order]:
        product = "KELP"
        LIMIT = 50
        WINDOW_SIZE = 10
        MIN_LOCKOUTS = 2

        position = state.position.get(product, 0)
        orders: List[Order] = []

        try:
            kelp_data = trader_data.get("kelp_data_out", {})
            past_prices = kelp_data.get("kelp_past_prices", [])
            lockout_window = kelp_data.get("kelp_lockout", [])[-WINDOW_SIZE:]
        except:
            past_prices, lockout_window = [], []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        past_prices.append(mid_price)
        lockout_window.append(abs(position) == LIMIT)
        if len(past_prices) > 5:
            past_prices = past_prices[-5:]
        if len(lockout_window) > WINDOW_SIZE:
            lockout_window.pop(0)

        spread = best_ask - best_bid

        bid_vol = sum([v for _, v in order_depth.sell_orders.items()])
        ask_vol = sum([-v for _, v in order_depth.buy_orders.items()])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

        if len(past_prices) < 5:
            FAIR_VALUE = mid_price
        else:
            coeffs = np.array([ 0.15251124,  0.15576791,  0.17976303,  0.21362462,  0.29735446,
                -0.00146693, -0.01186836])
            intercept = 1.9887312933824433
            input_features = list(past_prices[-5:]) + [spread, imbalance]
            FAIR_VALUE = np.dot(coeffs, input_features) + intercept

        print(f"{product}: predicted FV: {FAIR_VALUE}")

        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        buy_vol = LIMIT - position
        sell_vol = LIMIT + position

        max_buy_price = FAIR_VALUE - 1 if position > LIMIT * 0.5 else FAIR_VALUE
        min_sell_price = FAIR_VALUE + 1 if position < -LIMIT * 0.5 else FAIR_VALUE

        for price, volume in sell_orders:
            if buy_vol > 0 and price <= max_buy_price:
                qty = min(buy_vol, -volume)
                orders.append(Order(product, price, qty))
                print(f"[TAKE] {product} {qty} @ {price}")
                buy_vol -= qty

        for price, volume in buy_orders:
            if sell_vol > 0 and price >= min_sell_price:
                qty = min(sell_vol, volume)
                orders.append(Order(product, price, -qty))
                print(f"[TAKE] {product} {-qty} @ {price}")
                sell_vol -= qty

        if sum(lockout_window) >= MIN_LOCKOUTS:
            print("Readjusting Liquidity Provision")
            price_delta = max(0, 2 - (sum(lockout_window) - MIN_LOCKOUTS))
            liq_buy_price = FAIR_VALUE - price_delta
            liq_sell_price = FAIR_VALUE + price_delta

            if buy_vol > 0:
                orders.append(Order(product, liq_buy_price, buy_vol // 2))
                print(f"[LIQ ADJ] {product} {buy_vol//2} @ {liq_buy_price}")
            if sell_vol > 0:
                orders.append(Order(product, liq_sell_price, -sell_vol // 2))
                print(f"[LIQ ADJ] {product} {-sell_vol//2} @ {liq_sell_price}")

        if buy_vol > 0:
            best_bid = max(buy_orders, key=lambda x: x[1])[0] if buy_orders else int(FAIR_VALUE - 1)
            orders.append(Order(product, min(max_buy_price, best_bid + 1), buy_vol))
            print(f"[QUOTE] {product} {buy_vol} @ {min(max_buy_price, best_bid + 1)}")

        if sell_vol > 0:
            best_ask = min(sell_orders, key=lambda x: x[1])[0] if sell_orders else int(FAIR_VALUE + 1)
            orders.append(Order(product, max(min_sell_price, best_ask - 1), -sell_vol))
            print(f"[QUOTE] {product} {-sell_vol} @ {max(min_sell_price, best_ask - 1)}")

        self.kelp_data_out = {
            "kelp_past_prices": past_prices,
            "kelp_lockout": lockout_window
        }

        return orders

    def ink_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict):
        product = "SQUID_INK"
        LIMIT = 50
        MA_WINDOW = 10
        DEVIATION_THRESHOLD = 4

        position = state.position.get(product, 0)
        orders: List[Order] = []

        buy_orders = sorted(order_depth.sell_orders.items())
        sell_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders

        mid_price = (best_bid + best_ask) / 2

        ink_data = trader_data.get("ink_past_prices", {})
        past_prices = ink_data.get("ink_past_prices", [])
        past_prices.append(mid_price)
        if len(past_prices) > MA_WINDOW:
            past_prices = past_prices[-MA_WINDOW:]

        if len(past_prices) < MA_WINDOW:
            self.ink_data_out = {"ink_past_prices": past_prices}
            return orders

        moving_avg = sum(past_prices) / len(past_prices)
        deviation = mid_price - moving_avg

        print(f"INK: mid={mid_price:.2f}, MA={moving_avg:.2f}, deviation={deviation:.2f}")

        if deviation >= DEVIATION_THRESHOLD and position > -LIMIT:
            sell_price = best_bid
            vol = min(order_depth.buy_orders[sell_price], position + LIMIT)
            if vol > 0:
                orders.append(Order(product, sell_price, -vol))
                position -= vol
                print(f"INK: SELL {vol} @ {sell_price} (high deviation)")

        if deviation <= -DEVIATION_THRESHOLD and position < LIMIT:
            buy_price = best_ask
            vol = min(-order_depth.sell_orders[buy_price], LIMIT - position)
            if vol > 0:
                orders.append(Order(product, buy_price, vol))
                position += vol
                print(f"INK: BUY {vol} @ {buy_price} (low deviation)")

        self.ink_data_out = {"ink_past_prices": past_prices}
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

        traderData = json.dumps({
            "kelp_data_out": getattr(self, "kelp_data_out", {}),
            "resin_lockout_data": getattr(self, "resin_lockout_data", {}),
            "ink_past_prices": getattr(self, "ink_data_out", {})
        })

        return result, conversions, traderData
