from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState
import json

class Trader:
    def resin_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        return []

    def kelp_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> List[Order]:
        product = "KELP"
        LIMIT = 50
        WINDOW_SIZE = 10
        MIN_LOCKOUTS = 2

        orders: List[Order] = []
        position = state.position.get(product, 0)

        # --- Current market snapshot ---
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        if not buy_orders or not sell_orders:
            return orders

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        # Order flow imbalance
        bid_vol = sum([v for _, v in buy_orders])
        ask_vol = sum([-v for _, v in sell_orders])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

        # --- Get traderData ---
        kelp_data = trader_data.get("KELP", {})
        past_prices = kelp_data.get("past_prices", [])
        lockout_window = kelp_data.get("lockout", [])

        # --- Update past prices ---
        past_prices.append(mid_price)
        if len(past_prices) > 5:
            past_prices = past_prices[-5:]

        # --- Predict mid price ---
        if len(past_prices) < 5:
            predicted_price = mid_price  # fallback
        else:
            coeffs = [0.15118617, 0.15507352, 0.17414473, 0.22756946, 0.29089259, 0.02699926, -0.8650637]
            inputs = past_prices[-5:] + [spread, imbalance]
            predicted_price = sum(c * x for c, x in zip(coeffs, inputs))

        # --- Update lockout window ---
        lockout_window.append(abs(position) == LIMIT)
        if len(lockout_window) > WINDOW_SIZE:
            lockout_window.pop(0)

        buy_vol = LIMIT - position
        sell_vol = LIMIT + position

        max_buy_price = predicted_price - 1 if position > LIMIT * 0.5 else predicted_price
        min_sell_price = predicted_price + 1 if position < -LIMIT * 0.5 else predicted_price

        # --- Market taking ---
        for price, volume in sell_orders:
            if buy_vol > 0 and price <= max_buy_price:
                qty = min(buy_vol, -volume)
                orders.append(Order(product, price, qty))
                buy_vol -= qty

        for price, volume in buy_orders:
            if sell_vol > 0 and price >= min_sell_price:
                qty = min(sell_vol, volume)
                orders.append(Order(product, price, -qty))
                sell_vol -= qty

        # --- Adaptive forced liquidation ---
        lockout_count = sum(lockout_window)
        if lockout_count >= MIN_LOCKOUTS:
            price_delta = max(0, 2 - (lockout_count - MIN_LOCKOUTS))
            liq_buy_price = predicted_price - price_delta
            liq_sell_price = predicted_price + price_delta

            if buy_vol > 0:
                orders.append(Order(product, int(liq_buy_price), buy_vol // 2))
            if sell_vol > 0:
                orders.append(Order(product, int(liq_sell_price), -sell_vol // 2))

        # --- Passive quoting ---
        if buy_vol > 0:
            orders.append(Order(product, int(predicted_price - 3), buy_vol))
        if sell_vol > 0:
            orders.append(Order(product, int(predicted_price + 3), -sell_vol))

        self.kelp_data_out = {
            "KELP": {
                "past_prices": past_prices,
                "lockout": lockout_window
            }
        }

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

        traderData = json.dumps(getattr(self, "kelp_data_out", trader_data))
        return result, conversions, traderData
