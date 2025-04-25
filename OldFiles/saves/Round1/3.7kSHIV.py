from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState
import json

class Trader:

    def resin_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "RAINFOREST_RESIN"
        FAIR_VALUE = 10000
        LIMIT = 50
        WINDOW_SIZE = 10

        position = state.position.get(product, 0)
        orders: List[Order] = []

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception as e:
            trader_data = {}

        # Load past lockout data
        lockout_window = trader_data.get("lockout", [])
        lockout_window = lockout_window[-WINDOW_SIZE:]

        # Update lockout status
        lockout_window.append(abs(position) == LIMIT)
        if len(lockout_window) > WINDOW_SIZE:
            lockout_window.pop(0)

        hard_liquidate = all(lockout_window)
        soft_liquidate = sum(lockout_window) >= WINDOW_SIZE // 2 and lockout_window[-1]

        orderbook = order_depth
        buy_orders = sorted(orderbook.buy_orders.items(), reverse=True)
        sell_orders = sorted(orderbook.sell_orders.items())

        to_buy = LIMIT - position
        to_sell = LIMIT + position

        max_buy_price = FAIR_VALUE - 1 if position > LIMIT * 0.5 else FAIR_VALUE
        min_sell_price = FAIR_VALUE + 1 if position < -LIMIT * 0.5 else FAIR_VALUE

        # --- Market taking (best fills first) ---
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                orders.append(Order(product, price, qty))
                to_buy -= qty

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                orders.append(Order(product, price, -qty))
                to_sell -= qty

        # --- Soft/hard liquidation ---
        if hard_liquidate and to_buy > 0:
            orders.append(Order(product, FAIR_VALUE, to_buy // 2))
        elif soft_liquidate and to_buy > 0:
            orders.append(Order(product, FAIR_VALUE - 2, to_buy // 2))

        if hard_liquidate and to_sell > 0:
            orders.append(Order(product, FAIR_VALUE, -to_sell // 2))
        elif soft_liquidate and to_sell > 0:
            orders.append(Order(product, FAIR_VALUE + 2, -to_sell // 2))

        # --- Passive quoting ---
        if to_buy > 0:
            best_bid = max(buy_orders, key=lambda x: x[1])[0] if buy_orders else FAIR_VALUE - 2
            orders.append(Order(product, min(max_buy_price, best_bid + 1), to_buy))

        if to_sell > 0:
            best_ask = min(sell_orders, key=lambda x: x[1])[0] if sell_orders else FAIR_VALUE + 2
            orders.append(Order(product, max(min_sell_price, best_ask - 1), -to_sell))

        # Save lockout window as traderData
        self.lockout_data = {"lockout": lockout_window}
        return orders

    def kelp_strategy(self, order_depth: Dict) -> List[Order]:
        return []

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        order_depth = state.order_depths
        trader_data = {}

        for product in order_depth.keys():
            if product == "RAINFOREST_RESIN":
                result[product] = self.resin_strategy(state, order_depth[product])
            elif product == "KELP":
                result[product] = self.kelp_strategy(order_depth[product])

        if hasattr(self, 'lockout_data'):
            trader_data = json.dumps(self.lockout_data)

        return result, conversions, trader_data
