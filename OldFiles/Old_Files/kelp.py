from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState
import json

class Trader:
    def __init__(self):
        self.window = 10
        self.ofi_history = []

    def kelp_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "KELP"
        LIMIT = 50
        FAIR_VALUE = 2000
        SPREAD_THRESHOLD = 2
        MAX_TAKE_VOL = 15
        MM_VOL = 5
        EXIT_VOL = 1

        orders = []
        position = state.position.get(product, 0)

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread = best_ask - best_bid

        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = order_depth.sell_orders[best_ask]

        # --- Compute Order Flow Imbalance ---
        ofi = bid_vol + ask_vol
        self.ofi_history.append(ofi)
        if len(self.ofi_history) > self.window:
            self.ofi_history.pop(0)

        rolling_ofi = sum(self.ofi_history)
        print(f"[KELP] OFI: {rolling_ofi}, Spread: {spread}, Pos: {position}")

        # --- Market Taking (when spread is good and OFI strong) ---
        if spread >= SPREAD_THRESHOLD:
            if rolling_ofi > 40 and best_ask <= FAIR_VALUE and position < LIMIT:
                buy_qty = min(-ask_vol, LIMIT - position, MAX_TAKE_VOL)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    print(f"[TAKE UP] BUY {buy_qty} @ {best_ask}")

            if rolling_ofi < -40 and best_bid >= FAIR_VALUE and position > -LIMIT:
                sell_qty = min(bid_vol, position + LIMIT, MAX_TAKE_VOL)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    print(f"[TAKE DOWN] SELL {sell_qty} @ {best_bid}")

        # --- Exit trades to reduce overexposure ---
        if position > LIMIT * 0.9:
            orders.append(Order(product, best_bid, -EXIT_VOL))
            print(f"[EXIT] SELL {EXIT_VOL} @ {best_bid}")
        elif position < -LIMIT * 0.9:
            orders.append(Order(product, best_ask, EXIT_VOL))
            print(f"[EXIT] BUY {EXIT_VOL} @ {best_ask}")

        # --- Market Making (with OFI bias) ---
        mm_bid_price = best_bid + 1 if rolling_ofi > 0 else best_bid
        mm_ask_price = best_ask - 1 if rolling_ofi < 0 else best_ask

        if position < LIMIT:
            orders.append(Order(product, mm_bid_price, MM_VOL))
        if position > -LIMIT:
            orders.append(Order(product, mm_ask_price, -MM_VOL))

        return orders

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ""

        for product, depth in state.order_depths.items():
            if product == "KELP":
                result[product] = self.kelp_strategy(state, depth)
            else:
                result[product] = []

        return result, conversions, trader_data
