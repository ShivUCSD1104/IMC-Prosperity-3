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
        MIN_LOCKOUTS = 2  # Start price scaling only after this

        position = state.position.get(product, 0)
        orders: List[Order] = []

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        lockout_window = trader_data.get("lockout", [])[-WINDOW_SIZE:]
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

        # --- Market taking ---
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

        # --- Adaptive forced liquidation price ---
        if lockout_count >= MIN_LOCKOUTS:
            print("Readjusting Liquidity Provision")
            price_delta = max(0, 2 - (lockout_count - MIN_LOCKOUTS))  # 2 → 1 → 0
            liq_buy_price = FAIR_VALUE - price_delta
            liq_sell_price = FAIR_VALUE + price_delta

            if buy_vol > 0:
                orders.append(Order(product, liq_buy_price, buy_vol // 2))
                print(f"[LIQ ADJ] {product} {buy_vol//2} @ {liq_buy_price}")
            if sell_vol > 0:
                orders.append(Order(product, liq_sell_price, -sell_vol // 2))
                print(f"[LIQ ADJ] {product} {-sell_vol//2} @ {liq_sell_price}")

        # --- Passive quoting ---
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

        position = state.position.get(product, 0)
        orders: List[Order] = []

        buy_orders = sorted(order_depth.sell_orders.items())
        sell_orders = sorted(order_depth.buy_orders.items(), reverse=True)

        kelp_data = trader_data.get("kelp_data_out", {})
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
            "past_prices": past_prices,
        }

        return orders

    def ink_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> List[Order]:
        product = "SQUID_INK"
        LIMIT = 50
        position = state.position.get(product, 0)
        orders: List[Order] = []

        # Load and update historical mid prices
        ink_data = trader_data.get("ink_data", {})
        mid_prices = ink_data.get("mid_prices", [])

        # Calculate current mid price from order book
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=0)
        if best_bid == 0 or best_ask == 0:
            return orders  # no valid market data yet

        mid_price = (best_bid + best_ask) / 2
        mid_prices.append(mid_price)

        if len(mid_prices) < 25:
            self.ink_data = {"mid_prices": mid_prices[-50:]}
            return orders

        # Compute Z-score
        window = np.array(mid_prices[-20:])
        mean = np.mean(window)
        std = np.std(window)
        z = (mid_price - mean) / std if std > 1e-6 else 0

        ENTRY_Z = 1.5
        EXIT_Z = 0.15

        # --- ENTRY LOGIC (Mean Reversion) ---
        if position < LIMIT and z < -ENTRY_Z:
            qty = LIMIT - position
            orders.append(Order(product, int(mid_price), qty))
            print(f"INK: BUY {qty} @ {int(mid_price)} (Z={z:.2f})")

        elif position > -LIMIT and z > ENTRY_Z:
            qty = position + LIMIT
            orders.append(Order(product, int(mid_price), -qty))
            print(f"INK: SELL {qty} @ {int(mid_price)} (Z={z:.2f})")

        # --- EXIT LOGIC (Reversion to Mean) ---
        elif position > 0 and abs(z) < EXIT_Z:
            orders.append(Order(product, int(mid_price), -position))
            print(f"INK: EXIT LONG {position} @ {int(mid_price)} (Z={z:.2f})")

        elif position < 0 and abs(z) < EXIT_Z:
            orders.append(Order(product, int(mid_price), -position))
            print(f"INK: EXIT SHORT {position} @ {int(mid_price)} (Z={z:.2f})")

        # Save updated trader state
        self.ink_data = {"mid_prices": mid_prices[-50:]}
        return orders
    
    ###########################
    ######## ROUND 2 ##########
    ###########################
    
    def hedge_and_execute(self, state: TradingState, product_name: str, volume: int) -> List[Order]:
        orders = []
        if product_name == "PICNIC_BASKET1":
            components = {"CROISSANTS": -6, "JAMS": -3, "DJEMBES": -1}
            limits = {"CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60}
        elif product_name == "PICNIC_BASKET2":
            components = {"CROISSANTS": -4, "JAMS": -2}
            limits = {"CROISSANTS": 250, "JAMS": 350}
        else:
            return orders

        if not hasattr(self, "hedge_data_out"):
            self.hedge_data_out = {}

        for comp, ratio in components.items():
            hedge_volume = volume * ratio
            depth = state.order_depths[comp]
            position = state.position.get(comp, 0)
            limit = limits[comp]
            executed = 0
            remaining = min(abs(hedge_volume), limit - abs(position))

            if hedge_volume > 0:
                for ask in sorted(depth.sell_orders):
                    if executed >= remaining:
                        break
                    avail = -depth.sell_orders[ask]
                    trade_vol = min(avail, remaining - executed)
                    orders.append(Order(comp, ask, trade_vol))
                    executed += trade_vol
            else:
                for bid in sorted(depth.buy_orders, reverse=True):
                    if executed >= remaining:
                        break
                    avail = depth.buy_orders[bid]
                    trade_vol = min(avail, remaining - executed)
                    orders.append(Order(comp, bid, -trade_vol))
                    executed += trade_vol

            self.hedge_data_out[comp] = self.hedge_data_out.get(comp, 0) + executed * (1 if hedge_volume > 0 else -1)

        return orders

    def picnic_basket1_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "PICNIC_BASKET1"
        LIMIT = 60
        CROISSANTS = 6
        JAMS = 3
        DJEMBES = 1

        orders: List[Order] = []
        position = state.position.get(product, 0)

        cro_depth = state.order_depths.get("CROISSANTS")
        jam_depth = state.order_depths.get("JAMS")
        djem_depth = state.order_depths.get("DJEMBES")
        basket_depth = order_depth

        if not cro_depth or not jam_depth or not djem_depth:
            return orders

        cro_mid = (max(cro_depth.buy_orders) + min(cro_depth.sell_orders)) / 2 if cro_depth.buy_orders and cro_depth.sell_orders else None
        jam_mid = (max(jam_depth.buy_orders) + min(jam_depth.sell_orders)) / 2 if jam_depth.buy_orders and jam_depth.sell_orders else None
        djem_mid = (max(djem_depth.buy_orders) + min(djem_depth.sell_orders)) / 2 if djem_depth.buy_orders and djem_depth.sell_orders else None

        if cro_mid is None or jam_mid is None or djem_mid is None:
            return orders

        synthetic_value = CROISSANTS * cro_mid + JAMS * jam_mid + DJEMBES * djem_mid

        for ask in sorted(basket_depth.sell_orders):
            if ask < synthetic_value and position < LIMIT:
                volume = min(-basket_depth.sell_orders[ask], LIMIT - position)
                orders.append(Order(product, ask, volume))
                orders += self.hedge_and_execute(state, product, volume)
                break

        for bid in sorted(basket_depth.buy_orders, reverse=True):
            if bid > synthetic_value and position > -LIMIT:
                volume = min(basket_depth.buy_orders[bid], position + LIMIT)
                orders.append(Order(product, bid, -volume))
                orders += self.hedge_and_execute(state, product, -volume)
                break

        return orders

    def picnic_basket2_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "PICNIC_BASKET2"
        LIMIT = 100
        CROISSANTS = 4
        JAMS = 2

        orders: List[Order] = []
        position = state.position.get(product, 0)

        cro_depth = state.order_depths.get("CROISSANTS")
        jam_depth = state.order_depths.get("JAMS")
        basket_depth = order_depth

        if not cro_depth or not jam_depth:
            return orders

        cro_mid = (max(cro_depth.buy_orders) + min(cro_depth.sell_orders)) / 2 if cro_depth.buy_orders and cro_depth.sell_orders else None
        jam_mid = (max(jam_depth.buy_orders) + min(jam_depth.sell_orders)) / 2 if jam_depth.buy_orders and jam_depth.sell_orders else None

        if cro_mid is None or jam_mid is None:
            return orders

        synthetic_value = CROISSANTS * cro_mid + JAMS * jam_mid

        for ask in sorted(basket_depth.sell_orders):
            if ask < synthetic_value and position < LIMIT:
                volume = min(-basket_depth.sell_orders[ask], LIMIT - position)
                orders.append(Order(product, ask, volume))
                orders += self.hedge_and_execute(state, product, volume)
                break

        for bid in sorted(basket_depth.buy_orders, reverse=True):
            if bid > synthetic_value and position > -LIMIT:
                volume = min(basket_depth.buy_orders[bid], position + LIMIT)
                orders.append(Order(product, bid, -volume))
                orders += self.hedge_and_execute(state, product, -volume)
                break

        return orders

    def croissants_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> List[Order]:
        LIMIT = 250
        product = "CROISSANTS"
        position = state.position.get(product, 0)
        hedge = trader_data.get("hedge_data_out", {}).get(product, 0)
        if abs(position + hedge) >= LIMIT:
            return []
        return []

    def jams_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> List[Order]:
        LIMIT = 350
        product = "JAMS"
        position = state.position.get(product, 0)
        hedge = trader_data.get("hedge_data_out", {}).get(product, 0)
        if abs(position + hedge) >= LIMIT:
            return []
        return []

    def djembes_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> List[Order]:
        LIMIT = 60
        product = "DJEMBES"
        position = state.position.get(product, 0)
        hedge = trader_data.get("hedge_data_out", {}).get(product, 0)
        if abs(position + hedge) >= LIMIT:
            return []
        return []

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        self.hedge_data_out = {}

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
            elif product == "PICNIC_BASKET1":
                result[product] = self.picnic_basket1_strategy(state, state.order_depths[product])
            elif product == "PICNIC_BASKET2":
                result[product] = self.picnic_basket2_strategy(state, state.order_depths[product])
            elif product == "CROISSANTS":
                result[product] = self.croissants_strategy(state, state.order_depths[product], trader_data)
            elif product == "JAMS":
                result[product] = self.jams_strategy(state, state.order_depths[product], trader_data)
            elif product == "DJEMBES":
                result[product] = self.djembes_strategy(state, state.order_depths[product], trader_data)

        traderData = json.dumps({
            "kelp_data_out": getattr(self, "kelp_data_out", {}),
            "resin_lockout_data": getattr(self, "resin_lockout_data", {}),
            "ink_data": getattr(self, "ink_data", {}),
            "hedge_data_out": self.hedge_data_out
        })

        return result, conversions, traderData


