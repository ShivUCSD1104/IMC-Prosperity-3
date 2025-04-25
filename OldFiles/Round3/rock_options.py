from typing import Dict, List, Tuple
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

        ENTRY_Z = 1.75
        EXIT_Z = 0.4

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

    def picnic_basket1_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> Tuple[List[Order], Dict[str, int]]:
        product = "PICNIC_BASKET1"
        LIMIT = 60
        CROISSANTS, JAMS, DJEMBES = 6, 3, 1

        orders: List[Order] = []
        hedge_volumes: Dict[str, int] = {"CROISSANTS": 0, "JAMS": 0, "DJEMBES": 0}
        position = state.position.get(product, 0)

        cro_depth = state.order_depths.get("CROISSANTS")
        jam_depth = state.order_depths.get("JAMS")
        djem_depth = state.order_depths.get("DJEMBES")
        basket_depth = order_depth

        if not cro_depth or not jam_depth or not djem_depth:
            return orders, hedge_volumes

        cro_mid = (max(cro_depth.buy_orders) + min(cro_depth.sell_orders)) / 2
        jam_mid = (max(jam_depth.buy_orders) + min(jam_depth.sell_orders)) / 2
        djem_mid = (max(djem_depth.buy_orders) + min(djem_depth.sell_orders)) / 2


        synthetic_fair_value = CROISSANTS * cro_mid + JAMS * jam_mid + DJEMBES * djem_mid

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = best_bid + best_ask / 2

        basket1_data = trader_data.get("basket1_data", {})
        past_diffs = basket1_data.get("past_differences", [])

        difference = synthetic_fair_value - mid_price
        past_diffs.append(difference)

        diff_mean = np.mean(past_diffs[-50:])
        norm_diff = difference / (2 * diff_mean)
        scale_factor = min( 2.1 * (1 / (1 + np.exp(-4.6 * norm_diff )) - 0.5), 1 )

        for ask in sorted(basket_depth.sell_orders):
            if ask < synthetic_fair_value and position < LIMIT:
                volume = int(np.ceil(basket_depth.sell_orders[ask] * scale_factor)) 
                orders.append(Order(product, ask, volume))
                hedge_volumes["CROISSANTS"] += volume * CROISSANTS
                hedge_volumes["JAMS"] += volume * JAMS
                hedge_volumes["DJEMBES"] += volume * DJEMBES
                print(f"[BUY {product}] {volume} @ {ask} < FV {synthetic_fair_value:.2f}")
                break

        for bid in sorted(basket_depth.buy_orders, reverse=True):
            if bid > synthetic_fair_value and position > -LIMIT:
                volume = int(np.ceil(-basket_depth.buy_orders[bid] * scale_factor))
                orders.append(Order(product, bid, volume))
                hedge_volumes["CROISSANTS"] += volume * CROISSANTS
                hedge_volumes["JAMS"] += volume * JAMS
                hedge_volumes["DJEMBES"] += volume * DJEMBES
                print(f"[SELL {product}] {volume} @ {bid} > FV {synthetic_fair_value:.2f}")
                break
        
        self.basket1_data = {"past_differences": past_diffs[-100:]}

        return orders, hedge_volumes

    
    def picnic_basket2_strategy(self, state: TradingState, order_depth: OrderDepth, trader_data: Dict) -> Tuple[List[Order], Dict[str, int]]:
        product = "PICNIC_BASKET2"
        LIMIT = 100
        CROISSANTS, JAMS = 4, 2

        orders: List[Order] = []
        hedge_volumes: Dict[str, int] = {"CROISSANTS": 0, "JAMS": 0}
        position = state.position.get(product, 0)

        cro_depth = state.order_depths.get("CROISSANTS")
        jam_depth = state.order_depths.get("JAMS")
        basket_depth = order_depth

        if not cro_depth or not jam_depth:
            return orders, hedge_volumes

        cro_mid = (max(cro_depth.buy_orders) + min(cro_depth.sell_orders)) / 2
        jam_mid = (max(jam_depth.buy_orders) + min(jam_depth.sell_orders)) / 2

        synthetic_fair_value = CROISSANTS * cro_mid + JAMS * jam_mid

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = best_bid + best_ask / 2

        basket2_data = trader_data.get("basket2_data", {})
        past_diffs = basket2_data.get("past_differences", [])

        difference = synthetic_fair_value - mid_price
        past_diffs.append(difference)

        diff_std = np.std(past_diffs[-50:]) + 1e-5
        diff_mean = np.mean(past_diffs[-50:])
        norm_diff = difference / (2 * diff_mean)
        raw_scale = 2.1 * (1 / (1 + np.exp(-4.6 * norm_diff )) - 0.5)
        scale_factor = min((raw_scale * np.exp(2.2 * diff_std/diff_mean)), 1)

        for ask in sorted(basket_depth.sell_orders):
            if ask < synthetic_fair_value - 1 and position < LIMIT:
                volume = int(np.ceil(basket_depth.sell_orders[ask] * scale_factor))
                orders.append(Order(product, ask, volume))
                hedge_volumes["CROISSANTS"] += volume * CROISSANTS
                hedge_volumes["JAMS"] += volume * JAMS
                print(f"[BUY {product}] {volume} @ {ask} < FV {synthetic_fair_value:.2f}")

        for bid in sorted(basket_depth.buy_orders, reverse=True):
            if bid > synthetic_fair_value + 1 and position > -LIMIT:
                volume = int(np.ceil(-basket_depth.buy_orders[bid] * scale_factor))
                orders.append(Order(product, bid, volume))
                hedge_volumes["CROISSANTS"] += volume * CROISSANTS
                hedge_volumes["JAMS"] += volume * JAMS
                print(f"[SELL {product}] {volume} @ {bid} > FV {synthetic_fair_value:.2f}")

        self.basket2_data = {"past_differences": past_diffs[-100:]}

        return orders, hedge_volumes

    def croissants_strategy(self, state: TradingState, order_depth: OrderDepth, hedge_volume: int) -> List[Order]:
        LIMIT = 250
        product = "CROISSANTS"
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if hedge_volume == 0 or abs(position + hedge_volume) > LIMIT:
            return []

        if hedge_volume > 0:
            best_ask = min(order_depth.sell_orders)
            orders.append(Order(product, best_ask, hedge_volume))
            print(f"[HEDGE BUY] {product} {hedge_volume} @ {best_ask}")
        else:
            best_bid = max(order_depth.buy_orders)
            orders.append(Order(product, best_bid, hedge_volume))
            print(f"[HEDGE SELL] {product} {hedge_volume} @ {best_bid}")

        return orders

    def jams_strategy(self, state: TradingState, order_depth: OrderDepth, hedge_volume: int) -> List[Order]:
        LIMIT = 350
        product = "JAMS"
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if hedge_volume == 0 or abs(position + hedge_volume) > LIMIT:
            return []

        if hedge_volume > 0:
            best_ask = min(order_depth.sell_orders)
            orders.append(Order(product, best_ask, hedge_volume))
            print(f"[HEDGE BUY] {product} {hedge_volume} @ {best_ask}")
        else:
            best_bid = max(order_depth.buy_orders)
            orders.append(Order(product, best_bid, hedge_volume))
            print(f"[HEDGE SELL] {product} {hedge_volume} @ {best_bid}")

        return orders

    def djembes_strategy(self, state: TradingState, order_depth: OrderDepth, hedge_volume: int) -> List[Order]:
        LIMIT = 60
        product = "DJEMBES"
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if hedge_volume == 0 or abs(position + hedge_volume) > LIMIT:
            return []

        if hedge_volume > 0:
            best_ask = min(order_depth.sell_orders)
            orders.append(Order(product, best_ask, hedge_volume))
            print(f"[HEDGE BUY] {product} {hedge_volume} @ {best_ask}")
        else:
            best_bid = max(order_depth.buy_orders)
            orders.append(Order(product, best_bid, hedge_volume))
            print(f"[HEDGE SELL] {product} {hedge_volume} @ {best_bid}")

        return orders

    ###########################
    ######## ROUND 3 ##########
    ########################### 
    
    def volcanic_voucher_strategy(self, state: TradingState, order_depth: OrderDepth, product: str, trader_data: Dict) -> List[Order]:
        LIMIT = 200
        product = product
        position = state.position.get(product, 0)
        orders: List[Order] = []

        STRIKE = int(product[22:])

        underlying = state.order_depths.get("VOLCANIC_ROCK")
        if not underlying or not underlying.buy_orders or not underlying.sell_orders:
          return orders, 0

        best_bid = max(underlying.buy_orders)
        best_ask = min(underlying.sell_orders)
        mid_price = (best_bid + best_ask) / 2

        #PRICING
        volcanic_data = trader_data.get("volcanic_data", {})
        past_prices = volcanic_data.get("past_prices", [])
        past_prices.append(mid_price)

        #TUNABLE PARAMS
        WINDOW = 50
        ALPHA = 0.53
        THRESHOLD = 1
        
        #ACTUAL STRATEGY

        if len(past_prices) < WINDOW:
          self.volcanic_data = {"past_prices": past_prices}
          return orders, 0

        fair_price = np.mean(past_prices[-WINDOW:])
        volatility = np.std(past_prices[-WINDOW:])

        intrinsic_value = max(0, fair_price - STRIKE)
        model_price = intrinsic_value + ALPHA * volatility

        best_voucher_bid = max(order_depth.buy_orders.keys(), default=None)
        best_voucher_ask = min(order_depth.sell_orders.keys(), default=None)

        hedge_volume = 0
        delta = max(0, min(1, 1 - (STRIKE - fair_price) / 500)) if fair_price < STRIKE else 1

        if best_voucher_ask is not None and best_voucher_ask < model_price - THRESHOLD and position < LIMIT:
            volume = min(order_depth.sell_orders[best_voucher_ask], LIMIT - position)
            #volume = order_depth.sell_orders[best_voucher_ask]
            orders.append(Order(product, best_voucher_ask, volume))
            print(f"[{product}] BUY {volume} @ {best_voucher_ask:.1f} | model={model_price:.1f}, delta={delta:.2f}")
            hedge_volume += -int(round(volume * delta))
            #CHANGED TO INCREMENT INSTEAD OF = 
        if best_voucher_bid is not None and best_voucher_bid > model_price + THRESHOLD and position > -LIMIT:
            volume = min(order_depth.buy_orders[best_voucher_bid], position + LIMIT)
            #volume = order_depth.buy_orders[best_voucher_bid]
            orders.append(Order(product, best_voucher_bid, -volume))
            print(f"[{product}] SELL {volume} @ {best_voucher_bid:.1f} | model={model_price:.1f}, delta={delta:.2f}")
            hedge_volume += int(round(volume * delta))
            #CHANGED TO INCREMENT INSTEAD OF = 
        self.volcanic_data = {"past_prices": past_prices}

        return orders, hedge_volume
    
        
    def volcanic_rock_strategy(self, state: TradingState, order_depth: OrderDepth, hedge_volume: int) -> List[Order]:
        LIMIT = 400
        product = "VOLCANIC_ROCK"
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if hedge_volume == 0:
            return []

        if hedge_volume > 0 and position < LIMIT:
            for ask in sorted(order_depth.sell_orders):
                vol = min(-order_depth.sell_orders[ask], hedge_volume, LIMIT - position)
                orders.append(Order(product, ask, vol))
                hedge_volume -= vol
                print(f"[HEDGE] BUY {vol} VOLCANIC_ROCK @ {ask}")
                if hedge_volume <= 0:
                    break

        elif hedge_volume < 0 and position > -LIMIT:
            for bid in sorted(order_depth.buy_orders, reverse=True):
                vol = min(order_depth.buy_orders[bid], -hedge_volume, position + LIMIT)
                orders.append(Order(product, bid, -vol))
                hedge_volume += vol
                print(f"[HEDGE] SELL {vol} VOLCANIC_ROCK @ {bid}")
                if hedge_volume >= 0:
                    break

        return orders

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        basket_hedge_volumes = {
            "CROISSANTS": 0,
            "JAMS": 0,
            "DJEMBES": 0
        }

        for product in state.order_depths:
            if product == "RAINFOREST_RESIN":
                result[product] = self.resin_strategy(state, state.order_depths[product])
            elif product == "KELP":
                result[product] = self.kelp_strategy(state, state.order_depths[product], trader_data)
            elif product == "SQUID_INK":
                result[product] = self.ink_strategy(state, state.order_depths[product], trader_data)
            elif product == "PICNIC_BASKET1":
                orders, hedges = self.picnic_basket1_strategy(state, state.order_depths[product], trader_data)
                result[product] = orders
                for k, v in hedges.items():
                    basket_hedge_volumes[k] += v
            elif product == "PICNIC_BASKET2":
                orders, hedges = self.picnic_basket2_strategy(state, state.order_depths[product], trader_data)
                result[product] = orders
                for k, v in hedges.items():
                    basket_hedge_volumes[k] += v

        for product in state.order_depths:
            if product == "CROISSANTS":
                result[product] = self.croissants_strategy(state, state.order_depths[product], basket_hedge_volumes["CROISSANTS"])
            elif product == "JAMS":
                result[product] = self.jams_strategy(state, state.order_depths[product], basket_hedge_volumes["JAMS"])
            elif product == "DJEMBES":
                result[product] = self.djembes_strategy(state, state.order_depths[product], basket_hedge_volumes["DJEMBES"])

        ##### -------- ROUND 3 --------

        volcanic_hedge = 0
        for product in state.order_depths:
            if product.startswith("VOLCANIC_ROCK_VOUCHER"):
                orders, hedge = self.volcanic_voucher_strategy(state, state.order_depths[product], product, trader_data)
                result[product] = orders
                volcanic_hedge += hedge

        if "VOLCANIC_ROCK" in state.order_depths:
            result["VOLCANIC_ROCK"] = self.volcanic_rock_strategy(state, state.order_depths["VOLCANIC_ROCK"], volcanic_hedge)

        traderData = json.dumps({
            "kelp_data_out": getattr(self, "kelp_data_out", {}),
            "resin_lockout_data": getattr(self, "resin_lockout_data", {}),
            "ink_data": getattr(self, "ink_data", {}),
            "basket1_data": getattr(self, "basket1_data", {}),
            "basket2_data": getattr(self, "basket2_data", {}),
            "volcanic_data": getattr(self, "volcanic_data", {})
        })

        return result, conversions, traderData


