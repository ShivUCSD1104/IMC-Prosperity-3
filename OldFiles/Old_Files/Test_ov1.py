from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    
    def resin_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        """
        Improved market making strategy with better risk management.
        """
        product = "RAINFOREST_RESIN"
        LIMIT = 50

        position = state.position.get(product, 0)
        orders: List[Order] = []

        # Sort the sells (ask side) ascending by price => these are prices we can BUY from
        asks = sorted(order_depth.sell_orders.items())  
        # Sort the buys (bid side) descending by price => these are prices we can SELL into
        bids = sorted(order_depth.buy_orders.items(), reverse=True)

        # ---------------------------------------------------------------------
        # 1) Compute dynamic fair value from best bid & best ask (if both sides exist)
        # ---------------------------------------------------------------------
        if len(bids) == 0 or len(asks) == 0:
            # If there's no liquidity on one side, handle carefully. 
            # Simple fallback: do nothing or only trade passively on the existing side
            return orders

        best_bid, best_bid_vol = bids[0]
        best_ask, best_ask_vol = asks[0]
        fair_value = (best_bid + best_ask) / 2

        # ---------------------------------------------------------------------
        # 2) Potentially do some "market taking" if extremely favorable
        # ---------------------------------------------------------------------
        # e.g. if best_ask is much lower than fair_value, buy it. 
        # But watch position limit:
        if best_ask < fair_value - 2 and position < LIMIT:
            # Decide how much we can buy
            volume_to_buy = min(best_ask_vol * -1, LIMIT - position)  # ask_vol is negative in many data models
            if volume_to_buy > 0:
                orders.append(Order(product, best_ask, volume_to_buy))
                position += volume_to_buy
        
        # Similarly, if best_bid is much higher than fair_value, sell it
        if best_bid > fair_value + 2 and position > -LIMIT:
            volume_to_sell = min(best_bid_vol, position + LIMIT)  # bid_vol is positive in many data models
            if volume_to_sell > 0:
                orders.append(Order(product, best_bid, -volume_to_sell))
                position -= volume_to_sell

        # ---------------------------------------------------------------------
        # 3) Market making (posting our own quotes) if inside spread is large enough
        # ---------------------------------------------------------------------
        spread = best_ask - best_bid
        min_spread_needed = 2  # Only quote if spread is >= 2
        if spread < min_spread_needed:
            return orders  # Spread too tight, skip posting

        # ---------------------------------------------------------------------
        # 4) Inventory-based skew in price to avoid extreme positions
        #    - The more net-long we are, the lower we place our buy price
        #      (less likely to buy), and the more aggressively we place our sell price
        # ---------------------------------------------------------------------
        # A simple linear skew: alpha in [0, 2, 3...], tune it to your liking
        alpha = 2.0  
        skew = alpha * (position / LIMIT)  # position in [-50..50], skew in [-alpha..+alpha]

        midprice = fair_value
        base_offset = 3  # basic offset from mid

        buy_price = midprice - base_offset - skew
        sell_price = midprice + base_offset - skew

        # ---------------------------------------------------------------------
        # 5) Volume scaling based on distance to position limits
        # ---------------------------------------------------------------------
        dist_to_buy_limit = (LIMIT - position) / (2 * LIMIT)  # fraction in [0..1]
        dist_to_sell_limit = (LIMIT + position) / (2 * LIMIT) # fraction in [0..1]

        base_size = 15

        buy_vol  = int(base_size * dist_to_buy_limit)
        sell_vol = int(base_size * dist_to_sell_limit)

        # Ensure we never exceed the absolute position limit
        buy_vol  = min(buy_vol, max(0, LIMIT - position))
        sell_vol = min(sell_vol, max(0, LIMIT + position))

        # If we are basically at the limit, maybe skip placing the quote on that side entirely
        if position >= LIMIT:
            buy_vol = 0
        if position <= -LIMIT:
            sell_vol = 0

        # ---------------------------------------------------------------------
        # 6) Place our passive buy & sell orders (market making)
        # ---------------------------------------------------------------------
        if buy_vol > 0:
            orders.append(Order(product, int(buy_price), buy_vol))
        if sell_vol > 0:
            orders.append(Order(product, int(sell_price), -sell_vol))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        for product, depth in state.order_depths.items():
            if product == "RAINFOREST_RESIN":
                result[product] = self.resin_strategy(state, depth)
            elif product == "KELP":
                # ...
                pass

        # Typically the competition runner expects (orders, conversions, traderData)
        # If your environment needs it:
        return result, 0, "SAMPLE"
