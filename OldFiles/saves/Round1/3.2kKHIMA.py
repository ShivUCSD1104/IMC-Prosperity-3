from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    
    def resin_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        """
        Market making and taking strategy centered around 10000
        """
        product = "RAINFOREST_RESIN"
        LIMIT = 50
        FAIR_VALUE = 10000

        position = state.position.get(product, 0)
        orders: List[Order] = []

        buy_orders = sorted(order_depth.sell_orders.items())  # market sells → our buys
        sell_orders = sorted(order_depth.buy_orders.items(), reverse=True)  # market buys → our sells

        # --- Market taking if favorable ---
        for ask_price, ask_vol in buy_orders:
            if ask_price < FAIR_VALUE and position < LIMIT:
                vol = min(-ask_vol, LIMIT - position)
                print(f"[TAKE] {product} BUY {vol} @ {ask_price}")
                orders.append(Order(product, ask_price, vol))
                position += vol

        for bid_price, bid_vol in sell_orders:
            if bid_price > FAIR_VALUE and position > -LIMIT:
                vol = min(bid_vol, position + LIMIT)
                print(f"[TAKE] {product} SELL {vol} @ {bid_price}")
                orders.append(Order(product, bid_price, -vol))
                position -= vol

        # --- Market making if spread allows ---
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid

            if spread >= 0:
                # editing: tune market pricing
                midprice = int((best_bid + best_ask) / 2) 
                buy_price = midprice - 3
                sell_price = midprice + 3

                # Volume scales down as we near position limit
                # We keep trading near limit maximising the number of transactions per iteration
                if (LIMIT - position) < 5:
                    print("L-pos flag")
                if (position + LIMIT < 5):
                    print("pos+L flag")
                buy_vol = min(4, LIMIT - position)
                sell_vol = min(4, position + LIMIT)

                if buy_vol > 0:
                    print(f"[MM] {product} Quote BUY {buy_vol} @ {buy_price}")
                    orders.append(Order(product, buy_price, int(buy_vol*3.5)))
                if sell_vol > 0:
                    print(f"[MM] {product} Quote SELL {sell_vol} @ {sell_price}")
                    orders.append(Order(product, sell_price, -int(sell_vol*3.5)))
                
        return orders

    
    def kelp_strategy(self, order_depth: Dict) -> Order:
        """
        Kelp is a volatile product so this strategy focusses on filling the lowest bids and then the lowest asks to generate a large profit 
        """ 
        return None

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}
        order_depth = state.order_depths
        
        for product in state.order_depths.keys():
            if (product == "RAINFOREST_RESIN"):
                result[product] = self.resin_strategy(state, order_depth[product])
            elif (product == "KELP"):
                pass

        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 0 

        # Return the dict of orders
        # These possibly contain buy or sell orders
        # Depending on the logic above
        return result, conversions, traderData
