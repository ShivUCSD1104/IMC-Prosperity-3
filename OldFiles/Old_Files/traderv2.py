from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    
    WINDOW = 10
    RESIN_FV = 10000
    POSITION_LIMIT = 50

    def get_vwap(self, trades, window: int, time: int) -> int:
        windowed_trades = [t for t in trades if t.timestamp >= time - 100 * window]
        vwap = sum(t.price * t.quantity for t in windowed_trades)
        vol = sum(abs(t.quantity) for t in windowed_trades)
        return int(vwap / vol) if vol else self.RESIN_FV

    def own_fair_value(self, state: TradingState, product: str) -> int:
        trades = state.own_trades.get(product, [])
        return self.get_vwap(trades, self.WINDOW, state.timestamp)

    def market_fair_value(self, state: TradingState, product: str) -> int:
        trades = state.market_trades.get(product, [])
        return self.get_vwap(trades, self.WINDOW, state.timestamp)

    def resin_strategy(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = "RAINFOREST_RESIN"
        position = state.position.get(product, 0)
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        # Compute fair values
        our_fv = self.own_fair_value(state, product)
        market_fv = self.market_fair_value(state, product)
        print(f"[FV] Our: {our_fv} | Market: {market_fv} | Pos: {position}")

        # Decide MM bias
        spread = best_ask - best_bid
        quote_range = range(1, 4)  # distance from best_bid/ask

        if market_fv > self.RESIN_FV: # Market is rich → prefer to sell
            for i in quote_range:
                price = self.RESIN_FV + i
                if position > -self.POSITION_LIMIT:
                    vol = min(5, position + self.POSITION_LIMIT)
                    orders.append(Order(product, price, -vol))
                    print(f"[MM] SELL @ {price}")

        elif market_fv < self.RESIN_FV: # Market is cheap → prefer to buy
            for i in quote_range:
                price = self.RESIN_FV - i
                if position < self.POSITION_LIMIT:
                    vol = min(5, self.POSITION_LIMIT - position)
                    orders.append(Order(product, price, vol))
                    print(f"[MM] BUY @ {price}")
        else: # Market neutral → quote both sides
            buy_price = self.RESIN_FV - 2
            sell_price = self.RESIN_FV + 2
            if position < self.POSITION_LIMIT:
                orders.append(Order(product, buy_price, 3))
            if position > -self.POSITION_LIMIT:
                orders.append(Order(product, sell_price, -3))
            print(f"[MM] Passive Layer @ {buy_price}/{sell_price}")

        # Force bias correction (adjust our own FV to 10000)
        if our_fv > self.RESIN_FV and position > -self.POSITION_LIMIT:
            orders.append(Order(product, best_bid, -2))  # force sell
            print(f"[REVERT] SELL to push FV ↓")
        elif our_fv < self.RESIN_FV and position < self.POSITION_LIMIT:
            orders.append(Order(product, best_ask, 2))  # force buy
            print(f"[REVERT] BUY to push FV ↑")

        return orders

    def kelp_strategy(self, state: TradingState, order_depth: Dict) -> List[Order]:
        """
        Kelp is a volatile product so this strategy focusses on filling the lowest bids and then the lowest asks to generate a large profit 
        """ 
        return 

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
            if (product == "KELP"):
                pass

        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 0 

        # Return the dict of orders
        # These possibly contain buy or sell orders
        # Depending on the logic above
        return result, conversions, traderData
