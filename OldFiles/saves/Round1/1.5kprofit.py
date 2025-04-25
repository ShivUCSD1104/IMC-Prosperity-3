from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    
    def extract_bids_and_asks(self, state: TradingState, product: str) -> Dict[str, List[float]]:
         """
         Extracts the lowest and highest bids and asks from current state.order_depths
         """
         order_depth: OrderDepth = state.order_depths[product]

         top_bp = max(order_depth.buy_orders.keys())
         top_bv = order_depth.buy_orders[top_bp]

         low_bp = min(order_depth.buy_orders.keys())
         low_bv = order_depth.buy_orders[low_bp]
         
         top_ap = max(order_depth.sell_orders.keys())
         top_av = order_depth.sell_orders[top_ap]
         
         low_ap = min(order_depth.sell_orders.keys())
         low_av = order_depth.sell_orders[low_ap]
         
         return {
                'highest_bid': [top_bp, top_bv], 
                 'lowest_bid': [low_bp, low_bv], 
                 'highest_ask': [top_ap, top_av], 
                 'lowest_ask': [low_ap, low_av]
                 }
    
    def resin_strategy(self, state: TradingState, bids_and_asks: Dict) -> List[Order]:
        """
        Resin is a stable product therefore mean reversion seems like the most optimal strategy
        """
        if not bids_and_asks:
            return []
        
        LIMIT = 50

        position = state.position.get("RAINFOREST_RESIN", 0)

        orders: List[Order] = []
        best_bid = bids_and_asks['highest_bid']
        best_ask = bids_and_asks['lowest_ask']

        best_bid_price = best_bid[0]
        best_ask_price = best_ask[0]
        best_bid_vol = best_bid[1]
        best_ask_vol = best_ask[1]

        fair_price = (best_bid_price + best_ask_price)/2

        spread = best_ask_price - best_bid_price

        buy_price = best_bid[0] + 1  # Overbid
        sell_price = best_ask[0] - 1  # Undercut

        #scaled_volume = int(max(1, min(spread, LIMIT)))
        scaled_volume = float('inf')

        if spread >= 3:
          # Normal market-making mode
          buy_price = best_bid_price + 1
          sell_price = best_ask_price - 1

          if position < LIMIT:
              buy_volume = min(LIMIT - position, scaled_volume)
              print(f"BUY RESIN @ {buy_price} | {buy_volume}")
              orders.append(Order("RAINFOREST_RESIN", buy_price, buy_volume))

          if position > -LIMIT:
              sell_volume = min(position + LIMIT, scaled_volume)
              print(f"SELL RESIN @ {sell_price} | {sell_volume}")
              orders.append(Order("RAINFOREST_RESIN", sell_price, -sell_volume))

        else:
            # Tight spread → mean-reversion logic
            if best_ask_price < fair_price and position < LIMIT:
                # Price is low → mean reversion BUY
                buy_price = best_ask_price
                buy_volume = min(LIMIT - position, scaled_volume)
                print(f"[MEAN REVERT] Tight spread BUY @ {buy_price} | {buy_volume}")
                orders.append(Order("RAINFOREST_RESIN", buy_price, buy_volume))

            elif best_bid_price > fair_price and position > -LIMIT:
                # Price is high → mean reversion SELL
                sell_price = best_bid_price
                sell_volume = min(position + LIMIT, scaled_volume)
                print(f"[MEAN REVERT] Tight spread SELL @ {sell_price} | {sell_volume}")
                orders.append(Order("RAINFOREST_RESIN", sell_price, -sell_volume))
            else:
                print("[MEAN REVERT] No edge in tight spread. Holding.")

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
        
        for product in state.order_depths.keys():
            if (product == "RAINFOREST_RESIN"):
                bids_and_asks = self.extract_bids_and_asks(state, "RAINFOREST_RESIN")
                result[product] = self.resin_strategy(state, bids_and_asks)
            elif (product == "KELP"):
                pass

        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 0 

        # Return the dict of orders
        # These possibly contain buy or sell orders
        # Depending on the logic above
        return result, conversions, traderData
