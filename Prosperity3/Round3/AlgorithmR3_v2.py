import json
import jsonpickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # TODO: Add logic

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    INK = "SQUID_INK"

class Trader:
    
    params = {
        Product.RESIN: {
            "fair_value": 10000,
            "take_width": 1,
            "clear_width": 0,
            # for making
            "disregard_edge": 0,  # disregards orders for joining or pennying within this value from fair
            "join_edge": 2,  # joins orders within this edge
            "default_edge": 4,
            "soft_position_limit": 20,
        },
        Product.KELP: {
            "take_width": 1,
            "clear_width": 0,
            "prevent_adverse": True,
            "adverse_volume": 15,
            "reversion_beta": -0.229,
            "disregard_edge": 1,
            "join_edge": 0,
            "default_edge": 1,
        },
        Product.INK: {
            "take_width": 1,
            "clear_width": 0,
            "prevent_adverse": True,
            "adverse_volume": 15,
            #"reversion_beta": -0.229,
            "disregard_edge": 1,
            "join_edge": 0,
            "default_edge": 1,
        },
    }
        
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.limits[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.limits[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.limits[product] - (position + buy_order_volume)
        sell_quantity = self.limits[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
        
    
    
    #######################################################################################
    
    
    #define commodity limits here
    def get_limits(self):
        self.limits = {
            'RAINFOREST_RESIN': 50,
            'KELP': 50,
            'SQUID_INK': 50,
            'CROISSANTS': 250,
            'JAMS': 350,
            'DJEMBES': 60,
            'PICNIC_BASKET1': 60,
            'PICNIC_BASKET2': 100,
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200
        }
            
        
    #check at trading time if we are going to exceed limits, return available space for commodity
    def get_liquidity(self, state, product, total_pos_change, side = "BUY", position=None):
        #take into account the other orders we have already put in
        if position is None:
            position = state.position.get(product,0)
        
        if side == "BUY" or side == 1:
            return max(self.limits[product] - position - total_pos_change, 0)
        elif side == "SELL" or side == -1:
            return min(-self.limits[product] - position - total_pos_change, 0)
        
        
    
    #predefined or maybe even dynamic acceptable prices per product
    def initialize_data(self, returns_symbols):
        #init data
        self.traderObject = {}
        
        self.ink_timestamp = 0

        #set data
        self.data = {
            'RAINFOREST_RESIN': [10000],
            'KELP': [2000],
            'SQUID_INK': [2000],
        }
        
        #variables
        self.variables = {
                    
        }
        
        
        indicators = ['_RETURNS', '_SMA_RETURNS', '_EMA_RETURNS', '_EMA']
        for symbol in returns_symbols:
            for indicator in indicators:
                tag = symbol + indicator
                self.data[tag] = [0]
         
        
                 
    #update data
    def update_data(self, vwap_products, returns_symbols, state):
        for product in state.order_depths:
            if product not in vwap_products:
                continue
            
            #get order data
            order_depth = state.order_depths[product]

            if len(order_depth.buy_orders) + len(order_depth.sell_orders) != 0 and state.timestamp != 0:
                use_vwap = True #for testing
                
                if (use_vwap):
                    vwap = self.get_vwap_mid(order_depth)
                else:
                    min_bid = min(order_depth.buy_orders.keys())
                    max_ask = max(order_depth.sell_orders.keys())
                    midpoint = int((min_bid + max_ask)/2)
                    vwap = midpoint
                    
                #logger.print(f"Mean of order depths for {product}: {mean}")
                
                #vwap
                self.data.setdefault(product, []).append(vwap)
                
                if product in returns_symbols:
                    #RETURNS
                    symbol = product+"_RETURNS"
                    ret = self.data[product][-1] - self.data[product][-2]
                    #if ret != 0:
                    self.data[symbol].append(ret)
                    
                    #SMA of returns
                    symbol = product+"_SMA_RETURNS"
                    #returns_array = np.array([self.data[product+"_RETURNS"]])
                    #self.data[symbol].append(np.average(returns_array))
                    #self.data[symbol].append(self.data[product+"_RETURNS"][-1]**2)
                    
                    #EMA returns
                    period = 15#self.data[product+"_ROLLING_WINDOW"][-1]
                    smoothing = 2
                    factor = smoothing/(period+1)
                    symbol = product+"_EMA_RETURNS"
                    
                    #self.data[symbol].append(self.data[product+"_RETURNS"][-1]*factor+self.data[symbol][-1]*(1-factor))
                    
                    #EMA
                    period = 15#self.data[product+"_ROLLING_WINDOW"][-1]
                    smoothing = 2
                    factor = smoothing/(period+1)
                    symbol = product+"_EMA"
                    
                    #self.data[symbol].append(self.data[product][-1]*factor+self.data[symbol][-1]*(1-factor))
                    
              
            else:
                for product in self.data.values():
                    product.append(product[-1])
                    
                    
        #Keep rolling window for memory purposes
        for symbol in self.data.keys():
                self.data[symbol] = [round(x, 4) for x in self.data[symbol] if isinstance(x, (int,float))][-200:]
        
        #smaller windows
        symbols = [product+"_EMA_RETURNS", product+"_TRADES"]
        for symbol in symbols:
            if symbol in self.data:
                self.data[symbol] = self.data[symbol][-15:]
            
                    
    #save data to json string for loading in new executions
    def serialize(self):
        var = {
            'data': self.data,
            'variables': self.variables,
            'traderObject': self.traderObject,
            'ink_stamp': self.ink_timestamp
        }
        
        return jsonpickle.encode(var)
    
    def deserialize(self, json):
        var = jsonpickle.decode(json)
        self.data = var['data']
        self.variables = var['variables']
        self.traderObject = var['traderObject']
        self.ink_timestamp = var['ink_stamp']
            
    def get_synthetic_book(self, depth, basket):
        # init synthetic basket order depth
        synthetic_book = OrderDepth()
        
        # calculate best bid and ask for each component
        components = ['CROISSANTS', 'JAMS', 'DJEMBES']
        best_bid = {}
        best_ask = {}
        implied_bid = 0
        implied_ask = 0
        
        for component in components:
            book = depth[component]
            best_bid[component] = max(book.buy_orders.keys()) if book.buy_orders else 0
            best_ask[component] = min(book.sell_orders.keys()) if book.sell_orders else 0
            
            # calculate implied bid / ask for synthetic basket
            """
            if basket == "PICNIC_BASKET1":
                weight = self.basket_weights['SYNTHETIC_3'][component]
            else:
                """
            
            weight =  self.basket_weights[basket][component]
                
            implied_bid += best_bid[component] * weight
            implied_ask -= best_ask[component] * weight
            
        # calculate the max number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            bid_volume = {}
            implied_bid_volume = 0
            
            for component in components:
                book = depth[component]
                bid_volume[component] = book.buy_orders[best_bid[component]]
            
            # start constructing our synthetic order book
            implied_bid_volume = min(bid_volume.values())
            synthetic_book.buy_orders[implied_bid] = implied_bid_volume
            
        if implied_ask < float("inf"):
            ask_volume = {}
            implied_ask_volume = 0
            
            for component in components:
                book = depth[component]
                ask_volume[component] = book.sell_orders[best_ask[component]]
            
            # start constructing our synthetic order book
            implied_ask_volume = min(ask_volume.values())
            synthetic_book.sell_orders[implied_bid] = -implied_ask_volume
            
        return synthetic_book
    
    def convert_synthetic_basket_orders(self, state, synthetic_orders, depth, basket):
        # init dict to store orders per component
        components = ['CROISSANTS', 'JAMS', 'DJEMBES']
        component_orders = {component: [] for component in components}
        
        # get best bid / ask for synthetic basket
        synthetic_book = self.get_synthetic_book(depth, basket)
        best_bid = max(synthetic_book.buy_orders.keys()) if synthetic_book.buy_orders else 0
        best_ask = min(synthetic_book.sell_orders.keys()) if synthetic_book.sell_orders else 0
        
        clear = False
        
        #logger.print("Trying to convert orders.")
        # iterate each synthetic order
        for order in synthetic_orders:
            #get price and quantity
            price = order.price
            quantity = order.quantity
            
            if price == -1:
                clear = True
            
            component_prices = {}
            component_quantities = {}
            place_orders = True
            
            # check if the synthetic basket order aligns with best bid or ask
                # Buy, trade components at best ask prices
            for component in components:
                weight = self.basket_weights[basket][component]
                quantity *= -weight
                
                component_quantities[component] = quantity

                if clear:
                    quantity = state.position.get(component, 0)
                    
                    
                if quantity > 0 and price >= best_ask:
                    component_prices[component] = min(depth[component].sell_orders.keys())
                elif quantity < 0 and price <= best_bid:    
                    component_prices[component] = max(depth[component].buy_orders.keys())
                else:
                    # synthetic basket doesn't align with best bid or ask
                    place_orders = False
            
                
            
            # create orders for each component
            component_orders = {}
            if place_orders:
                for component in components:
                    
                    #weight = self.basket_weights['SYNTHETIC_4'][component]
                    weight = self.basket_weights[basket][component]
                    #quantity = state.position.get(basket, 0)*-weight - state.position.get(component, 0)
                    quantity = component_quantities[component]

                    if quantity < 0:
                        order_size = self.get_liquidity(state, component, 0, 'SELL')
                        order_size = max(order_size, quantity)
                    else:
                        order_size = self.get_liquidity(state, component, 0, 'BUY')
                        order_size = min(order_size, quantity)
                        
                    if clear:
                        order_size = -state.position.get(component, 0)
                        
                    #logger.print(f"basket pos: {state.position.get(basket, 0)}, weight: {weight}")
                    #logger.print(f"{component}: quantity: {quantity}, order_size: {order_size}, price: {component_prices[component]}")
                    component_orders.setdefault(component, []).append(Order(component, component_prices[component], order_size))
            
        return component_orders
    
    def execute_spread_orders(self, state, target_pos, basket_pos, depth, basket):
        # if synthetics and baskets are equal, no opportuntiy
        #if target_pos == basket_pos:
            #return {}

        target_quantity = abs(target_pos - basket_pos)
        basket_depth = depth[basket]
        synthetic_book = self.get_synthetic_book(depth, basket)
        
        
        #logger.print(f"Trying to execute spread orders, target pos: {target_pos}")
        # if position wants to be greater than basket position, short synthetic and buy basket
        if target_pos > 0:
            #basket_ask_price = min(basket_depth.sell_orders.keys())
            #basket_ask_volume = abs(basket_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_book.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_book.buy_orders[synthetic_bid_price])
            
            #orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            order_size = synthetic_bid_volume #min(orderbook_volume, target_quantity)# #
            
            #basket_orders = [Order(basket, basket_ask_price, order_size)]
            synthetic_orders = [Order('SYNTHETIC', synthetic_bid_price, -order_size)]
            
            aggregate_orders = self.convert_synthetic_basket_orders(state, synthetic_orders, depth, basket)
            #aggregate_orders[basket] = basket_orders
            return aggregate_orders
        elif target_pos < 0:
            #otherwise, position wants to be lower than basket position, long synthetic and short baskets
            #basket_bid_price = max(basket_depth.buy_orders.keys())
            #basket_bid_volume = abs(basket_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_book.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_book.sell_orders[synthetic_ask_price])
            
            #orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            order_size = synthetic_ask_volume #min(orderbook_volume, target_quantity)# #
            
            #basket_orders = [Order(basket, basket_bid_price, -order_size)]
            synthetic_orders = [Order('SYNTHETIC', synthetic_ask_price, order_size)]
            
            aggregate_orders = self.convert_synthetic_basket_orders(state, synthetic_orders, depth, basket)
            #aggregate_orders[basket] = basket_orders
            return aggregate_orders
        elif target_pos == 0:
            synthetic_orders = [Order('SYNTHETIC', -1, 0)]
            return self.convert_synthetic_basket_orders(state, synthetic_orders, depth, basket)
        return {}
        
    def buy_baskets(self, target_pos_1, target_pos_2, basket_pos_1, basket_pos_2, basket_1, basket_2, depth):
        if target_pos_1 == basket_pos_1 and target_pos_2 == basket_pos_2:
            return {}
        
        target_quantity_1 = abs(target_pos_1 - basket_pos_1)
        target_quantity_2 = abs(target_pos_2 - basket_pos_2)
        basket_1_depth = depth[basket_1]
        basket_2_depth = depth[basket_2]
        
        #logger.print(f"Basket1: target pos: {target_pos_1}, target_quantity: {target_quantity_1}, pos: {basket_pos_1}")
        #logger.print(f"Basket2: target pos: {target_pos_2}, target_quantity: {target_quantity_2}, pos: {basket_pos_2}")

        
        basket_depth = basket_1_depth
        if target_pos_1 > basket_pos_1:
            basket_ask_price = min(basket_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_depth.sell_orders[basket_ask_price])
            
            order_size_1 = min(basket_ask_volume, target_quantity_1)
            basket_price_1 = basket_ask_price
                        
        else:
            #otherwise, position wants to be lower than basket position, long synthetic and short baskets
            basket_bid_price = max(basket_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_depth.buy_orders[basket_bid_price])
            
            order_size_1 = -basket_bid_volume
            basket_price_1 = basket_bid_price
        
        basket_depth = basket_2_depth
        if target_pos_2 > basket_pos_2:
            basket_ask_price = min(basket_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_depth.sell_orders[basket_ask_price])
            
            order_size_2 = min(basket_ask_volume, target_quantity_2)
            basket_price_2 = basket_ask_price
                        
        else:
            #otherwise, position wants to be lower than basket position, long synthetic and short baskets
            basket_bid_price = max(basket_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_depth.buy_orders[basket_bid_price])
            
            order_size_2 = -basket_bid_volume
            basket_price_2 = basket_bid_price
            
        sign_1 = 1 if order_size_1 > 0 else -1
        sign_2 = 1 if order_size_2 > 0 else -1
        order_size_1 = abs(order_size_1)
        order_size_2 = abs(order_size_2)
        
        order_size_1 = int(min(order_size_1, target_quantity_1)*sign_1)
        order_size_2 = int(min(order_size_2, target_quantity_2)*sign_2)
    
        orders = {}
        orders[basket_1] = [Order(basket_1, basket_price_1, order_size_1)]
        orders[basket_2] = [Order(basket_2, basket_price_2, order_size_2)]

        return orders
                
        
    def spread_orders(self, state):
        depth = state.order_depths
        if 'PICNIC_BASKET1' not in depth or 'PICNIC_BASKET2' not in depth:
            return None
        
        basket_1_depth = depth['PICNIC_BASKET1']
        basket_2_depth = depth['PICNIC_BASKET2']

        synthetic_book_1 = self.get_synthetic_book(depth, 'PICNIC_BASKET1')
        synthetic_book_2 = self.get_synthetic_book(depth, 'PICNIC_BASKET2')

        basket_1_vwmp = self.get_vwap_mid(basket_1_depth)
        synthetic_1_vwmp = self.get_vwap_mid(synthetic_book_1)
        basket_2_vwmp = self.get_vwap_mid(basket_2_depth)
        synthetic_2_vwmp = self.get_vwap_mid(synthetic_book_2)
        
        
        spread_1 = basket_1_vwmp - synthetic_1_vwmp
        spread_2 = basket_2_vwmp - synthetic_2_vwmp
        spread_3 = 2*spread_1 - 3*spread_2
        
        #logger.print(f"Spread3: {spread_3}")
        
        basket = "SYNTHETIC_3"
        window = self.basket_params[basket]['spread_window']
        num_std = self.basket_params[basket]['num_std']
        mean = self.basket_params[basket]['spread_mean']
        lookback = self.basket_params[basket]['lookback']
        target_pos = self.basket_params[basket]['target_pos']
        basket_pos_1 = state.position.get('PICNIC_BASKET1', 0)
        basket_pos_2 = state.position.get('PICNIC_BASKET2', 0)
        
        
        product = basket+"_SPREAD"
        
        self.check_synthetic(state, -basket_pos_1, spread_1, update=True)
        if self.update_boll_band(state, product, spread_3, window, num_std) is None:
            return None
        
        regime = self.identify_regime(state, product, lookback)
        
        if regime is None:
            return None
        
        self.variables.setdefault(product+'_signal', None)
        self.variables[product+'_signal'] = self.generate_signal(product, regime, mean, strict=True)
        
        
        if self.variables[product+'_signal'] is None:
            logger.print("No signal")
            return None
        
        
        #logger.print(f"Signal: {self.variables[product+'_signal']} for {product}")
        #logger.print(f"Regime: {regime}")
        

        ratio1 = 2
        ratio2 = 3
        
        if self.variables[product+'_signal'] == "SELL":
            #sell p1 buy p2
            return {**self.buy_baskets(-target_pos * ratio1, target_pos * ratio2, basket_pos_1, basket_pos_2, 'PICNIC_BASKET1', 'PICNIC_BASKET2', depth), **self.check_synthetic(state, -2*target_pos, spread_1, update=False)}
        elif self.variables[product+'_signal'] == "BUY":
            #buy p1 sell p2
            return {**self.buy_baskets(target_pos * ratio1, -target_pos * ratio2, basket_pos_1, basket_pos_2, 'PICNIC_BASKET1', 'PICNIC_BASKET2', depth), **self.check_synthetic(state, 2*target_pos, spread_1, update=False)}
        elif self.variables[product+'_signal'] == "CLEAR":
            return {**self.buy_baskets(0, 0, basket_pos_1, basket_pos_2, 'PICNIC_BASKET1', 'PICNIC_BASKET2', depth), **self.check_synthetic(state, 0, spread_1)}
        #self.variables[basket+"_prev_z"] = z_score
        return None
    
    
    def check_synthetic(self, state, target_pos, spread, update=True):
        depth = state.order_depths
        basket = "PICNIC_BASKET1"
        window = self.basket_params[basket]['spread_window']
        num_std = self.basket_params[basket]['num_std']
        mean = self.basket_params[basket]['spread_mean']
        lookback = self.basket_params[basket]['lookback']
        #target_pos = self.basket_params[basket]['target_pos']
        vol_multiplier = self.basket_params[basket]['vol_multiplier']
        vol_window = self.basket_params[basket]['vol_window']
        
        product = basket+"_SPREAD"
        jams_pos = state.position.get('JAMS', 0)
        croissants_pos = state.position.get('CROISSANTS', 0)
        djembes_pos = state.position.get('DJEMBES', 0)
        
        self.variables.setdefault('BASKET1_POS', 0)
        basket_pos = self.variables['BASKET1_POS']
        #basket_pos = state.position.get(basket, 0)
        
        j_weight = self.basket_weights[basket]['JAMS']
        c_weight = self.basket_weights[basket]['CROISSANTS']
        d_weight = self.basket_weights[basket]['DJEMBES']
        
        j_bal = jams_pos*-j_weight == basket_pos
        c_bal = croissants_pos*-c_weight == basket_pos
        d_bal = djembes_pos*-d_weight == basket_pos
        
        balanced = True if j_bal and c_bal and d_bal else False
        
        
        
        if update:
            if self.update_boll_band(state, product, spread, window, num_std, vol_multiplier, vol_window) is None:
                return {}
        
        regime = self.identify_regime(state, product, lookback)
        
        if regime is None:
            logger.print("No regime for spread_1")
            return {}
        
        self.variables.setdefault(product+'_signal', None)
        self.variables[product+'_signal'] = self.generate_signal(product, regime, mean, strict=True)
        signal = self.variables[product+'_signal']
        
        if self.variables[product+'_signal'] is None:
            logger.print("No signal")
            return {}
        
        
        #logger.print(f"Signal: {self.variables[product+'_signal']} for {product}")
        #logger.print(f"Regime: {regime}")
        
        last_target_pos = self.variables.get(product+'last_target_pos', 0)
        self.variables[product+'last_target_pos'] = target_pos
        
        if not update:
            if signal == "SELL" and target_pos < 0:
                #sell p1 buy p2
                #logger.print(f"Trying to sell spread at {target_pos} target_pos")
                return self.execute_spread_orders(state, -target_pos, basket_pos, depth, basket)
            elif signal == "BUY" and target_pos > 0:
                #buy p1 sell p2
                #logger.print("trying to buy spread at {target_pos} target pos")
                return self.execute_spread_orders(state, -target_pos, basket_pos, depth, basket)
            elif signal == "CLEAR" or target_pos != last_target_pos :  
                return self.execute_spread_orders(state, 0, basket_pos, depth, basket)
        else:
            if not balanced and basket_pos < 0 and signal == "SELL":
                return self.execute_spread_orders(state, -basket_pos, basket_pos, depth, basket)
            elif not balanced and basket_pos > 0 and signal == "BUY":
                return self.execute_spread_orders(state, -basket_pos, basket_pos, depth, basket)
            elif signal == "CLEAR":
                return self.execute_spread_orders(state, 0, basket_pos, depth, basket)
        return {}
    
    
            
            
    def get_vwap_mid(self, product_depth):
        buy_orders = product_depth.buy_orders
        sell_orders = product_depth.sell_orders
        
        best_bid = max(buy_orders.keys()) if buy_orders else 0
        bid_volume = abs(buy_orders[best_bid]) if buy_orders else 0
        best_ask = min(sell_orders.keys()) if sell_orders else 0
        ask_volume = abs(sell_orders[best_ask]) if sell_orders else 0
        
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0

        
        return (best_bid * ask_volume + best_ask * bid_volume)/(total_volume)
    
    def update_boll_band(self, state, product, value, window, num_std, ema=0, vol_multiplier=0, vol_window=0):
        self.data.setdefault(product, []).append(value)
        if len(self.data[product]) < window:
            return None
        
        sma = np.mean(self.data[product][-window:])
        std = np.std(self.data[product][-window:])
        if ema != 0:
            price = self.ema_with_halflife(self.data[product], ema)[-1]
        else:
            price = self.data[product][-1]
            
        if vol_multiplier != 0 and vol_window != 0:
            vol = np.std(self.data[product][-vol_window:])
            num_std *= vol_multiplier * vol/std
        
        #self.data.setdefault(product+"_UPPER", []).append(sma + std*num_std)
        #self.data.setdefault(product+"_LOWER", []).append(sma - std*num_std)
        upper = sma + std*num_std
        lower = sma - std*num_std
        
        above_upper = True if price >= upper else False
        below_lower = True if price <= lower else False
        
        self.variables.setdefault(product+'_last_upper_signal', -1)
        self.variables.setdefault(product+'_last_lower_signal', -1)
        
        if above_upper:
            self.variables[product+'_last_upper_signal'] = state.timestamp
        if below_lower:
            self.variables[product+'_last_lower_signal'] = state.timestamp
        
        self.variables.setdefault(product+'_last_upper', 0)
        self.variables.setdefault(product+'_upper', 0)
        self.variables.setdefault(product+'_last_lower', 0)
        self.variables.setdefault(product+'_lower', 0)
        
        self.variables[product+'_last_upper'] = self.variables[product+'_upper']
        self.variables[product+'_upper'] = upper
        self.variables[product+'_last_lower'] = self.variables[product+'_lower']
        self.variables[product+'_lower'] = lower
        
        return 1
        
    def ema_with_halflife(self, data, halflife):
        data = np.asarray(data, dtype=np.float64)
        alpha = 1 - np.exp(np.log(0.5) / halflife)
        
        ema = np.zeros_like(data)
        ema[0] = data[0]  # initialize
        
        for t in range(1, len(data)):
            ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    
        return ema

    def identify_regime(self, state, product, lookback):
        if len(self.data.get(product, [])) < lookback:
            logger.print("Not enough data for regime")
            return None
        
        last_upper_signal = self.variables[product+'_last_upper_signal']
        last_lower_signal = self.variables[product+'_last_lower_signal']
        timestamp = state.timestamp
        
        if last_upper_signal == -1 and last_lower_signal == -1:
            return None

        
        recent_above = True if (last_upper_signal - timestamp)/100 <= lookback else False
        recent_below = True if (last_upper_signal - timestamp)/100 <= lookback else False

        
        if recent_above and recent_below:
            return 'OSCILLATING'
        elif recent_above:
            return 'RIDING_UPPER'
        elif recent_below:
            return 'RIDING_LOWER'
        else:
            return 'NEUTRAL'
        
    def generate_signal(self, product, regime, true_mean, strict=False):
            
        price = self.data[product][-1]
        last_price = self.data[product][-2]
        upper = self.variables[product+'_upper']
        last_upper = self.variables[product+'_last_upper']
        lower = self.variables[product+'_lower']
        last_lower = self.variables[product+'_last_lower']

        signal = self.variables[product+'_signal']
        
        if strict:
            upper_req = price > true_mean
            lower_req = price < true_mean
        else:
            upper_req = True
            lower_req = True
        
        if regime == "OSCILLATING":
            if price >= upper and last_price < last_upper and upper_req:
                signal = "SELL"
            elif price <= lower and last_price > last_lower and lower_req:
                signal = "BUY"
        elif regime == "RIDING_UPPER":
            if price > true_mean:
                signal == "CLEAR"
            else:
                signal == "BUY"
        elif regime == "RIDING_LOWER":
            if price < true_mean:
                signal == "CLEAR"
            else:
                signal == "SELL"
        elif regime == "NEUTRAL":
            signal == "CLEAR"
        
        return signal
        
    def squid_ink_returns(self, state):
        depth = state.order_depths
        
        squid_ink_depth = depth['SQUID_INK']
        price = self.get_vwap_mid(squid_ink_depth)
        
        basket = "SQUID_INK_RETURNS"
        window = self.basket_params[basket]['spread_window']
        num_std = self.basket_params[basket]['num_std']
        mean = self.basket_params[basket]['spread_mean']
        lookback = self.basket_params[basket]['lookback']
        target_pos = self.basket_params[basket]['target_pos']
        position = state.position.get('SQUID_INK', 0)
        
        
        product = basket+"_RETURNS"
        
        self.variables.setdefault('SQUID_INK_LAST_PRICE', 0)
        last_price = self.variables['SQUID_INK_LAST_PRICE']
        current_return = price/last_price - 1 if last_price != 0 else 0
        self.variables['SQUID_INK_LAST_PRICE'] = price

        #if self.update_boll_band(state, product, current_return, window, num_std) is None:
            #return None
        
        #regime = self.identify_regime(state, product, lookback)
        
        #if regime is None:
            #return None
        
        self.variables.setdefault(product+'_signal', None)
        #self.variables[product+'_signal'] = self.generate_signal(product, regime, mean, strict=True)
        
        
        
        self.data.setdefault(product, []).append(current_return)
        if len(self.data[product]) < window:
            return None
        
        sma = np.mean(self.data[product][-window:-1])
        std = np.std(self.data[product][-window:-1])
        z_score = (current_return - sma)/std
        

        if abs(z_score) < num_std:
            self.variables[product+"_signal"] = "SELL"
        else:
            self.variables[product+"_signal"] = "CLEAR"
        
        if self.variables[product+'_signal'] is None:
            return None
        
        best_bid = max(squid_ink_depth.buy_orders.keys())
        best_ask = min(squid_ink_depth.sell_orders.keys())
        bid_volume = squid_ink_depth.buy_orders[best_bid]
        ask_volume = abs(squid_ink_depth.sell_orders[best_ask])
        
            
        if self.variables[product+'_signal'] == "SELL":
            INK_make_orders, _, _ = self.make_orders(
                Product.INK,
                squid_ink_depth,
                price,
                position,
                15,
                15,
                self.params[Product.INK]["disregard_edge"],
                self.params[Product.INK]["join_edge"],
                self.params[Product.INK]["default_edge"],
            )
            return {'SQUID_INK': INK_make_orders}
        elif self.variables[product+'_signal'] == "CLEAR":
            if position > 0:
                order_price = best_bid
            else:
                order_price = best_ask
            return {'SQUID_INK': [Order('SQUID_INK', order_price, -position)]}
        #self.variables[basket+"_prev_z"] = z_score
        return None
    
    def update_ema(self, prev_ema, new_value, alpha):
        return (1 - alpha) * prev_ema + alpha * new_value
        
    def get_options_orders(self, state):
        DAY = 3
        
        orders = {}
        options = {
            9500: {'iv_mean': self.get_expected_iv(9500), 'weight': .5},
            9750: {'iv_mean': self.get_expected_iv(9750), 'weight': .5},
            10000: {'iv_mean': self.get_expected_iv(10000), 'weight': .5},
            10250: {'iv_mean': self.get_expected_iv(10250), 'weight': .5},
            10500: {'iv_mean': self.get_expected_iv(10500), 'weight': .5}
        }
        self.variables.setdefault('options_data', options)
        options = self.variables['options_data']
        
        valid_trades = {key: {} for key in options.keys()}
        positions = [0, 0, 0, 0, 0]
        total_delta = 0
        S = self.get_vwap_mid(state.order_depths['VOLCANIC_ROCK'])
        
        
        #update log returns,
        
        self.data.setdefault("ROCK", [])
        self.data["ROCK"].append(S)

        window = 100
        self.variables.setdefault('rv', 0)
        
        rv_series = self.realized_vol(self.data["ROCK"], window=window)
    
        if not rv_series.dropna().empty:
            self.variables['rv'] = rv_series.dropna().iloc[-1]
        else:
            self.variables['rv'] = np.nan 

        logger.print(f"RV: {self.variables['rv']}")
        rv = self.variables['rv']
        
        
        for strike, option in options.items():
            product = 'VOLCANIC_ROCK_VOUCHER_'+str(strike)
            
            #set vars
            C_market = self.get_vwap_mid(state.order_depths[product])
            
            K = int(strike)
            T = (7 - DAY)/365
            
            #get iv/greeks
            option['iv'] = self.implied_vol_newton(C_market, S, K, T)
            option['vega'] = self.bs_vega(S, K, T, option['iv'])
            option['delta'] = self.bs_delta(S, K, T, option['iv'])
            option['ATM'] = True if abs(S - int(strike)) <= 500 else False
            
            if np.isnan(option['iv']) or np.isnan(option['vega']) or np.isnan(option['delta']):
                continue
            
            total_delta += option['delta']*state.position.get(product, 0)
            
            #rolling iv_vs_rv and iv_mean_revert
            prev_iv_rv = options.get("iv_rv", 0)
            prev_iv_mean_rev = option.get("iv_mean_revert", 0)
            iv_rv = option['iv'] - rv
            iv_mean_rev = option['iv'] - option['iv_mean']
            
            alpha = 1
            option['iv_rv'] = self.update_ema(prev_iv_rv, iv_rv, alpha/(window+1))
            option['iv_mean_revert'] = self.update_ema(prev_iv_mean_rev, iv_mean_rev, alpha/(window+1))
            
            
            if state.timestamp/100 < window:
                continue
            
            #signal
            signal = self.generate_volatility_signal(option['iv_mean_revert'], option['iv_rv'], weight_mean_reversion = option['weight'], threshold=.004)
            if not option['ATM']:
                signal = 0
            
            #logger.print(f'OSig: {signal}, Strike: {strike}')
            
            valid_trades[strike]['delta'] = option['delta']
            valid_trades[strike]['vega'] = option['vega']
            valid_trades[strike]['signal'] = signal
            valid_trades[strike]['strike'] = strike
            
            if signal == 0:
                valid_trades[strike]['target_pos'] = 0

        
        
        # optimize vega/delta, make best trade based on vega/delta
        deltas = []
        vegas = []
        
        for strike, trade in valid_trades.items():
            if trade is None:
                deltas.append(0)
                vegas.append(0)
            else:
                deltas.append(trade.get('delta', 0)*trade.get('signal',0))
                vegas.append(trade.get('vega', 0))
            
        
        #if state.timestamp/100 % 10 == 0:
        positions, position_delta = self.optimize_positions(deltas, vegas, previous_r=self.variables.get('prev_r', [0, 0, 0, 0, 0]))
        self.variables['prev_r'] = positions.tolist()
        
        logger.print(f'opt: {positions}, total delta: {int(total_delta)}')
        
        for index, trade in enumerate(valid_trades.values()):
            if trade.get('strike', None) is not None:
                options[trade['strike']]['target_pos'] = positions[index]*trade.get('signal', 0)
                logger.print(f'target_pos: {positions[index]}, signal: {trade["signal"]}')
            
        
        # update positions
        threshold = 10
        for strike, option in options.items():
            product = "VOLCANIC_ROCK_VOUCHER_"+str(strike)
            position = state.position.get(product, 0)
            target_pos = option.get('target_pos', 0)
            sign = 1 if position > 0 else -1
            target_sign = 1 if target_pos > 0 else -1
            profit = abs(option.get('buy_price', 0) - self.get_vwap_mid(state.order_depths[product]))*position
            change = True if sign != target_sign else False
            if position != target_pos and abs(self.variables.get(f'last_target_pos_{product}', 0) - target_pos) > threshold or (change and profit >= 0):
                orders[product] = self.place_target_order(state, product,  target_pos)
                option['buy_price'] = self.get_vwap_mid(state.order_depths[product])
            self.variables[f'last_target_pos_{product}'] = target_pos
                
            
            
        # hedge delta
        measured_delta = position_delta
        measured_delta = total_delta
        
        position = state.position.get('VOLCANIC_ROCK', 0)        
        threshold = 0
        if state.timestamp/100 % 100 == 0:
            target_pos = int(measured_delta)
            if position != target_pos and abs(target_pos - self.variables.get('last_total_delta', 0)) > threshold: 
                orders['VOLCANIC_ROCK'] = self.place_target_order(state, 'VOLCANIC_ROCK', target_pos)
        
        
        self.variables['last_total_delta'] = target_pos    
        self.variables['options_data'] = options
        
        return orders
    
    def generate_volatility_signal(self, iv_mean_revert, iv_vs_rv, weight_mean_reversion=0.5, threshold=0.05):
    
        # Combine signals into one score
        score = weight_mean_reversion * iv_mean_revert + (1 - weight_mean_reversion) * iv_vs_rv
        logger.print(f"score: {score}, iv_rv: {iv_vs_rv} iv_mean_rev: {iv_mean_revert}")

        # Generate trading signal: 1 = long vega, -1 = short vega, 0 = do nothing
        signal = 0
        if score > threshold: signal = -1  # Short vega (sell option)
        if score < -threshold: signal = 1  # Long vega (buy option)
    
        return signal   
    
    def place_target_order(self, state, product, target_pos):
        depth = state.order_depths[product]
        position = state.position.get(product, 0)
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
        best_bid_volume = depth.buy_orders[best_bid] if depth.buy_orders else 0
        best_ask_volume = depth.sell_orders[best_ask] if depth.sell_orders else 0
        
        sign = 1 if target_pos - position > 0 else -1
        
        price = best_bid if sign < 0 else best_ask
        volume = best_bid_volume if sign < 0 else best_ask_volume
        
        if price == 0 or volume == 0:
            return []
        
        availability = self.get_liquidity(state, product, 0, sign)
        order_size = min(abs(volume), abs(availability))*sign
        
        return [Order(product, price, order_size)]
    
    def optimize_positions(
            self, d, v,
            r_bounds=(0, 200),
            dot_product_bounds=(-400, 400),
            num_samples=10000,
            batch_size=1000,
            previous_r=None,
            mutation_strength=5
            ):
        d = np.array(d)
        v = np.array(v)
        assert len(d) == len(v), "d and v must be the same length"
    
        num_vars = len(d)
        full_r = np.zeros(num_vars, dtype=int)
        mask_nonzero = d != 0
        active_indices = np.where(mask_nonzero)[0]
        d_active = d[mask_nonzero]
        v_active = v[mask_nonzero]
        dim = len(d_active)
    
        best_value_sum = -float('inf')
        best_r_partial = None
        best_dot_product = None
    
        for i in range(num_samples // batch_size):
            r_samples = np.random.randint(r_bounds[0], r_bounds[1] + 1, size=(batch_size, dim))
    
            # Insert mutated warm-start solution at top of batch
            if previous_r is not None:
                prev_r_partial = np.array(previous_r)[active_indices]
                mutated = prev_r_partial + np.random.randint(-mutation_strength, mutation_strength + 1, size=dim)
                mutated = np.clip(mutated, r_bounds[0], r_bounds[1])
                r_samples[0] = mutated
    
            dot_products = r_samples @ d_active
            value_sums = r_samples @ v_active
    
            mask_valid = (dot_products >= dot_product_bounds[0]) & (dot_products <= dot_product_bounds[1])
            if not np.any(mask_valid):
                continue
    
            valid_values = value_sums[mask_valid]
            valid_r = r_samples[mask_valid]
            valid_dot = dot_products[mask_valid]
    
            max_idx = np.argmax(valid_values)
            if valid_values[max_idx] > best_value_sum:
                best_value_sum = valid_values[max_idx]
                best_r_partial = valid_r[max_idx]
                best_dot_product = valid_dot[max_idx]
    
        if best_r_partial is not None:
            full_r[active_indices] = best_r_partial
    
        return full_r, best_dot_product

    def realized_vol(self, prices, window=1000, step_size=100):
        prices = np.asarray(prices)
        shifted = np.roll(prices, step_size)
        shifted[:step_size] = np.nan
        log_returns = np.log(prices / shifted)
        log_returns[:step_size] = np.nan
     
        dt = step_size / (365 * 10000)
     
        squared_returns = log_returns**2
     
        # Auto alpha based on smoothing window
        alpha = None
        if alpha is None:
            window = 100
            alpha = 2 / (window + 1)
     
        rv_squared = pd.Series(squared_returns).ewm(alpha=alpha, adjust=False).mean()
        rv = np.sqrt(rv_squared / dt)
    
        return rv #np.sqrt(rolling_rv)

    
    def get_log_return(self, price_now, price_prev):
        if price_prev < 1e-6 or price_now < 1e-6:
           return 0
        log_ret = np.log(price_now / price_prev)
        return log_ret
    
    def get_expected_iv(self, strike):
        historical_smile = [1.39857394e-07, -2.87781744e-03,  1.49626367e+01]
        expected_iv = np.polyval(historical_smile, strike)
        return expected_iv

    def norm_cdf(self, x):
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))

    def norm_pdf(self, x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    # Black-Scholes call price
    def bs_call_price(self, S, K, T, sigma):
        if sigma <= 0 or T <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)

    # Vega (derivative of price w.r.t. sigma)
    def bs_vega(self, S, K, T, sigma):
        if sigma <= 0 or T <= 0:
            return 0
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return S * self.norm_pdf(d1) * np.sqrt(T)
    
    def bs_delta(self, S, K, T, sigma):
        if sigma <= 0 or T <= 0:
            return 0.0
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return self.norm_cdf(d1)  # For a call option

    def implied_vol_newton(self, C_market, S, K, T, sigma_init=0.15, tol=1e-5, max_iter=100):
        #C_market = market call price
        #S = spot price
        #K = strike
        #T = time to expiry
        
        sigma = sigma_init
        for i in range(max_iter):
            price = self.bs_call_price(S, K, T, sigma)
            vega = self.bs_vega(S, K, T, sigma)
    
            if vega == 0:
                break  # avoid division by 0
            
            diff = price - C_market
            sigma -= diff / vega
    
            if abs(diff) < tol:
                return sigma
    
        return np.nan  # If not converged
    
    def run(self, state: TradingState):
        #for log visualization
        result = {}
        conversions = 0
        trader_data = ""
        
        
        stable = ['RAINFOREST_RESIN']
        vwap_products = []
        returns_symbols = []
        
        for symbol in state.own_trades.keys():
            if symbol == "DJEMBES":
                for trade in state.own_trades[symbol]:
                    self.variables['BASKET1_POS'] += trade.quantity
    
        
        #set limits for commodities
        self.get_limits()
        
        
        #initalize or update
        if state.traderData == "":
            self.initialize_data(returns_symbols)
        else:
            self.deserialize(state.traderData)
            self.update_data(vwap_products=vwap_products, returns_symbols=returns_symbols, state=state) 
            
        # set basket weights
        self.basket_weights = {
            'PICNIC_BASKET1': {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 1},
            'PICNIC_BASKET2': {'CROISSANTS': 4, 'JAMS': 2, 'DJEMBES': 0},
            'SYNTHETIC_3': {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 0},
            #'SYNTHETIC_4': {'CROISSANTS': 12, 'JAMS': 8, 'DJEMBES': 1},
        }
        self.basket_params = {
            'PICNIC_BASKET1': {'spread_window': 100, 'lookback': 100, 'spread_mean': 24, 'num_std': 1.5, 'ema': 5, 'target_pos': 60, 'vol_multiplier': 2, 'vol_window': 45},
            'PICNIC_BASKET2': {'spread_window': 50, 'lookback': 100, 'spread_mean': 31.792656681639464, 'num_std': 1, 'ema': 0, 'target_pos': 15},
            'SYNTHETIC_3': {'spread_window': 100, 'lookback': 100, 'spread_mean': 0, 'num_std': 3, 'ema': 2, 'target_pos': 30},
            'D-C': {'spread_window': 100, 'lookback': 100, 'spread_mean': 9138.39, 'num_std': 1, 'ema': 1, 'target_pos': 60, 'vol_multiplier': 2, 'vol_window': 45},
            'SQUID_INK_RETURNS': {'spread_window': 100, 'lookback': 30, 'spread_mean': 0, 'num_std': 3, 'ema': 2, 'target_pos': 50, 'vol_multiplier': 0, 'vol_window': 5}
        }
    
        #init result dict
        result = {}
                
        #skip these
        skip = ['CROISSANTS', 'JAMS', 'DJEMBES']

                    
        for product in state.order_depths:
            if product in skip:
                continue
            
            #init variables for all products
            depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            position = state.position.get(product, 0)
            total_pos_change = 0
        
            if product in stable:
                if product == "RAINFOREST_RESIN":
                    RESIN_position = (
                        state.position[Product.RESIN]
                        if Product.RESIN in state.position
                        else 0
                    )
                    RESIN_take_orders, buy_order_volume, sell_order_volume = (
                        self.take_orders(
                            Product.RESIN,
                            state.order_depths[Product.RESIN],
                            self.params[Product.RESIN]["fair_value"],
                            self.params[Product.RESIN]["take_width"],
                            RESIN_position,
                        )
                    )
                    RESIN_clear_orders, buy_order_volume, sell_order_volume = (
                        self.clear_orders(
                            Product.RESIN,
                            state.order_depths[Product.RESIN],
                            self.params[Product.RESIN]["fair_value"],
                            self.params[Product.RESIN]["clear_width"],
                            RESIN_position,
                            buy_order_volume,
                            sell_order_volume,
                        )
                    )
                    RESIN_make_orders, _, _ = self.make_orders(
                        Product.RESIN,
                        state.order_depths[Product.RESIN],
                        self.params[Product.RESIN]["fair_value"],
                        RESIN_position,
                        buy_order_volume,
                        sell_order_volume,
                        self.params[Product.RESIN]["disregard_edge"],
                        self.params[Product.RESIN]["join_edge"],
                        self.params[Product.RESIN]["default_edge"],
                        True,
                        self.params[Product.RESIN]["soft_position_limit"],
                    )
                    result[Product.RESIN] = (
                        RESIN_take_orders + RESIN_clear_orders + RESIN_make_orders
                    )
                    continue
                    
                    
            else:  

                if product == "SQUID_INK":
                    
                    if product == "SQUID_INK":
                        orders = self.squid_ink_returns(state)
                        if orders is not None:
                            if 'SQUID_INK' in orders:
                                result['SQUID_INK'] = orders['SQUID_INK']
                
                if product == "KELP":
                    if Product.KELP in self.params and Product.KELP in state.order_depths:
                        KELP_position = (
                            state.position[Product.KELP]
                            if Product.KELP in state.position
                            else 0
                        )
                        KELP_fair_value = self.KELP_fair_value(
                            state.order_depths[Product.KELP], self.traderObject
                        )
                        KELP_take_orders, buy_order_volume, sell_order_volume = (
                            self.take_orders(
                                Product.KELP,
                                state.order_depths[Product.KELP],
                                KELP_fair_value,
                                self.params[Product.KELP]["take_width"],
                                KELP_position,
                                self.params[Product.KELP]["prevent_adverse"],
                                self.params[Product.KELP]["adverse_volume"],
                            )
                        )
                        KELP_clear_orders, buy_order_volume, sell_order_volume = (
                            self.clear_orders(
                                Product.KELP,
                                state.order_depths[Product.KELP],
                                KELP_fair_value,
                                self.params[Product.KELP]["clear_width"],
                                KELP_position,
                                buy_order_volume,
                                sell_order_volume,
                            )
                        )
                        KELP_make_orders, _, _ = self.make_orders(
                            Product.KELP,
                            state.order_depths[Product.KELP],
                            KELP_fair_value,
                            KELP_position,
                            buy_order_volume,
                            sell_order_volume,
                            self.params[Product.KELP]["disregard_edge"],
                            self.params[Product.KELP]["join_edge"],
                            self.params[Product.KELP]["default_edge"],
                        )
                        result[Product.KELP] = (
                            KELP_take_orders + KELP_clear_orders + KELP_make_orders
                        )
                    continue
            
                
                
                
        #round_2 = ['PICNIC_BASKET1', 'PICNIC_BASKET2']
        #if product in round_2:   
        
        orders1 = self.spread_orders(state)
        orders2 = None#self.component_orders(state)
        
    
        items = ['CROISSANTS', 'JAMS', 'DJEMBES', 'PICNIC_BASKET1', 'PICNIC_BASKET2']
        #items = ['PICNIC_BASKET1', 'PICNIC_BASKET2']
        for item in items:
            if orders1 is not None:
                if item in orders1:
                    result.setdefault(item, []).extend(orders1[item])
            if orders2 is not None:
                if item in orders2:
                    result.setdefault(item, []).extend(orders2[item])
                    
            
        #ROUND 3
        
        items = [
            'VOLCANIC_ROCK',
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500'
        ]
        
        orders = self.get_options_orders(state)
        for item in items:
            if orders is not None:
                if item in orders:
                    result.setdefault(item, []).extend(orders[item])
    
    
        traderData = self.serialize()
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, traderData
