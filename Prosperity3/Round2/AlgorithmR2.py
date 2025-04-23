# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:55:54 2025

@author: Ryan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:51:05 2025

@author: rspaeth1
"""

import json
import jsonpickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List

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
        
    
    #define commodity limits here
    def get_limits(self):
        self.limits = {
            'RAINFOREST_RESIN': 50,
            'KELP': 50,
            'SQUID_INK': 50,
            'CROSSAINTS': 250,
            'JAMS': 350,
            'DJEMBES': 60,
            'PICNIC_BASKET1': 60,
            'PICNIC_BASKET2': 100
        }
            
        
    #check at trading time if we are going to exceed limits, return available space for commodity
    def get_liquidity(self, state, product, total_pos_change, side = "BUY"):
        #take into account the other orders we have already put in
        position = state.position.get(product,0)
        
        if side == "BUY":
            return self.limits[product] - position - total_pos_change
        else:
            return -self.limits[product] - position - total_pos_change 
        
        
    
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
                self.data[symbol] = self.data[symbol][-50:]
        
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
        
    def euclidean_distance(self, a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def predict_cluster(self, features):
        closest = None
        min_dist = float('inf')
        for i, center in enumerate(self.cluster_centers):
            dist = self.euclidean_distance(features, center)
            if dist < min_dist:
                min_dist = dist
                closest = i
        return closest
    
    def get_recent_trades(self, state, product):
        if product in state.market_trades:
            for trade in state.market_trades[product]:
                trade_price = trade.price
                trade_time = trade.timestamp
                trade_qty = trade.quantity
                
                
                #get best bid ask
                best_bid = min(state.order_depths[product].buy_orders.keys())
                best_ask = max(state.order_depths[product].sell_orders.keys())
                
                #infer aggressor
                if trade_price >= best_ask:
                    aggressor = "buyer"
                elif trade_price <= best_bid:
                    aggressor = "seller"
                else:
                    aggressor = "unknown"
                    
                #store
                self.data.setdefault(product+"_TRADES", []).append({
                    'timestamp': trade_time,
                    'price': trade_price,
                    'quantity': trade_qty,
                    'aggressor': aggressor
                })
                
            
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
            implied_bid += best_bid[component] * self.basket_weights[basket][component]
            implied_ask -= best_ask[component] * self.basket_weights[basket][component]
            
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
    
    def convert_synthetic_basket_orders(self, synthetic_orders, depth, basket):
        # init dict to store orders per component
        components = ['CROISSANTS', 'JAMS', 'DJEMBES']
        component_orders = {component: [] for component in components}
        
        # get best bid / ask for synthetic basket
        synthetic_book = self.get_synthetic_book(depth, basket)
        best_bid = max(synthetic_book.buy_orders.keys()) if synthetic_book.buy_orders else 0
        best_ask = min(synthetic_book.sell_orders.keys()) if synthetic_book.sell_orders else 0
        
        # iterate each synthetic order
        for order in synthetic_orders:
            #get price and quantity
            price = order.price
            quantity = order.quantity
            
            component_prices = {}
            
            # check if the synthetic basket order aligns with best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy, trade components at best ask prices
                for component in components:
                    component_prices[component] = min(depth[component].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell, trade components at best bid prices
                for component in components:
                    component_prices[component] = max(depth[component].buy_orders.keys())
            else:
                # synthetic basket doesn't align with best bid or ask
                continue
            
            # create orders for each component
            component_orders = {}
            for component in components:
                component_orders.setdefault(component, []).append(Order(component, component_prices[component], quantity*self.basket_weights[basket][component]))
            
        return component_orders
    
    def execute_spread_orders(self, target_pos, basket_pos, depth, basket):
        # if synthetics and baskets are equal, no opportuntiy
        if target_pos == basket_pos:
            return None
        

        target_quantity = abs(target_pos - basket_pos)
        basket_depth = depth[basket]
        synthetic_book = self.get_synthetic_book(depth, basket)
        
        # if position wants to be greater than basket position, short synthetic and buy basket
        if target_pos > basket_pos:
            basket_ask_price = min(basket_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_depth.sell_orders[basket_ask_price])
            
            synthetic_bid_price = max(synthetic_book.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_book.buy_orders[synthetic_bid_price])
            
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            order_size = min(orderbook_volume, target_quantity)
            
            basket_orders = [Order(basket, basket_ask_price, order_size)]
            synthetic_orders = [Order('SYNTHETIC', synthetic_bid_price, -order_size)]
            
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, depth, basket)
            aggregate_orders[basket] = basket_orders
            return aggregate_orders
        else:
            #otherwise, position wants to be lower than basket position, long synthetic and short baskets
            basket_bid_price = max(basket_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_depth.buy_orders[basket_bid_price])
            
            synthetic_ask_price = min(synthetic_book.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_book.sell_orders[synthetic_ask_price])
            
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            order_size = min(orderbook_volume, target_quantity)
            
            basket_orders = [Order(basket, basket_bid_price, -order_size)]
            synthetic_orders = [Order('SYNTHETIC', synthetic_ask_price, order_size)]
            
            aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, depth, basket)
            aggregate_orders[basket] = basket_orders
            return aggregate_orders
        
    def spread_orders(self, depth, basket, basket_pos):
        basket_depth = depth[basket]
        synthetic_book = self.get_synthetic_book(depth, basket)
        basket_vwap = self.get_vwap_mid(basket_depth)
        synthetic_vwap = self.get_vwap_mid(synthetic_book)
        spread = basket_vwap - synthetic_vwap
        self.data.setdefault(basket+'_spread_history', []).append(spread)
        
        if len(self.data[basket+'_spread_history']) < self.basket_params[basket]['spread_window']:
            return None
        
        spread_std = np.std(self.data[basket+'_spread_history'])
        spread_mean = self.basket_params[basket]['spread_mean']
        z_score = (spread-spread_mean)/spread_std
        threshold = self.basket_params[basket]['z_threshold']
        target_pos = self.basket_params[basket]['target_pos']
        
        
        if z_score >= threshold:
            if basket_pos != -target_pos:
                return self.execute_spread_orders(-target_pos, basket_pos, depth, basket)
            
        if z_score <= -threshold:
            if basket_pos != target_pos:
                return self.execute_spread_orders(target_pos, basket_pos, depth, basket)
            
        self.variables[basket+"_prev_z"] = z_score
        return None
            
            
    def get_vwap_mid(self, product_depth):
        best_bid = max(product_depth.buy_orders.keys())
        bid_volume = abs(product_depth.buy_orders[best_bid])
        best_ask = min(product_depth.sell_orders.keys())
        ask_volume = abs(product_depth.sell_orders[best_ask])
        
        return (best_bid * ask_volume + best_ask * bid_volume)/(bid_volume + ask_volume)
            
        
    def run(self, state: TradingState):
        #for log visualization
        result = {}
        conversions = 0
        trader_data = ""
        
        
        stable = ['RAINFOREST_RESIN']
        vwap_products = ['SQUID_INK']
        returns_symbols = ['SQUID_INK']
    
        
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
            'PICNIC_BASKET2': {'CROISSANTS': 4, 'JAMS': 2, 'DJEMBES': 0}
        }
        self.basket_params = {
            'PICNIC_BASKET1': {'spread_window': 50, 'spread_mean': 44.64155846485088, 'z_threshold': 1.5, 'target_pos': 15},
            'PICNIC_BASKET2': {'spread_window': 50, 'spread_mean': 31.792656681639464, 'z_threshold': 1.5, 'target_pos': 15}
        }

            
        #squid ink clusters
        self.cluster_centers = [
            [-5.50300940e-03,  1.79465959e-01, -5.15481333e-01,  5.58667054e-01,
             -5.68913380e-01, -1.44676936e-01, -1.34553283e+00,  1.51909091e+00,
             -1.92004128e-01,  3.19987614e-02, -7.98162410e-02, -6.17045495e-02,
             -3.78815464e-03, -6.26182894e-02],
            
            [ 3.45520389e-02, -7.65434686e-02,  5.25293350e-02,  5.76979586e-01,
             -5.70876805e-01, -1.49666688e-01,  6.82753644e-01, -6.54930283e-01,
             -1.92263676e-01, -4.67025941e-01, -2.09824623e-01, -6.90508382e-02,
              2.19759774e-02, -3.99958189e-02],
            
            [ 2.76015425e-02,  6.76377028e-03,  2.00761054e-02, -1.30217706e+00,
              1.35468339e+00, -3.70799630e-02, -1.18593925e+00,  1.21330145e+00,
             -1.81820464e-01,  1.79763207e+00,  6.17463998e-02, -6.40744422e-02,
             -1.33706734e-01,  9.46868119e-01],
            
            [-6.61339175e-02, -1.25125484e-01,  6.20218667e-01, -1.55918123e+00,
              1.55439388e+00,  4.68596791e-01,  6.84863385e-01, -6.54119060e-01,
             -1.92374057e-01,  4.33348048e-01,  1.42664095e-02, -2.45304576e-02,
             -4.37521205e-03, -3.80196766e-02],
            
            [ 6.71814198e-03,  3.95628733e-01, -8.35769748e-01,  5.14952205e-01,
             -5.48987517e-01, -7.19819251e-02, -1.32590293e+00, -4.00644852e-01,
              4.90058604e+00,  3.70511561e-01,  2.79519254e+00,  1.44017699e+00,
             -3.94027608e-02, -1.94846448e-02]
        ]




        #get trades
        self.get_recent_trades(state, 'SQUID_INK')
 
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
                    buy_orders = depth.buy_orders
                    sell_orders = depth.sell_orders
                    
                    # Sorted bid (descending), ask (ascending)
                    sorted_bids = sorted(buy_orders.items(), reverse=True)
                    sorted_asks = sorted(sell_orders.items())
                    
                    # Extract top 3 bid volumes (if they exist)
                    bid_volume_1 = sorted_bids[0][1] if len(sorted_bids) > 0 else 0
                    bid_volume_2 = sorted_bids[1][1] if len(sorted_bids) > 1 else 0
                    bid_volume_3 = sorted_bids[2][1] if len(sorted_bids) > 2 else 0
                    
                    # Extract top 3 ask volumes
                    ask_volume_1 = sorted_asks[0][1] if len(sorted_asks) > 0 else 0
                    ask_volume_2 = sorted_asks[1][1] if len(sorted_asks) > 1 else 0
                    ask_volume_3 = sorted_asks[2][1] if len(sorted_asks) > 2 else 0
                    
                    # Min bid, max ask, midpoint
                    min_bid = min(depth.buy_orders.keys())
                    max_ask = max(depth.sell_orders.keys())
                    midpoint = int((min_bid + max_ask)/2)
                    
                    
                    
                    # Get trade data if available
                    trade_history = self.data.get(product+"_TRADES", [])
                    if len(trade_history) >= 10:
                       
                        #trade quantity and aggressor
                        last_timestamp = trade_history[-1]['timestamp']
                        trades_at_last_tick = [
                            t for t in trade_history if t['timestamp'] == last_timestamp
                        ]
                        
                        # Default to False
                        qty_5 = False
                        qty_10 = False
                        qty_14 = False
                        
                        for trade in trades_at_last_tick:
                            qty = trade['quantity']
                            side = trade['aggressor']
                        
                            if (
                                side in ['buyer', 'seller'] and
                                qty >= 5
                                # You can also add per-trade filters here if needed
                            ):
                                qty_5 = True
                            if (
                                side in ['buyer', 'seller'] and
                                qty >= 10
                                # You can also add per-trade filters here if needed
                            ):
                                qty_10 = True
                            if (
                                side in ['buyer', 'seller'] and
                                qty >= 14
                                # You can also add per-trade filters here if needed
                            ):
                                qty_14 = True


                        buy_qty = sum(t['quantity'] for t in trades_at_last_tick if t['aggressor'] == 'buyer')
                        sell_qty = sum(t['quantity'] for t in trades_at_last_tick if t['aggressor'] == 'seller')
                        

                        # L1 Consumption
                        L1_Consumption = (
                            buy_qty / ask_volume_1 if buy_qty > 0 and ask_volume_1 > 0
                            else sell_qty / bid_volume_1 if sell_qty > 0 and bid_volume_1 > 0
                            else 0
                        )
                        
                        # L2
                        L2_Consumption = (
                            buy_qty / ask_volume_2 if buy_qty > 0 and ask_volume_2 > 0
                            else sell_qty / bid_volume_2 if sell_qty > 0 and bid_volume_2 > 0
                            else 0
                        )
                        
                        # L3
                        L3_Consumption = (
                            buy_qty / ask_volume_3 if buy_qty > 0 and ask_volume_3 > 0
                            else sell_qty / bid_volume_3 if sell_qty > 0 and bid_volume_3 > 0
                            else 0
                        )
                        

                        
                        recent_trades = trade_history[-10:] if len(trade_history) >= 10 else trade_history
                        rolling_volume = sum(t['quantity'] for t in recent_trades)
                        rolling_direction = sum(1 for t in recent_trades if t['aggressor'] == 'buyer') / len(recent_trades)

                        time_since_last_trade = state.timestamp - trade_history[-1]['timestamp'] if trade_history else 0


                        # Returns and volatility
                        recent_vol = np.std(self.data[product+"_RETURNS"][-50:])
                        mean = np.mean(self.data[product+"_RETURNS"][-50:])
                        current_return = self.data[product+"_RETURNS"][-1]
                        z_score = (current_return - mean)/recent_vol
                        
                        features = [
                            rolling_volume,
                            rolling_direction,
                            z_score,
                            bid_volume_1,
                            bid_volume_2,
                            bid_volume_3,
                            ask_volume_1,
                            ask_volume_2,
                            ask_volume_3,
                            L1_Consumption,
                            L2_Consumption,
                            L3_Consumption,
                            time_since_last_trade,
                            recent_vol
                        ]
                        

                        # Standardize the live feature vector
                        feature_means = [28.508624, 0.342836, -0.020399, 19.505016, 6.481006, 0.576823, 18.499718, 7.787172, 0.909142, 0.455988, 0.084706, 0.008395, 338.135498, 1.515699]
                        feature_stds = [10.303549, 0.15584, 1.043431, 10.909242, 11.352722, 3.854053, 10.989926, 11.888883, 4.725908, 0.678711, 0.403701, 0.121575, 352.361114, 0.991207]
                        
                        standardized_features = [
                            (x - mean) / std if std != 0 else 0
                            for x, mean, std in zip(features, feature_means, feature_stds)
                        ]

                        
                        cluster = self.predict_cluster(standardized_features)


                        """
                        logger.print(f"Cluster: {cluster}")
                        logger.print(f"Abs Z-score: {abs(z_score)}")
                        logger.print(f"Recent Vol: {recent_vol}")
                        """
                        
                        #get time since execution
                        ticks_since_execution = (state.timestamp - self.ink_timestamp)/100
                        
                        signal_1 = (
                            qty_14
                            and abs(z_score) > 1
                            and L1_Consumption > 0.4
                        )

                        
                        if signal_1:
                            if cluster == 2:
                                liquidity = self.get_liquidity(state, product, total_pos_change, 'BUY')
   
                                orders.append(Order(product, max_ask, int(liquidity)))
                                total_pos_change += liquidity
                                self.ink_timestamp = state.timestamp
                            elif cluster == 4:
                                liquidity = self.get_liquidity(state, product, total_pos_change, 'SELL')
                                
                                orders.append(Order(product, midpoint - 1, int(liquidity)))
                                total_pos_change += liquidity
                                self.ink_timestamp = state.timestamp
                        elif ticks_since_execution >= 30 and position != 0:
                            #availability = self.check_limits(state, product, side = 'BUY')
                            #order_size = availability/2
                            
                            if position < 0:
                                order_price = max_ask
                            else:
                                order_price = min_bid
                            
                            orders.append(Order(product, order_price, int(-position)))
                            total_pos_change += -position
                    
                            
              
              
                        #if inventory is too high, offload some 
                        liquidate = True
                        if liquidate:
                            inventory_ratio = abs(position / self.limits[product])
                            if inventory_ratio > .95: #change this value to affect inventory reduction
                                
                                # Adaptive position scaling based on inventory ratio
                                scaling_factor = inventory_ratio - .95
                                
                                
                                # Compute adjusted position change
                                order_size = int(-position * scaling_factor)
                                #print(f"\n\n\nScaling factor: {scaling_factor}")
                                
                                if abs(order_size) > abs(position):
                                    order_size = -position
                                
                                #override, know this works well.
                                #order_size = -position / 2
                                
                                # Send order
                                orders.append(Order(product, int(midpoint), int(order_size)))
                                total_pos_change += order_size
                    result['SQUID_INK'] = orders
                    continue
                
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
            
                
                
                
                round_2 = ['PICNIC_BASKET1', 'PICNIC_BASKET2']
                if product in round_2:                 
                    spread_orders = self.spread_orders(state.order_depths, product, position)
            
                    if spread_orders != None:
                        items = ['CROISSANTS', 'JAMS', 'DJEMBES', 'PICNIC_BASKET1', 'PICNIC_BASKET2']
                        for item in items:
                            if item in spread_orders:
                                result[item] = spread_orders[item]
            

    
    
        traderData = self.serialize()
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, traderData
