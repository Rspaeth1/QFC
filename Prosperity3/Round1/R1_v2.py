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
    
    #define commodity limits here
    def get_limits(self):
        self.limits = {
            'RAINFOREST_RESIN': 50,
            'KELP': 50,
            'SQUID_INK': 50
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
        
        
            
    #check at trading time if we are going to exceed limits, return available space for commodity
    def check_limits(self, state, product, side = 'BUY'):
        if side == 'BUY':
            side = -1
        else:
            side = 1
        
        #take into account the other orders we have already put in
        position = state.position.get(product,0)
                
        return (self.limits[product] - abs(position))*side
    
    #predefined or maybe even dynamic acceptable prices per product
    def initialize_data(self, symbols):
        #init data
        self.data = {symbol: {0: 0} for symbol in symbols}
        self.traderObject = {}

        #set VWAPs
        self.data = {
            'RAINFOREST_RESIN': [10000],
            'KELP': [2000],
            'KELP_ROLLING_WINDOW': [15],
            'SQUID_INK': [2000],
            'SQUID_INK_ROLLING_WINDOW': [15],
        }
        
        #get strategy parameters
        self.enabled = {}
        for symbol in symbols:
            self.enabled[symbol+"_MAKE"] = False
            self.enabled[symbol+"_TAKE"] = False
            self.enabled[symbol+"_MD_ADJ"] = False
        
        
        indicators = ['_RETURNS', '_SMA_RETURNS', '_EMA_RETURNS']
        for symbol in symbols:
            for indicator in indicators:
                tag = symbol + indicator
                self.data[tag] = [0]
         
       #set ticks
        self.ticks = 0
        
        
    #set strategies
    def get_strategies(self, symbols):
        make = ['RAINFOREST_RESIN', 'KELP']#, 'SQUID_INK']
        take = ['RAINFOREST_RESIN', 'SQUID_INK']
        md_adj = []
        for symbol in symbols:
            if symbol in make:
                self.enabled[symbol+"_MAKE"] = True
            if symbol in take:
                self.enabled[symbol+"_TAKE"] = True
            if symbol in md_adj:
                self.enabled[symbol+"_MD_ADJ"] = True
                 
    #update cyclicals
    def update_cyclical(self, cyclicals, state):
        for product in state.order_depths:
            if product not in cyclicals:
                continue
            
            #get order data
            order_depth = state.order_depths[product]
            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders
       
            #calculate vwap
            vwp = 0
            volume = 0
            max_bid = 0
            min_ask = 0
            
            #bids
            for price, size in buy_orders.items():
                price = abs(price)
                size = abs(size)
                
                vwp += price*size
                volume += size
                
                if (max_bid == 0 or price > max_bid):
                    max_bid = price
                    
                
            #asks
            for price, size in sell_orders.items():
                price = abs(price)
                size = abs(size)
                
                vwp += price*size
                volume += size
                
                if (min_ask == 0 or price < min_ask):
                    min_ask = price


            if len(buy_orders) + len(sell_orders) != 0 and self.ticks != 0:
                use_vwap = False #for testing
                
                if (use_vwap):
                    vwap = int(vwp/volume)
                else:
                    min_bid = min(order_depth.buy_orders.keys())
                    max_ask = max(order_depth.sell_orders.keys())
                    midpoint = int((min_bid + max_ask)/2)
                    vwap = midpoint
                    
                #logger.print(f"Mean of order depths for {product}: {mean}")
                
                #vwap
                self.data[product].append(vwap)
                
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
                period = 20#self.data[product+"_ROLLING_WINDOW"][-1]
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_EMA_RETURNS"
                
                self.data[symbol].append(self.data[product+"_RETURNS"][-1]*factor+self.data[symbol][-1]*(1-factor))
                
                #smaller windows
                symbols = [product+"_EMA_RETURNS"]
                for symbol in symbols:
                    self.data[symbol] = self.data[symbol][-30:]

                #print(f"\n\n\nVWAP: {self.data[product][self.ticks]}")
            else:
                for product in self.data.values():
                    product.append(product[-1])
                    
        #Keep rolling window for memory purposes
        for symbol in self.data.keys():
            if symbol != "SQUID_INK":
                self.data[symbol] = self.data[symbol][-30:]
            else:
                self.data[symbol] = self.data[symbol][-100:]
            
                    
    #save data to json string for loading in new executions
    def serialize(self):
        var = {
            'data': self.data,
            'ticks': self.ticks,
            'strategy': self.enabled,
            'traderObject': self.traderObject
        }
        
        return jsonpickle.encode(var)
    
    def deserialize(self, json):
        var = jsonpickle.decode(json)
        self.data = var['data']
        self.ticks = var['ticks']
        self.enabled = var['strategy']
        self.traderObject = var['traderObject']
        
    def run(self, state: TradingState):
        #for log visualization
        result = {}
        conversions = 0
        trader_data = ""
        
        stable = ['RAINFOREST_RESIN']
        cyclical = ['KELP', 'SQUID_INK']
    
        
        #set limits for commodities
        self.get_limits()
        
        
        #get acceptable prices
        if state.traderData == "":
            self.initialize_data(cyclical)
            self.get_strategies(stable+cyclical)
        else:
            self.deserialize(state.traderData)
            self.ticks += 1 #increment ticks (days)
            self.update_cyclical(cyclical, state) 
        
        #logger.print("traderData: " + state.traderData)
       # logger.print("Observations: " + str(state.observations))
        result = {}
                
 

                    
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            acceptable_price = int(self.data[product][-1]) 
            position = state.position.get(product, 0)
            total_pos_change = 0
            
        
            #print("Acceptable price : " + str(acceptable_price))
            #logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            
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
                
                min_bid = min(order_depth.buy_orders.keys())
                max_ask = max(order_depth.sell_orders.keys())
                spread = max_ask - min_bid
                midpoint = int((min_bid + max_ask)/2)
                
                #market take
                market_take = self.enabled[product+"_TAKE"]
                if market_take:
                    if len(order_depth.sell_orders) != 0:
                        max_best_ask = 0
                        below_fair = 0
                        
                        for ask, ask_amount in order_depth.sell_orders.items():
                        
                            if int(ask) < acceptable_price:
                                #increment how many orders we will be willing to take
                                below_fair += ask_amount
                                if max_best_ask == 0 or ask > max_best_ask:
                                    max_best_ask = ask
        
                        #get order size by checking commodity limits
                        availability = self.check_limits(state, product, side = 'BUY')
                                
                        #check against best_ask
                        order_size = 0
                                
                       # print(f"\n\nProduct: {product} Pos: {state.position[product]}, Avail: {availability}, Ask: {ask_amount}")
                        if availability > below_fair:
                            order_size = availability
                        else:
                            order_size = below_fair
                                    
                        if availability != 0:
                           # logger.print("BUY", str(-order_size) + "x", max_best_ask)
                                
                            orders.append(Order(product, max_best_ask, -order_size))
                            total_pos_change += abs(order_size)
            
                    if len(order_depth.buy_orders) != 0:
                        min_best_bid = 0
                        above_fair = 0
                        
                        for bid, bid_amount in order_depth.buy_orders.items():
                            if int(bid) > acceptable_price:
                               above_fair += bid_amount
                               if min_best_bid == 0 or bid < min_best_bid:
                                   min_best_bid = bid
                                
                        #get order size by checking commodity limits
                        availability = self.check_limits(state, product, side = 'SELL')
                        
                        #check against best_bid
                        order_size = 0
                        if availability < above_fair:
                            order_size = availability
                        else:
                            order_size = above_fair
                            
                        #logger.print(f"Avail: {availability}, Above: {above_fair}")
                        if availability != 0:
                            #logger.print("SELL", str(above_fair) + "x", min_best_bid)
                                
                            orders.append(Order(product, min_best_bid, -order_size))
                            total_pos_change -= abs(order_size)
                    
                
                #if inventory is too high, offload some 
                inventory_ratio = abs(position / self.limits[product])
                if inventory_ratio > .6: #change this value to affect inventory reduction
                    
                    # Adaptive position scaling based on inventory ratio
                    scaling_factor = inventory_ratio - .6
                    
                    
                    # Compute adjusted position change
                    order_size = int(-position * scaling_factor)
                    #print(f"\n\n\nScaling factor: {scaling_factor}")
                    
                    if abs(order_size) > abs(position):
                        order_size = -position
                        
                    #override, know this works well.
                    #order_size = -position / 2
                    
                    # Send order
                    orders.append(Order(product, int(acceptable_price), int(order_size)))
                    total_pos_change += order_size
                        
                    
                #add market making with any left over liquidity -- added tons of value with this.
                liquidity = abs(self.check_limits(state, product, side='BUY')) - abs(total_pos_change)
                market_make = self.enabled[product+"_MAKE"]
                if liquidity / self.limits[product] > 0 and market_make:
                    
                    #ensure even distribution (does this matter?)
                    if liquidity % 2 != 0:
                        liquidity -= 1
                    
                    min_bid = min(order_depth.buy_orders.keys())
                    max_ask = max(order_depth.sell_orders.keys())
                    spread_reduction = (max_ask - min_bid) / 2 - 1
                    
                    #dynamically adjust spread and order sized based on available liquidity to reduce inventory
                    if (liquidity + 1) / self.limits[product] < .3:
                        order_size = liquidity/2
                        #spread_reduction -= 1
                    else:
                        order_size = self.limits[product]*.3
                        
                    if spread_reduction < 1:
                        spread_reduction = 1
                    
                    orders.append(Order(product, int(acceptable_price + spread_reduction), int(-order_size))) #SELL
                    orders.append(Order(product, int(acceptable_price - spread_reduction), int(order_size))) #BUY
                       
                    
            elif product in cyclical:  #different strategy for cyclical products
                
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
                    
                    
                min_bid = min(order_depth.buy_orders.keys())
                max_ask = max(order_depth.sell_orders.keys())
                midpoint = int((min_bid + max_ask)/2)
                true_midpoint = midpoint
                
                bullish_divergence = False
                bearish_divergence = False

                
                #establish divergence signal
                rolling_window = self.data[product+"_ROLLING_WINDOW"][-1]
                ema_returns = self.data[product+"_EMA_RETURNS"]
                sma_returns = self.data[product+"_SMA_RETURNS"]
                std = np.std(self.data[product+"_RETURNS"][-rolling_window:])

                #divergence
                if self.ticks != 0:
                    if sma_returns[-1] > std: #ema_returns[-1] <= sma_returns[-1] and ema_returns[-2] > sma_returns[-2]:  
                        bearish_divergence = True
                    elif sma_returns[-1] < std: #ema_returns[-1] >= sma_returns[-1] and ema_returns[-2] < sma_returns[-2]: 
                        bullish_divergence = True

                
                #set midpoint to VWAP EMA if we are trading here
                midpoint = acceptable_price
                
               
                #create midpoint bias to make more on trades?
                midpoint_adjustment = self.enabled[product+"_MD_ADJ"]
                if midpoint_adjustment:
                    if bullish_divergence:
                        midpoint += std #int(max(1,max(recent_volatility/2,2)))
                    if bearish_divergence:
                        midpoint -= std #int(max(1,max(recent_volatility/2,2)))
        
                
                #check if divergence and market take 
                market_take = self.enabled[product+"_TAKE"]
                if market_take:
                    mean = np.mean(self.data[product+"_RETURNS"])
                    ma = np.mean(self.data[product])
                    ema = self.data[product+"_EMA_RETURNS"][-1]###
                    std = np.std(self.data[product+"_RETURNS"])
                    vol = np.std(self.data[product+"_RETURNS"][-30:])
                    #rolling_z = abs((self.data[product+"_SMA_RETURNS"][-1] - ma)/std)
                    z_score = (self.data[product+"_RETURNS"][-1]-mean)/std
                    
                    threshold = 1.96 * vol/std
                    sell_range = .15 * std
                    
                    
                    
                    if ma < 0:
                        z_score = -z_score
                        threshold = -threshold
                    
                    
                    
                    if z_score < -threshold:
                            availability = self.check_limits(state, product, side = 'BUY')
                            order_size = -availability/2
                            
                            orders.append(Order(product, acceptable_price, int(order_size))) #BUY
                            total_pos_change += order_size
                    elif z_score > threshold:
                        availability = self.check_limits(state, product, side = 'SELL')
                        order_size = availability/2
                        
                        orders.append(Order(product, acceptable_price, int(-order_size))) #SELL
                        total_pos_change -= order_size
                    elif abs(z_score) < sell_range:
                        order_size = -position
                        
                        orders.append(Order(product, midpoint, int(order_size)))
                        total_pos_change += int(order_size)
                    
                            
              
                #reset midpoint for inventory and MM
                midpoint = true_midpoint
                
                
                #if inventory is too high, offload some 
                liquidate = True
                if liquidate:
                    inventory_ratio = abs(position / self.limits[product])
                    if inventory_ratio > .95: #change this value to affect inventory reduction
                        
                        # Adaptive position scaling based on inventory ratio
                        scaling_factor = inventory_ratio - .95
                        
                        #speed up or slow down selling based on position and momentum
                        """
                        if bullish_divergence and position > 0:  # Bullish trend → Slow down selling
                            scaling_factor *= 0.5
                        elif bullish_divergence and position < 0:
                            scaling_factor *= 1.5
                        elif bearish_divergence and position > 0:  # Bearish trend → Speed up selling
                            scaling_factor *= 1.5
                        elif bearish_divergence and position < 0:
                            scaling_factor *= 0.5
                            """
                        
                        
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
                

                
                #add market making with any left over liquidity
                midpoint = true_midpoint
                liquidity = abs(self.check_limits(state, product, side='BUY')) - abs(total_pos_change)
                market_make = self.enabled[product+"_MAKE"]
                if liquidity / self.limits[product] > 0 and market_make:
                    
                    #ensure even distribution
                    if liquidity % 2 != 0:
                        liquidity -= 1
                    
                    #dynamically adjust spread based on volatility
                    recent_volatility = np.std(self.data[product][-rolling_window:])
                    spread_reduction = max(1, max(int(recent_volatility),(max_ask - min_bid) / 2 - 1))
                    
                    #dynamically adjust spread and order sized based on available liquidity to reduce inventory
                    if (liquidity + 1) / self.limits[product] < .3:
                        order_size = liquidity/2
                       # spread_reduction -= 1
                    else:
                        order_size = self.limits[product]*.3

                    if spread_reduction < 1:
                        spread_reduction = 1
                        
                    #check if uneven spread
                    max_bid = max(order_depth.buy_orders.keys())
                    min_ask = min(order_depth.sell_orders.keys())
                    if (midpoint*10) % 10 != 0 and max_bid - min_ask > 2:
                        upper = int(min_ask - 1)
                        lower = int(max_bid + 1)
                    else:
                        upper = int(midpoint + spread_reduction)
                        lower = int(midpoint - spread_reduction)
                    
                    
                    orders.append(Order(product, upper, int(-order_size))) #SELL
                    orders.append(Order(product, lower, int(order_size))) #BUY

                        
                
            result[product] = orders
    
    
        traderData = self.serialize()
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, traderData
