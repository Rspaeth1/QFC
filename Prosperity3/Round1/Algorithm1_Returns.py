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


class Trader:
    
    #define commodity limits here
    def get_limits(self):
        self.limits = {
            'RAINFOREST_RESIN': 50,
            'KELP': 50
        }
            
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
    def get_acceptable_prices(self, symbols):
        #init data
        self.data = {symbol: {0: 0} for symbol in symbols}

        #set VWAPs
        self.data = {
            'RAINFOREST_RESIN': [10000],
            'KELP': [2021],
            'KELP_RETURNS': [0],
            'KELP_SMA_RETURNS': [0],
            'KELP_EMA_RETURNS': [0],
            'KELP_ROLLING_WINDOW': 10
        }
        
        #set prices
        self.acceptable_prices = {symbol: int(self.data[symbol][0]) for symbol in symbols}
 
       #set ticks
        self.ticks = 0
                 
        
    def update_cyclical(self, symbols, state):
        for product in state.order_depths:
            if product not in symbols:
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
                use_vwap = True #for testing
                
                if (use_vwap):
                    vwap = int(vwp/volume)
                else:
                    vwap = int((max_bid + min_ask)/2)
                    
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
                returns_array = np.array([self.data[product+"_RETURNS"]])
                self.data[symbol].append(np.average(returns_array))
                
                #EMA returns
                period = self.data[product+"_ROLLING_WINDOW"]
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_EMA_RETURNS"
                
                self.data[symbol].append(self.data[product+"_RETURNS"][-1]*factor+self.data[symbol][-1]*(1-factor))
                
                #Keep rolling window for memory purposes
                for symbol in self.data.keys():
                    if symbol == product+'_ROLLING_WINDOW':
                        continue
                    self.data[symbol] = self.data[symbol][-30:]
                
                #smaller windows
                symbols = [product+"_EMA_RETURNS"]
                for symbol in symbols:
                    self.data[symbol] = self.data[symbol][-self.data[product+"_ROLLING_WINDOW"]:]

                #print(f"\n\n\nVWAP: {self.data[product][self.ticks]}")
            else:
                for product in self.data.values():
                    product.append(product[-1])
            
            self.acceptable_prices[product] = int(self.data[product][-1])
        
    #save data to json string for loading in new executions
    def serialize(self):
        var = {
            'acceptable_prices': self.acceptable_prices,
            'data': self.data,
            'ticks': self.ticks,
        }
        
        return jsonpickle.encode(var)
    
    def deserialize(self, json):
        var = jsonpickle.decode(json)
        self.acceptable_prices = var['acceptable_prices']
        self.data = var['data']
        self.ticks = var['ticks']
        
        
    def run(self, state: TradingState):
        #for log visualization
        result = {}
        conversions = 0
        trader_data = ""
        
        stable = ['RAINFOREST_RESIN']
        cyclical = ['KELP']
        
        #set limits for commodities
        self.get_limits()
        
        
        #get acceptable prices
        if state.traderData == "":
            self.get_acceptable_prices(stable+cyclical)
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
            
            acceptable_price = int(self.acceptable_prices[product]) 
            position = state.position.get(product, 0)
            total_pos_change = 0
        
            #print("Acceptable price : " + str(acceptable_price))
            logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            
            if product in stable:
                
                min_bid = min(order_depth.buy_orders.keys())
                max_ask = max(order_depth.sell_orders.keys())
                spread = max_ask - min_bid
                
                #market take
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
                        logger.print("BUY", str(-order_size) + "x", max_best_ask)
                            
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
                        logger.print("SELL", str(above_fair) + "x", min_best_bid)
                            
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
                    order_size = -position / 2
                    
                    # Send order
                    orders.append(Order(product, int(acceptable_price), int(order_size)))
                    total_pos_change += order_size
                        
                    
                #add market making with any left over liquidity -- added tons of value with this.
                liquidity = abs(self.check_limits(state, product, side='BUY')) - abs(total_pos_change)
                if liquidity / self.limits[product] > 0:
                    
                    #ensure even distribution (does this matter?)
                    if liquidity % 2 != 0:
                        liquidity -= 1
                    
                    min_bid = min(order_depth.buy_orders.keys())
                    max_ask = max(order_depth.sell_orders.keys())
                    spread_reduction = (max_ask - min_bid) / 2 - 1 #THIS WORKS BECAUSE ITS ONE LESS THAN THE MAX. KEEP THIS IN MIND. THAT'S BENEFIT TO TRADERS THAT YOU ARE CAPTURING
                    
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
                
                min_bid = min(order_depth.buy_orders.keys())
                max_ask = max(order_depth.sell_orders.keys())
                midpoint = int((min_bid + max_ask)/2)
                true_midpoint = midpoint
                
                bullish_divergence = False
                bearish_divergence = False

                
                #establish divergence signal
                rolling_window = self.data[product+"_ROLLING_WINDOW"]
                ema_returns = self.data[product+"_EMA_RETURNS"]
                sma_returns = self.data[product+"_SMA_RETURNS"]
                std = np.std(self.data[product+"_RETURNS"][-rolling_window:])

                #std = .243 #calculated std of average rolling window of 10 returns
                if self.ticks != 0:
                    if ema_returns[-1] > std: #ema_returns[-1] <= sma_returns[-1] and ema_returns[-2] > sma_returns[-2]:  
                        bearish_divergence = True
                    elif ema_returns[-1] < -std: #ema_returns[-1] >= sma_returns[-1] and ema_returns[-2] < sma_returns[-2]: 
                        bullish_divergence = True

                
                #set midpoint to VWAP EMA if we are trading here
                midpoint = acceptable_price
                
               
                #create midpoint bias to make more on trades?
                midpoint_adjustment = True
                if midpoint_adjustment:
                    if bullish_divergence:
                        midpoint += std #int(max(1,max(recent_volatility/2,2)))
                    if bearish_divergence:
                        midpoint -= std #int(max(1,max(recent_volatility/2,2)))
        
                
                #check if divergence and market take ----I think we actually get better results from not market taking at all with our strategy...
                market_take = False #to test if market taking increases profit
                if self.ticks != 0 and market_take:
                    if len(order_depth.sell_orders) != 0:
                        max_best_ask = 0
                        below_fair = 0
                        
                        for ask, ask_amount in order_depth.sell_orders.items():
                        
                            if int(ask) < int(midpoint):
                                #increment how many orders we will be willing to take
                                below_fair += ask_amount
                                if max_best_ask == 0 or ask > max_best_ask:
                                    max_best_ask = ask
        
                        #get order size by checking commodity limits
                        availability = self.check_limits(state, product, side = 'BUY')
                                
                        #check against best_ask
                        order_size = 0
                                
                        if availability > below_fair:
                            order_size = availability
                        else:
                            order_size = below_fair
                                    
                        inventory_ratio = abs(availability) / self.limits[product]
                        if availability != 0:
                            logger.print("BUY", str(-order_size) + "x", max_best_ask)
                                
                            orders.append(Order(product, int(max_best_ask), int(-order_size)))
                            total_pos_change += abs(order_size)
            
                    if len(order_depth.buy_orders) != 0:
                        min_best_bid = 0
                        above_fair = 0
                        
                        for bid, bid_amount in order_depth.buy_orders.items():
                            if int(bid) > int(midpoint):
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
                        inventory_ratio = abs(availability) / self.limits[product]
                        if availability != 0:
                            logger.print("SELL", str(above_fair) + "x", min_best_bid)
                                
                            orders.append(Order(product, int(min_best_bid), int(-order_size)))
                            total_pos_change -= abs(order_size)
              
                #reset midpoint for inventory and MM
                midpoint = true_midpoint
                
                
                #if inventory is too high, offload some 
                
                inventory_ratio = abs(position / self.limits[product])
                if inventory_ratio > .6: #change this value to affect inventory reduction
                    
                    # Adaptive position scaling based on inventory ratio
                    scaling_factor = inventory_ratio - .6
                    
                    #speed up or slow down selling based on position and momentum
                    if bullish_divergence and position > 0:  # Bullish trend → Slow down selling
                        scaling_factor *= 0.5
                    elif bullish_divergence and position < 0:
                        scaling_factor *= 1.5
                    elif bearish_divergence and position > 0:  # Bearish trend → Speed up selling
                        scaling_factor *= 1.5
                    elif bearish_divergence and position < 0:
                        scaling_factor *= 0.5
                    
                    
                    # Compute adjusted position change
                    order_size = int(-position * scaling_factor)
                    #print(f"\n\n\nScaling factor: {scaling_factor}")
                    
                    if abs(order_size) > abs(position):
                        order_size = -position
                    
                    #override, know this works well.
                    order_size = -position / 2
                    
                    # Send order
                    orders.append(Order(product, int(midpoint), int(order_size)))
                    total_pos_change += order_size
                

                
                #add market making with any left over liquidity
                midpoint = true_midpoint
                liquidity = abs(self.check_limits(state, product, side='BUY')) - abs(total_pos_change)
                if liquidity / self.limits[product] > 0:
                    
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
                    if (midpoint*10) % 10 != 0 and max_bid - min_ask > 1:
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
