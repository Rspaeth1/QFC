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
            'KELP': 50,
            'SQUID_INK': 50
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
    
    #initialize data
    def initialize_data(self, symbols):
        
        #init data
        self.data = {
            'RAINFOREST_RESIN': [10000],
            'KELP': [2000],
            'SQUID_INK': [2000],
            'KELP_ROLLING_WINDOW': [15],
            'SQUID_INK_ROLLING_WINDOW': [15],
        }
        
        #get strategy parameters
        self.enabled = {}
        for symbol in symbols:
            self.enabled[symbol+"_MAKE"] = False
            self.enabled[symbol+"_TAKE"] = False
            self.enabled[symbol+"_MD_ADJ"] = False
        
        indicators = ['_30', '_15', '_MACD', '_SIGNAL', '_HIST', '_HIGH', '_LOW', '_MACD_HIGH', '_MACD_LOW', '_RSI', '_RSI_GAINS', '_RSI_LOSSES', '_SMA', '_EMA', '_ROC', '_OBV', '_TSI', '_TSI_EMA', '_TSI_EMA_ABS', '_TSI_SIGNAL', '_HMA', '_RETURNS', '_SMA_RETURNS', '_EMA_RETURNS', '_EMA_RETURNS_LONG', '_EMA_RETURNS_LONG_ABS']
        for symbol in symbols:
            for indicator in indicators:
                tag = symbol + indicator
                self.data[tag] = [0]
    
        
        #set weights
        self.weights = {
            'KELP': [0],
            'SQUID_INK': [0]
        }
        indicators = ['_SMA', '_EMA', '_ROC', '_OBV', '_RSI', '_MACD', '_TSI', '_HMA']
        for symbol in symbols:
            for indicator in indicators:
                tag = symbol + indicator
                self.weights[tag] = 1/8
 
       #set ticks
        self.ticks = 0
            
    #set strategies
    def get_strategies(self, symbols):
        make = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']
        take = ['RAINFOREST_RESIN', 'SQUID_INK']
        md_adj = []
        for symbol in symbols:
            if symbol in make:
                self.enabled[symbol+"_MAKE"] = True
            if symbol in take:
                self.enabled[symbol+"_TAKE"] = True
            if symbol in md_adj:
                self.enabled[symbol+"+MD_ADJ"] = True
        
        
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
                use_vwap = True #for testing
                
                if (use_vwap):
                    vwap = int(vwp/volume)
                else:
                    vwap = int((max_bid + min_ask)/2)
                    
                #logger.print(f"Mean of order depths for {product}: {mean}")
                
                #vwap
                self.data[product].append(vwap)
                
                #calculate 30 EMA
                period = 30
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_30"
                
                self.data[symbol].append(vwap*factor+self.data[symbol][-1]*(1-factor))
                
                #MACD
                #calculate 15 EMA
                period = 15
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_15"
                
                self.data[symbol].append(vwap*factor+self.data[symbol][-1]*(1-factor))
                
                #calculate MACD
                symbol = product+"_MACD"
                symbol_26 = product+"_30"
                symbol_12 = product+"_15"
                self.data[symbol].append(self.data[symbol_12][-1] - self.data[symbol_26][-1])

                #calculate SIGNAL
                period = 12
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_MACD"
                
                self.data[product+"_SIGNAL"].append(self.data[symbol][-1]*factor+self.data[symbol][-2]*(1-factor))
                
                #calulate HIST
                symbol = product + "_HIST"
                self.data[symbol].append(self.data[product+"_MACD"][-1] - self.data[product+"_SIGNAL"][-1])
                
                #get local highs and lows
                symbol = product
                if len(self.data[symbol]) > 3:
                    last_price = self.data[symbol][-1]
                    second_last_price = self.data[symbol][-2]
                    third_last_price = self.data[symbol][-3]
                    
                    #local high
                    if second_last_price > last_price and second_last_price > third_last_price:
                        self.data[product+"_HIGH"].append(second_last_price)
                        
                    #local low
                    if second_last_price < last_price and second_last_price < third_last_price:
                        self.data[product+"_LOW"].append(second_last_price)
                        
                #get macd highs and lows
                symbol = product + "_HIST"
                if len(self.data[symbol]) > 3:
                    last_macd = self.data[symbol][-1]
                    second_last_macd = self.data[symbol][-2]
                    third_last_macd = self.data[symbol][-3]
                    
                    #local high
                    if second_last_macd > last_macd and second_last_macd > third_last_macd:
                        self.data[product+"_MACD_HIGH"].append(second_last_macd)
                        
                    #local low
                    if second_last_macd < last_macd and second_last_macd < third_last_macd:
                        self.data[product+"_MACD_LOW"].append(second_last_macd)
    
                
                #RETURNS
                symbol = product+"_RETURNS"
                recent_return = self.data[product][-1] - self.data[product][-2]
                #if ret != 0:
                self.data[symbol].append(recent_return)
                
                #SMA of returns
                symbol = product+"_SMA_RETURNS"
                returns_array = np.array([self.data[product+"_RETURNS"]])
                self.data[symbol].append(np.average(returns_array))
                
                #EMA returns
                period = self.data[product+"_ROLLING_WINDOW"][-1]
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_EMA_RETURNS"
                
                self.data[symbol].append(self.data[product+"_RETURNS"][-1]*factor+self.data[symbol][-1]*(1-factor))
                
                period = 30
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_EMA_RETURNS_LONG"
                
                self.data[symbol].append(self.data[product+"_RETURNS"][-1]*factor+self.data[symbol][-1]*(1-factor))
                
                period = 30
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_EMA_RETURNS_LONG_ABS"
                
                self.data[symbol].append(abs(self.data[product+"_RETURNS"][-1])*factor+self.data[symbol][-1]*(1-factor))
                
                
                #RSI
                gain = 0
                loss = 0
                period = 15
                
                if recent_return < 0:
                    loss = recent_return
                elif recent_return > 0:
                    gain = recent_return
                
                symbol = product + "_RSI_GAINS"
                self.data[symbol].append((self.data[symbol][-1]*(period-1) + gain)/period)
                avg_gain = np.mean(np.array(self.data[symbol][-period:]))
                
                symbol = product + "_RSI_LOSSES"
                self.data[symbol].append((self.data[symbol][-1]*(period-1) + loss)/period)
                avg_loss = np.mean(np.array(self.data[symbol][-period:]))

                
                symbol = product + "_RSI"
                RS = avg_gain/avg_loss
                RSI = 100 - (100/(1+RS))
                self.data[symbol].append(RSI)
                
                
                #SMA
                period = 15
                symbol = product + "_SMA"
                SMA = np.mean(np.array(self.data[product][-period:]))
                self.data[symbol].append(SMA)
                
                
                #EMA
                period = 15
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product + "_EMA"
                
                self.data[symbol].append(vwap*factor+self.data[symbol][-1]*(1-factor))
                
                
                #ROC
                period = 15
                symbol = product + "_ROC"
                if len(self.data[product]) > period:
                    last_period_price = self.data[product][-period]
                    ROC = (vwap-last_period_price)/last_period_price
                    self.data[symbol].append(ROC)
                
                
                #OBV
                symbol = product + "_OBV"
                last_OBV = self.data[symbol][-1]
                OBV = last_OBV
                if recent_return < 0:
                    OBV = last_OBV - volume
                elif recent_return > 0:
                    OBV = last_OBV + volume
                self.data[symbol].append(OBV)
                
                
                #TSI
                period = 15
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_TSI_EMA"
                
                self.data[symbol].append(self.data[product+"_EMA_RETURNS_LONG"][-1]*factor+self.data[symbol][-1]*(1-factor))
                TSI_PC = self.data[symbol][-1]
                
                period = 15
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_TSI_EMA_ABS"
                
                self.data[symbol].append(self.data[product+"_EMA_RETURNS_LONG_ABS"][-1]*factor+self.data[symbol][-1]*(1-factor))
                TSI_PC_ABS = self.data[symbol][-1]
                
                symbol = product + "_TSI"
                self.data[symbol].append(100*TSI_PC/TSI_PC_ABS)
                
                symbol = product + "_TSI_SIGNAL"
                period = 12
                smoothing = 2
                factor = smoothing/(period+1)
                
                self.data[symbol].append(self.data[product+"_TSI"][-1]*factor+self.data[symbol][-1]*(1-factor))

                
                #HMA
                period = 15
                if len(self.data[product]) > period:
                    WMA = 0
                    for i in range(period, 0, -1):
                        WMA += self.data[product][-i]*i
                    WMA /= period*(period+1)/2 #sum of i
                    
                    WMA_HALF = 0
                    for i in range(int(period/2), 0, -1):
                        WMA_HALF += self.data[product][-i]*i
                    WMA_HALF /= int(period/2)*(int(period/2)+1)/2 #sum of i
                    
                    RAW = WMA - WMA_HALF
                    
                    period = int(period**(.5))
                    symbol = product + "_HMA"
                    WMA = 0
                    for i in range(period, 0, -1):
                        WMA += RAW*i
                    WMA /= period*(period+1)/2 #sum of i
                    self.data[symbol].append(WMA)
                
                #Keep rolling window for memory purposes
                for symbol in self.data.keys():
                    self.data[symbol] = self.data[symbol][-30:]
                
                #smaller windows
                symbols = [product+"_EMA_RETURNS"]
                for symbol in symbols:
                    self.data[symbol] = self.data[symbol][-self.data[product+"_ROLLING_WINDOW"][-1]:]

                #print(f"\n\n\nVWAP: {self.data[product][self.ticks]}")
            
                #calculate weight
                indicators = ['SMA', 'EMA', 'ROC', 'OBV', 'RSI', 'MACD', 'TSI', 'HMA']
                total_weight = 0
                
                for symbol in indicators:
                    try:
                        current_indicator = self.data[product+"_"+symbol][-1]
                        last_indicator = self.data[product+"_"+symbol][-2]
                        current_price = self.data[product][-1]
                        last_price = self.data[product][-2]
                        trend = np.mean(np.diff(np.array(self.data[product+"_"+symbol])))
                        last_trend = np.mean(np.diff(np.array(self.data[product+"_"+symbol][:-1])))
                        price_trend = np.mean(np.diff(np.array(self.data[product])))
                        
                        score = 0
                        
                        #standardize value into signal
                        if symbol == "SMA" or symbol == "EMA":
                            if last_price >= last_indicator and current_price < current_indicator:
                                score = -1
                            elif trend < 0:
                                score = -.5
                            elif last_price <= last_indicator and current_price > current_indicator:
                                score = 1
                            elif trend > 0:
                                score +.5
                        elif symbol == 'HMA':
                            if last_trend >= 0 and trend < 0:
                                score = -1
                            elif trend < 0:
                                score = -.5
                            elif last_trend <= 0 and trend > 0:
                                score = 1
                            elif trend > 0:
                                score = .5
                        elif symbol == "ROC":
                            if last_indicator >= 0 and current_indicator < 0:
                                score = -1
                            elif trend < 0 and price_trend > 0:
                                score = -.5
                            elif last_indicator <= 0 and current_indicator > 0:
                                score = 1
                            elif trend > 0 and price_trend < 0:
                                score = -.5
                        elif symbol == 'RSI':
                            if last_indicator >= 70 and current_indicator < 70:
                                score = -1
                            elif trend < 0 and current_indicator > 30 and current_indicator < 70:
                                score = -.5
                            elif last_indicator <= 30 and current_indicator > 30:
                                score = 1
                            elif trend > 0 and current_indicator > 30 and current_indicator < 70:
                                score = .5
                        elif symbol == 'OBV':
                            if current_indicator < last_indicator and price_trend < 0:
                                score = -1
                            elif current_indicator < last_indicator and price_trend > 0:
                                score = -.5
                            elif current_indicator > last_indicator and price_trend > 0:
                                score = 1
                            elif current_indicator > last_indicator and price_trend < 0:
                                score = .5    
                        elif symbol == 'MACD':
                            current_indicator = self.data[product+"_HIST"][-1]
                            last_indicator = self.data[product+"_HIST"][-2]
                            trend = np.mean(np.diff(np.array(self.data[product+"_HIST"])))
                            last_trend = np.mean(np.diff(np.array(self.data[product+"_HIST"][:-2])))
                            if last_indicator >= 0 and current_indicator < 0:
                                score = -1
                            elif trend < 0:
                                score = -.5
                            elif last_indicator <= 0 and current_indicator > 0:
                                score = 1
                            elif trend > 0:
                                score = .5
                        elif symbol == 'TSI':
                            current_signal = self.data[product+"_TSI_SIGNAL"][-1]
                            last_signal = self.data[product+"_TSI_SIGNAL"][-2]
                            if last_indicator >= last_signal and current_indicator < current_signal and current_indicator >= 25:
                                score = -1
                            elif trend < 0:
                                score = -.5
                            elif last_indicator <= last_signal and current_indicator > current_signal and current_indicator < 25:
                                score = 1
                            elif trend > 0:
                                score = .5
                            
                        weight = self.weights[product+"_"+symbol]
                        total_weight += score*weight
                    except:
                        pass
                self.weights[product].append(total_weight) 
            else:
                for product in self.data.values():
                    product.append(product[-1])
                    
            
        
    #save data to json string for loading in new executions
    def serialize(self):
        var = {
            'data': self.data,
            'ticks': self.ticks,
            'strategies': self.enabled,
        }
        
        return jsonpickle.encode(var)
    
    def deserialize(self, json):
        var = jsonpickle.decode(json)
        self.data = var['data']
        self.ticks = var['ticks']
        self.enabled = var['strategies']

        
    def run(self, state: TradingState):
        #for log visualization
        result = {}
        conversions = 0
        trader_data = ""
        
        stable = ['RAINFOREST_RESIN']
        cyclical = ['KELP', 'SQUID_INK']
        
        #set limits for commodities
        self.get_limits()
        
        
        #update and init data
        if state.traderData == "":
            self.initialize_data(cyclical)
            self.get_strategies(cyclical+stable)
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
                
                #set weight
                weight = self.weights[product][-1]
                if len(self.weights[product]) > 1:
                    last_weight = self.weights[product][-2]
                else:
                    last_weight = 0
                weight_trend = np.mean(np.diff(np.array(self.weights[product][-15:])))
                
                
                #adjust midpoint
                midpoint_adjustment =  self.enabled[product+"_MD_ADJ"]
                if midpoint_adjustment:
                    if weight > .5:
                        midpoint += 1
                    elif weight < -.5:
                        midpoint -= 1
                    
                    
                
                
                #check if divergence and market take 
                market_take = self.enabled[product+"_TAKE"]
                if self.ticks != 0 and market_take:
                    if len(order_depth.sell_orders) != 0:

                        
                        
        
                        #get order size by checking commodity limits
                        availability = self.check_limits(state, product, side = 'BUY')
                          
                        order_size = 0
                        
                        if weight > .75:
                            order_size = 50
                        elif weight > .5:
                            order_size = 25
                        elif weight > .25:
                            order_size = 10
                        elif weight > .1:
                            order_size = 5
                            
                        
                                
                        if availability > order_size:
                            order_size = availability
                                    
                        inventory_ratio = abs(availability) / self.limits[product]
                        if availability != 0:
                            logger.print("BUY", str(order_size) + "x", midpoint)
                                
                            orders.append(Order(product, int(midpoint), int(order_size)))
                            total_pos_change += abs(order_size)
            
                    if len(order_depth.buy_orders) != 0:

                        

                                
                        #get order size by checking commodity limits
                        availability = self.check_limits(state, product, side = 'SELL')
                        
                        order_size = 0
                        
                        if weight < -.75:
                            order_size = -50
                        elif weight < -.5:
                            order_size = -25
                        elif weight < -.25:
                            order_size = -10
                        elif weight < -.1:
                            order_size = -5
                                
                        if availability < order_size:
                            order_size = availability
                            
                        #logger.print(f"Avail: {availability}, Above: {above_fair}")
                        inventory_ratio = abs(availability) / self.limits[product]
                        if availability != 0:
                            logger.print("SELL", str(order_size) + "x", midpoint)
                                
                            orders.append(Order(product, int(midpoint), int(order_size)))
                            total_pos_change -= abs(order_size)
              
                #reset midpoint for inventory and MM
                midpoint = true_midpoint
                
                
                #if inventory is too high, offload some 
                inventory_management = True
                if inventory_management:
                    inventory_ratio = abs(position / self.limits[product])
                    cross_down = last_weight >= 0 and weight < 0 and weight_trend < 0
                    cross_up = last_weight <= 0 and weight > 0 and weight_trend > 0
                    unsold = weight < 0 and weight_trend < 0 and position > 0
                    unbought = weight > 0 and weight_trend > 0 and position < 0
                    
                    if cross_up or cross_down: #change this value to affect inventory reduction
                        
                        # Adaptive position scaling based on inventory ratio
                        scaling_factor = inventory_ratio - .6
                        
                        
                        # Compute adjusted position change
                        order_size = int(-position * scaling_factor)
                        #print(f"\n\n\nScaling factor: {scaling_factor}")
                        
                        if abs(order_size) > abs(position):
                            order_size = -position
                        
                        #override, know this works well.
                        order_size = -position
                        
                        if order_size > 0:
                            midpoint -= 1
                        elif order_size < 0:
                            midpoint += 1
                            
                        # Send order
                        orders.append(Order(product, int(midpoint), int(order_size)))
                        total_pos_change += order_size
                    if unsold or unbought and not (cross_up or cross_down):
                        orders.append(Order(product, int(midpoint), int(-position/2)))
                        total_pos_change += -position/2
                

             
                
                #add market making with any left over liquidity
                market_making = self.enabled[product+"_MAKE"]
                midpoint = true_midpoint
                if market_making:
                    liquidity = abs(self.check_limits(state, product, side='BUY')) - abs(total_pos_change)
                    if liquidity / self.limits[product] > 0:
                        
                        #ensure even distribution
                        if liquidity % 2 != 0:
                            liquidity -= 1
                        
                        #dynamically adjust spread based on volatility
                        recent_volatility = np.std(self.data[product][-self.data[product+"_ROLLING_WINDOW"][-1]:])
                        spread_reduction = max(1, max(int(recent_volatility),(max_ask - min_bid) / 2 - 1))
                        
                        #dynamically adjust spread and order sized based on available liquidity to reduce inventory
                        if (liquidity + 1) / self.limits[product] < .3:
                            order_size = liquidity/2
                            #spread_reduction -= 1
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
