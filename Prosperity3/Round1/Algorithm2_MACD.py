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
    
    #init positions dict
    def get_positions(self, symbols):
        self.positions = {symbol: {'size': 0, 'price': 0} for symbol in symbols}
        
    #update positions
    def update_positions(self, state):
        for trades in state.own_trades.values():
            for trade in trades:
                symbol = trade.symbol
                price = trade.price
                size = trade.quantity
                if abs(self.positions[symbol]['size']) - abs(size) < 0:
                    self.positions[symbol]['size'] += size
                    self.positions[symbol]['price'] = price
                else:
                    old_size = self.positions[symbol]['size']
                    self.positions[symbol]['size'] += size
                    self.positions[symbol]['price'] = (old_size*self.positions[symbol]['price'] + size*price)/(old_size + size)
            
    
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
        #init vwaps
        self.VWAPs = {symbol: {'0': 0} for symbol in symbols}
        
        #get data from files
        """
        try:
            prices = pd.read_csv('C:\\Users\\rspaeth1\\OneDrive - University of Iowa\\Desktop\\Prosperity\\prices_round_0.csv', sep=';', header=0)
            trades = pd.read_csv('C:\\Users\\rspaeth1\\OneDrive - University of Iowa\\Desktop\\Prosperity\\trades_round_0.csv', sep=';', header=0)

            self.trade_dict = trades.to_dict(orient='index')
            
            for symbol in symbols:
                vwp = 0
                volume = 0
                for trade in self.trade_dict.values():
                    if trade['symbol'] == symbol:
                        vwp += trade['price']*trade['quantity']
                        volume += trade['quantity']
                self.VWAPs[symbol]['0'] = int(vwp/volume)
            
        except FileNotFoundError:
            self.VWAPs = {
                'RAINFOREST_RESIN': {'0': 10000},
                'KELP_26': {'0': 2018},
                'KELP_12': {'0': 2018},
                'KELP_MACD': {'0': 2018},
                'KELP_SIGNAL': {'0': 2018}
            }
            """
        #set VWAPs
        self.VWAPs = {
            'RAINFOREST_RESIN': [10000],
            'KELP': [2021],
            'KELP_26': [2021],
            'KELP_12': [2021],
            'KELP_MACD': [0],
            'KELP_SIGNAL': [0],
            'KELP_HIST': [0],
            'KELP_HIGH': [2021],
            'KELP_LOW': [2021],
            'KELP_MACD_HIGH': [0],
            'KELP_MACD_LOW': [0]
        }
        
        #set prices
        self.acceptable_prices = {symbol: int(self.VWAPs[symbol][0]) for symbol in symbols}
 
       #set ticks
        self.ticks = 0
        
    def update_stable(self, symbols, state):
        for product in state.order_depths:
            if product not in symbols:
                continue
            order_depth = state.order_depths[product]
            buy_orders = list(order_depth.buy_orders.keys())
            sell_orders = list(order_depth.sell_orders.keys())
            orders = buy_orders + sell_orders
            orders = np.array(orders)

            if len(orders) != 0:
                mean = np.mean(orders)
                #logger.print(f"Mean of order depths for {product}: {mean}")
                

                #self.acceptable_prices[product] = int((self.acceptable_prices[product] + mean)/2)
        
            
        
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
                self.VWAPs[product].append(vwap)
                
                #calculate 26 EMA
                period = 26
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_26"
                
                self.VWAPs[symbol].append(vwap*factor+self.VWAPs[symbol][-1]*(1-factor))
                
                #calculate 12 EMA
                period = 12
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_12"
                
                self.VWAPs[symbol].append(vwap*factor+self.VWAPs[symbol][-1]*(1-factor))
                
                #calculate MACD
                symbol = product+"_MACD"
                symbol_26 = product+"_26"
                symbol_12 = product+"_12"
                self.VWAPs[symbol].append(self.VWAPs[symbol_12][-1] - self.VWAPs[symbol_26][-1])

                #calculate SIGNAL
                period = 9
                smoothing = 2
                factor = smoothing/(period+1)
                symbol = product+"_MACD"
                
                self.VWAPs[product+"_SIGNAL"].append(self.VWAPs[symbol][-1]*factor+self.VWAPs[symbol][-2]*(1-factor))
                
                #calulate HIST
                symbol = product + "_HIST"
                self.VWAPs[symbol].append(self.VWAPs[product+"_MACD"][-1] - self.VWAPs[product+"_SIGNAL"][-1])
                
                #get local highs and lows
                symbol = product
                if len(self.VWAPs[symbol]) > 3:
                    last_price = self.VWAPs[symbol][-1]
                    second_last_price = self.VWAPs[symbol][-2]
                    third_last_price = self.VWAPs[symbol][-3]
                    
                    #local high
                    if second_last_price > last_price and second_last_price > third_last_price:
                        self.VWAPs[product+"_HIGH"].append(second_last_price)
                        
                    #local low
                    if second_last_price < last_price and second_last_price < third_last_price:
                        self.VWAPs[product+"_LOW"].append(second_last_price)
                        
                #get macd highs and lows
                symbol = product + "_HIST"
                if len(self.VWAPs[symbol]) > 3:
                    last_macd = self.VWAPs[symbol][-1]
                    second_last_macd = self.VWAPs[symbol][-2]
                    third_last_macd = self.VWAPs[symbol][-3]
                    
                    #local high
                    if second_last_macd > last_macd and second_last_macd > third_last_macd:
                        self.VWAPs[product+"_MACD_HIGH"].append(second_last_macd)
                        
                    #local low
                    if second_last_macd < last_macd and second_last_macd < third_last_macd:
                        self.VWAPs[product+"_MACD_LOW"].append(second_last_macd)
                
                #Keep rolling window for memory purposes
                symbols = [product, product+"_26", product+"_12", product+"_MACD", product+"_SIGNAL", product+"_HIST", product+"_HIGH", product+"_LOW", product+"_MACD_HIGH", product+"_MACD_LOW"]
                for symbol in symbols:
                    self.VWAPs[symbol] = self.VWAPs[symbol][-26:]

                #print(f"\n\n\nVWAP: {self.VWAPs[product][self.ticks]}")
            else:
                symbols = [product, product+"_26", product+"_12", product+"_MACD", product+"_SIGNAL", product+"_HIST", product+"_HIGH", product+"_LOW", product+"_MACD_HIGH", product+"_MACD_LOW"]
                for symbol in symbols:
                    self.VWAPs[symbol].append(self.VWAPs[symbol][self.ticks-1])
            
            self.acceptable_prices[product] = int(self.VWAPs[product][-1])
        
    #save data to json string for loading in new executions
    def serialize(self):
        var = {
            'acceptable_prices': self.acceptable_prices,
            'VWAPs': self.VWAPs,
            'ticks': self.ticks,
            'positions': self.positions
        }
        
        return jsonpickle.encode(var)
    
    def deserialize(self, json):
        var = jsonpickle.decode(json)
        self.acceptable_prices = var['acceptable_prices']
        self.VWAPs = var['VWAPs']
        self.ticks = var['ticks']
        self.positions = var['positions']
        
        
    def run(self, state: TradingState):
        #for log visualization
        result = {}
        conversions = 0
        trader_data = ""
        
        stable = ['RAINFOREST_RESIN']
        cyclical = ['KELP']
        
        #set limits for commodities
        self.get_limits()
        
        #set position dicts
        self.get_positions(stable+cyclical)
        
        #get acceptable prices
        if state.traderData == "":
            self.get_acceptable_prices(stable+cyclical)
        else:
            self.deserialize(state.traderData)
            self.ticks += 1 #increment ticks (days)
            #self.update_stable(stable, state) #update using SMA
            self.update_cyclical(cyclical, state) #update using EMA
            self.update_positions(state)
        
        #logger.print("traderData: " + state.traderData)
       # logger.print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            acceptable_price = self.acceptable_prices[product]  # Participant should calculate this value
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
                
                
                #if inventory is too high, offload some -- this might not be optimal
                inventory_ratio = abs(position / self.limits[product])
                if inventory_ratio > .6:
                    availability = abs(self.check_limits(state, product, side = 'BUY'))
                    availability -= total_pos_change
                    threshold = .5
                    adjustment = int(-position*threshold)
                    order_size = adjustment
                    if abs(order_size) > availability:
                        if position > 0:
                            sign = -1
                        else:
                            sign = 1
                        order_size = availability * sign
                        
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
                        spread_reduction -= 1
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
                
                """
                #check if there are orders at the midpoint and take them, then list them at 1 higher or lower?
                spread = max_ask - min_bid
                adjustment = max(1, spread // 3)
                if len(order_depth.sell_orders) != 0:
                    if midpoint in order_depth.sell_orders.keys():
                        orders.append(Order(product, int(midpoint), int(-order_depth.sell_orders[midpoint])))
                        orders.append(Order(product, midpoint + adjustment, order_depth.sell_orders[midpoint]))
                        total_pos_change += abs(order_depth.sell_orders[midpoint])
                
                if len(order_depth.buy_orders) != 0:
                    if midpoint in order_depth.buy_orders.keys():
                        orders.append(Order(product, int(midpoint), int(-order_depth.buy_orders[midpoint])))
                        orders.append(Order(product, midpoint - adjustment, order_depth.buy_orders[midpoint]))
                        total_pos_change -= abs(order_depth.buy_orders[midpoint])
                        """
                
                #establish divergence signal
                highs = self.VWAPs[product+"_HIGH"]
                macd_highs = self.VWAPs[product+"_MACD_HIGH"]
                
                if len(highs) > 3 and len(macd_highs) > 3:
                    if highs[-1] > highs[-2] > highs[-3] and macd_highs[-1] < macd_highs[-2] < macd_highs[-3]:
                        bearish_divergence = True
                    else:
                        bearish_divergence = False
                
                lows = self.VWAPs[product+"_LOW"]
                macd_lows = self.VWAPs[product+"_MACD_LOW"]
                
                if len(lows) > 3 and len(macd_lows) > 3:
                    if lows[-1] < lows[-2] < lows[-3] and macd_lows[-1] > macd_lows[-2] > macd_lows[-3]:
                        bullish_divergence = True
                    else:
                        bullish_divergence = False

                
                #try to incorporate MACD
                recent_volatility = np.std(self.VWAPs[product][-10:])
                """
                if bullish_divergence:
                    midpoint += max(1,recent_volatility/2)
                
                if bearish_divergence:
                    midpoint -= max(1,recent_volatility/2)
                """
                
                #check if divergence and market take
                if len(order_depth.sell_orders) != 0 and bullish_divergence:
                    max_best_ask = 0
                    below_fair = 0
                    
                    for ask, ask_amount in order_depth.sell_orders.items():
                    
                        if int(ask) < midpoint:
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
                            
                        orders.append(Order(product, int(max_best_ask), int(-order_size)))
                        total_pos_change += abs(order_size)
        
                if len(order_depth.buy_orders) != 0 and bearish_divergence:
                    min_best_bid = 0
                    above_fair = 0
                    
                    for bid, bid_amount in order_depth.buy_orders.items():
                        if int(bid) > midpoint:
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
                            
                        orders.append(Order(product, int(min_best_bid), int(-order_size)))
                        total_pos_change -= abs(order_size)
                
                        
                #if inventory is too high, offload some -- may be overfit, probably want dynamic based on average inventory accumulation or something
                
                inventory_ratio = abs(position / self.limits[product])
                if inventory_ratio > .4:
                    
                    """
                    availability = abs(self.check_limits(state, product, side = 'BUY'))
                    availability -= total_pos_change
                    threshold = inventory_ratio - .4
                    if inventory_ratio > .7:
                        threshold = .5
                    adjustment = int(-position*threshold)
                    order_size = max(min(adjustment,adjustment-total_pos_change),0) #not sure why this works, it was a mistake that worked better than intended with broken logic. I think this makes it so we aren't closing shorts? Could be bad.
                    if abs(order_size) > availability:
                        if position > 0:
                            sign = -1
                        else:
                            sign = 1
                        order_size = availability * sign
                        
                    orders.append(Order(product, int(midpoint), int(order_size)))
                    total_pos_change += order_size
                    """
                    # Adaptive position scaling based on inventory ratio
                    scaling_factor = 1 - min(0.5,inventory_ratio)
                    
                    # Adjust based on MACD momentum
                    #momentum = self.VWAPs[product+"_MACD"][-1] - self.VWAPs[product+"_SIGNAL"][-1]
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
                    
                    # Ensure we are within available trade limits
                    if abs(order_size) > abs(position):
                        order_size = -position
                    
                    # Send order
                    orders.append(Order(product, int(midpoint), int(order_size)))
                    total_pos_change += order_size
                
                
                #add market making with any left over liquidity
                liquidity = abs(self.check_limits(state, product, side='BUY')) - abs(total_pos_change)
                if liquidity / self.limits[product] > 0:
                    
                    #ensure even distribution
                    if liquidity % 2 != 0:
                        liquidity -= 1
                    
                    #dynamically adjust spread based on volatility
                    spread_reduction = min(max(1, int(recent_volatility / 2)),(max_ask - min_bid) / 2 - 1)
                    
                    #dynamically adjust spread and order sized based on available liquidity to reduce inventory
                    if (liquidity + 1) / self.limits[product] < .3:
                        order_size = liquidity/2
                        spread_reduction -= 1
                    else:
                        order_size = self.limits[product]*.3

                    if spread_reduction < 1:
                        spread_reduction = 1
                    
                    orders.append(Order(product, int(midpoint + spread_reduction), int(-order_size))) #SELL
                    orders.append(Order(product, int(midpoint - spread_reduction), int(order_size))) #BUY
                        
            result[product] = orders
    
    
        traderData = self.serialize()
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, traderData
