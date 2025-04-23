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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # TODO: Add logic

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data


class Trader:
    
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
            'VOLCANIC_ROCK_VOUCHER_10500': 200,
            'MAGNIFICENT_MACARONS': 75,
            'MAGNIFICENT_MACARONS_CONVERSION': 10
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
    
        
    #save data to json string for loading in new executions
    def serialize(self):
        var = {
            'data': self.data,
            'variables': self.variables
        }
        
        return jsonpickle.encode(var)
    
    def deserialize(self, json):
        var = jsonpickle.decode(json)
        self.data = var['data']
        self.variables = var['variables']
        
    ################################################################    
        
    def get_implied_bid_ask(self, observation):
        storage_cost = .1
        return observation.bidPrice - observation.exportTariff - observation.transportFees - storage_cost, observation.askPrice - observation.importTariff - observation.transportFees
        
    def get_vwmp(self, product_depth):
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
    
    def update_ema(self, prev_ema, new_value, alpha):
        if prev_ema is None or not np.isfinite(prev_ema):
               return new_value
        if new_value is None or not np.isfinite(new_value):
               return prev_ema
        return (1 - alpha) * prev_ema + alpha * new_value
    
    def get_fair_price_macaron(self, state, observation):
        window = 100
        alpha = 2/(window+1)
        
        
        
        sunlightIndex = observation.sunlightIndex
        transportFees = observation.transportFees
        exportTariff = observation.exportTariff
        importTariff = observation.importTariff
        
        # Sunlight Index
        
        CSI = 45
        self.variables.setdefault('last_sunlightIndex', sunlightIndex)
        last_sunlightIndex = self.variables['last_sunlightIndex']
        sunlight_mean = self.update_ema(last_sunlightIndex, sunlightIndex, alpha)
        self.variables['last_sunlightIndex'] = sunlight_mean
        sunlight_score = (CSI - sunlight_mean) / CSI
        
        self.variables.setdefault('last_sunlight_effect', sunlight_score)
        last_sunlight_effect = self.variables['last_sunlight_effect']
        sunlight_effect = self.update_ema(last_sunlight_effect, sunlight_score, alpha)
        self.variables['last_sunlight_effect'] = sunlight_effect
        
        
        # Tarrif Impact
        self.variables.setdefault('max_export', 11.5)
        self.variables.setdefault('max_import', 6)
        baseline_export = 9
        baseline_import = -3
        
        if abs(exportTariff) > self.variables['max_export']:
            self.variables['max_export'] = exportTariff
        if abs(importTariff) > self.variables['max_import']:
            self.variables['max_import'] = importTariff
            
        export_score = (exportTariff - baseline_export) / self.variables['max_export']
        import_score = (importTariff - baseline_import) / self.variables['max_import']
        
        self.variables.setdefault('last_export', export_score)
        last_export = self.variables['last_export']
        export_effect = self.update_ema(last_export, export_score, alpha)
        self.variables['last_export'] = export_effect
        
        self.variables.setdefault('last_import', import_score)
        last_import = self.variables['last_import']
        import_effect = self.update_ema(last_import, import_score, alpha)
        self.variables['last_import'] = import_effect
        
        
        # Volume spread / imbalance
        self.variables.setdefault('max_volume_spread', 10)
        self.variables.setdefault('max_volume_imbalance', .5224)
        baseline_spread = 8
        baseline_imbalance = .069
        
        product = 'MAGNIFICENT_MACARONS'
        depth = state.order_depths[product]
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bid_volume = sum(abs(depth.buy_orders[bid]) for bid in depth.buy_orders.keys())
        ask_volume = sum(abs(depth.sell_orders[ask]) for ask in depth.sell_orders.keys())
        total_volume = bid_volume + ask_volume
        volume_spread = best_ask - best_bid
        volume_imbalance = (bid_volume - ask_volume) / total_volume if total_volume != 0 else 0
        
        if abs(volume_spread) > self.variables['max_volume_spread']:
            self.variables['max_volume_spread'] = volume_spread
            
        if abs(volume_imbalance) > self.variables['max_volume_imbalance']:
            self.variables['max_volume_imbalance'] = volume_imbalance
            
        volume_spread_score = (volume_spread - baseline_spread) / self.variables['max_volume_spread']
        volume_imbalance_score = (volume_imbalance - baseline_imbalance) / self.variables['max_volume_imbalance']
        
        self.variables.setdefault('last_volume_spread', volume_spread_score)
        last_volume_spread = self.variables['last_volume_spread']
        volume_spread_effect = self.update_ema(last_volume_spread, volume_spread_score, alpha)
        self.variables['last_volume_spread'] = volume_spread_effect
        
        self.variables.setdefault('last_volume_imbalance', volume_imbalance_score)
        last_volume_imbalance = self.variables['last_volume_imbalance']
        volume_imbalance_effect = self.update_ema(last_volume_imbalance, volume_imbalance_score, alpha)
        self.variables['last_volume_imbalance'] = volume_imbalance_effect
        
        
        # Transport Effect
        self.variables.setdefault('max_transport', 2.1)
        baseline_transport = 1
        
        if abs(transportFees) > self.variables['max_transport']:
            self.variables['max_transport'] = transportFees
            
        transport_score = (transportFees - baseline_transport) / self.variables['max_transport']
        
        self.variables.setdefault('last_transport', transport_score)
        last_transport = self.variables['last_transport']
        transport_effect = self.update_ema(last_transport, transport_score, alpha)
        self.variables['last_transport'] = transport_effect
        
        # Rolling Returns Effect
        bidPrice = observation.bidPrice
        askPrice = observation.askPrice
        midPrice = (bidPrice + askPrice) / 2
        self.variables.setdefault('last_price_macaron', midPrice)
        last_price = self.variables['last_price_macaron']
        current_return = midPrice - last_price
        self.variables['last_price_macaron'] = midPrice
        
        self.variables.setdefault('last_return_macaron', current_return)
        last_return = self.variables['last_return_macaron']
        return_effect = self.update_ema(last_return, current_return, alpha)
        self.variables['last_return_macaron'] = return_effect
        #logger.print(f'vwmp: {vwmp}, return_effect: {return_effect}, last_return: {last_return}, last_price: {last_price}')
        
        
        # Regression
        X1 = 176.7514 * sunlight_effect
        X2 = -55.98311 * export_effect
        X3 = -14.5874 * import_effect
        X4 = 59.6211 * transport_effect
        X5 = 6.5902 * volume_spread_effect
        X6 = -147.6076 * volume_imbalance_effect
        X7 = 50.7946 * return_effect
        
        composite = X1 + X2 + X3 + X4 + X5 + X6 + X7
        
        self.variables.setdefault('last_composite', composite)
        last_composite = self.variables['last_composite']
        current_composite = self.update_ema(last_composite, composite, alpha)
        self.variables['last_composite'] = current_composite
        #logger.print(f"Sunlight: {X1:.2f}, Export: {X2:.2f}, Import: {X3:.2f}, Transport: {X4:.2f}, Spread: {X5:.2f}, Imb: {X6:.2f}, Return: {X7:.2f}, Composite: {current_composite:.2f}")
        logger.print(f"Sunlight: {sunlight_effect}, Export: {export_effect}, import: {import_effect}, transport: {transport_effect}, vol_spread: {volume_spread_effect}, vol_imbal: {volume_imbalance_effect}, ret: {return_effect}")
  
        return current_composite
    
    def get_indicators(self, depth, fair, midpoint):
        spread_mean = .02672
        spread_std = 55.4842
        panic_mean = -.1343
        panic_std = .2261
        volume_spread_mean = 8.0053
        volume_spread_std = 1.2085
        
        window = 100
        alpha = 2/(window+1)
        
        self.variables.setdefault('last_spread', fair - midpoint)
        last_spread = self.variables['last_spread']
        spread = fair - midpoint
        spread_ema = self.update_ema(last_spread, spread, alpha)
        self.variables['last_spread'] = spread_ema
        
        self.variables.setdefault('last_volume_z_spread', fair - midpoint)
        last_volume_spread = self.variables['last_volume_z_spread']
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        volume_spread = best_ask - best_bid
        volume_spread_ema = self.update_ema(last_volume_spread, volume_spread, alpha)
        self.variables['last_volume_z_spread'] = volume_spread_ema
        
        panic_z = (self.variables.get('last_sunlight_effect', panic_mean) - panic_mean) / panic_std
        spread_z = (self.variables.get('last_spread', spread_mean) - spread_mean) / spread_std
        volume_spread_z = (self.variables.get('last_volume_z_spread', volume_spread_mean) - volume_spread_mean) / volume_spread_std
        
        self.variables.setdefault('last_panic', panic_z)
        last_panic = self.variables['last_panic']
        panic_z = self.update_ema(last_panic, panic_z, alpha)
        self.variables['last_panic'] = panic_z
        
        self.variables.setdefault('last_spread_ema', spread_z)
        last_spread_ema = self.variables['last_spread_ema']
        spread_z = self.update_ema(last_spread_ema, spread_z, alpha)
        self.variables['last_spread_ema'] = spread_z
        
        return panic_z, spread_z, volume_spread_z
        
        
    def get_macaron_orders(self, state):
        
        product = 'MAGNIFICENT_MACARONS'
        orders = {product: []}
        if product in state.observations.conversionObservations:
            observation = state.observations.conversionObservations[product]
            implied_bid, implied_ask = self.get_implied_bid_ask(observation)
            
             
            depth = state.order_depths[product]
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            midpoint = (best_bid + best_ask) / 2
            best_bid_volume = abs(depth.buy_orders[best_bid])
            best_ask_volume = abs(depth.sell_orders[best_ask])
            observation_bid = int(observation.bidPrice)
            observation_ask = int(observation.askPrice)
            
            
            self.variables.setdefault('macaron_start', self.get_vwmp(state.order_depths['MAGNIFICENT_MACARONS']))
            fair = self.get_fair_price_macaron(state, observation) + self.variables['macaron_start']
            panic_z, spread_z, volume_spread_z = self.get_indicators(depth, fair, midpoint)

            
            spread_threshold = 1.75
            spread_close_threshold = .1
            
            sunlightIndex = observation.sunlightIndex  
            
            buy_condition = True #abs(spread_z)
            sell_condition = True #abs(spread_z)
            close_condition = abs(spread_z) > spread_close_threshold
            
            
            logger.print(f'Fair: {fair}, panic_z: {panic_z}, volume_s: {volume_spread_z}, spread: {spread_z}')
            
            
        
        
            self.variables.setdefault('target_position_macarons', 0)
            target_position = self.variables['target_position_macarons']
            
            if state.timestamp/100 > 100:
            
                threshold = 0      
            
                # Buys / Sells
                position = state.position.get(product, 0)
                if best_ask < fair + threshold and buy_condition:
                    target_position = 75
                elif best_bid > fair - threshold and sell_condition:
                    target_position = -75
                
                
                
                # Conversions
                if implied_ask < fair + threshold and buy_condition:
                    self.conversions += 10   
                elif implied_bid > fair - threshold and sell_condition:
                    self.conversions -= 10
                
                
                # Clears
                if close_condition:
                    position = 0
                    
                    
                    
                if position != target_position:
                    orders[product].extend(self.place_target_order(state, product, target_position))
                    
                self.variables['target_position_macarons'] = target_position
            return orders
        
    def place_target_order(self, state, product, target_pos):
        depth = state.order_depths[product]
        position = state.position.get(product, 0)
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
        best_bid_volume = depth.buy_orders[best_bid] if depth.buy_orders else 0
        best_ask_volume = depth.sell_orders[best_ask] if depth.sell_orders else 0
        
        target_quantity = target_pos - position
        sign = 1 if target_quantity > 0 else -1
        
        price = best_bid if sign < 0 else best_ask
        volume = best_bid_volume if sign < 0 else best_ask_volume
        
        if price == 0 or volume == 0:
            return []
        
        availability = self.get_liquidity(state, product, 0, sign)
        order_size = min(abs(volume), abs(availability), abs(int(target_quantity)))*sign
        
        return [Order(product, price, order_size)]
        
        
    def run(self, state: TradingState):
        result = {}
        self.conversions = 0
        trader_data = ""
        
        if state.timestamp == 0:
            self.data = {}
            self.variables = {}
        
        
        for key in self.data.keys():
            self.data[key] = self.data[key][-100:]
        
        #set limits for commodities
        self.get_limits()


        orders =  self.get_macaron_orders(state)
        items = ['MAGNIFICENT_MACARONS']
        if 'MAGNIFICENT_MACARONS' in state.order_depths.keys():
            for item in items:
                if orders is not None:
                    if item in orders:
                        result.setdefault(item, []).extend(orders[item])
        

        traderData = self.serialize()
        
        conversions = self.conversions
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, traderData
