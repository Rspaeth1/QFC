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
    
    #####################################################################
    
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
            'variables': self.variables,
            'traderObject': self.traderObject
        }
        
        return jsonpickle.encode(var)
    
    def deserialize(self, json):
        var = jsonpickle.decode(json)
        self.data = var['data']
        self.variables = var['variables']
        self.traderObject = var['variables']
        
    ################################################################ 
    
    def squid_ink_returns(self, state):
        depth = state.order_depths
        
        squid_ink_depth = depth['SQUID_INK']
        price = self.get_vwmp(squid_ink_depth)
        
        basket = "SQUID_INK_RETURNS"
        window = 100
        num_std = 3
        position = state.position.get('SQUID_INK', 0)

        product = basket+"_RETURNS"
        
        self.variables.setdefault('SQUID_INK_LAST_PRICE', 0)
        last_price = self.variables['SQUID_INK_LAST_PRICE']
        current_return = price/last_price - 1 if last_price != 0 else 0
        self.variables['SQUID_INK_LAST_PRICE'] = price

        self.variables.setdefault(product+'_signal', None)
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
        bid_volume = abs(squid_ink_depth.buy_orders[best_bid])
        ask_volume = abs(squid_ink_depth.sell_orders[best_ask])
        
            
        if self.variables[product+'_signal'] == "SELL":
            INK_make_orders, _, _ = self.make_orders(
                Product.INK,
                squid_ink_depth,
                price,
                position,
                bid_volume,
                ask_volume,
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
        return None
        
    def get_implied_bid_ask(self, observation):
        storage_cost = .1
        return observation.bidPrice - observation.exportTariff - observation.transportFees - storage_cost, observation.askPrice + observation.importTariff - observation.transportFees
        
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
        baseline_spread = 8.2
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
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            observation_bid = int(observation.bidPrice)
            observation_ask = int(observation.askPrice)
            self.variables.setdefault('macaron_start', self.get_vwmp(state.order_depths['MAGNIFICENT_MACARONS']))
            fair = self.get_fair_price_macaron(state, observation) + self.variables['macaron_start']
            logger.print(f'Fair value: {fair}')
            _, spread_z, __ = self.get_indicators(depth, fair, midpoint)
            
            self.variables.setdefault('last_buy', midpoint)
            buy_exit_condition = self.variables['last_buy'] < best_bid
            sell_exit_condition = self.variables['last_buy'] > best_ask
            
            
            buy_exit_condition = abs(spread_z) > .5 and buy_exit_condition
            sell_exit_condition = abs(spread_z) > .5 and sell_exit_condition
            
            
            total_pos_change = 0
            
            if state.timestamp/100 > 100:
            
                logger.print(f"best ask: {best_ask}, best_bid: {best_bid}, fair: {fair}")
                self.variables.setdefault('last_buy', 0)
                position = state.position.get(product, 0)
                if best_ask < fair or (position < 0 and sell_exit_condition):
                   
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    
                    logger.print(f'trying to buy {order_quantity} at {best_ask}')
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                    
                    if implied_ask < best_ask:
                        self.conversions = min(10, order_quantity)
                    
                    self.variables['last_buy'] = best_ask
                    
                    total_pos_change += order_quantity
                elif fair < best_bid or (position > 0 and buy_exit_condition):
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    
                    orders[product].extend([Order(product, int(best_bid), -int(order_quantity))])
                    
                    if implied_bid > best_bid:
                        self.conversions = max(-10, -order_quantity)
                    
                    self.variables['last_buy'] = best_bid
                    total_pos_change += -order_quantity
                
                """
                if implied_ask < fair:
                    availability = self.get_liquidity(state, product, total_pos_change, 'BUY')
                    conversion_availability = self.get_liquidity(state, product+"_CONVERSION", 0, 'BUY', self.conversions)
                    order_quantity = min(10, abs(availability), abs(conversion_availability))
                    
                    orders[product].extend([Order(product, int(observation_ask), order_quantity)])
                    self.conversions += order_quantity     
                    logger.print('buy conversion')
                    logger.print(f'o_ask: {observation_ask}, i_ask: {implied_ask}, bid: {best_bid}')
                elif fair < implied_bid:
                    availability = self.get_liquidity(state, product, total_pos_change, 'SELL')
                    conversion_availability = self.get_liquidity(state, product+"_CONVERSION", 0, 'SELL', self.conversions)
                    order_quantity = min(10, abs(availability), abs(conversion_availability))
                    
                    orders[product].extend([Order(product, int(observation_bid), -order_quantity)])
                    self.conversions -= order_quantity
                    logger.print('sell conversion')
                    """
            return orders
        
    def get_r2_orders(self, state):
        items = ['PICNIC_BASKET1', 'PICNIC_BASKET2', 'JAMS', 'CROISSANTS', 'DJEMBES']
        orders = {item: [] for item in items}
        pos_change = {item: 0 for item in items}
        
        # PICNIC BASKET 1
        product = "PICNIC_BASKET1"
        if product in state.market_trades:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                quantity_threshold = 10
                volume_threshold = 0
                
                if (buyer == "Penelope" or seller == "Camilla") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                   
                elif (seller == "Penelope" or buyer == "Camilla") and quantity >= quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
                    
                   
        # PICNIC BASKET 2
        product = "PICNIC_BASKET2"
        if product in state.market_trades:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                
                quantity_threshold = 10
                volume_threshold = 0
                
                if (seller == "Penelope" or buyer == "Camilla" ) and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                    
                elif (buyer == "Penelope" or seller == "Camilla") and quantity >= quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
                    
                
        """
        # JAMS
        product = "JAMS"
        if product in state.market_trades:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                
                quantity_threshold = 10
                volume_threshold = 0
                
                if (buyer == "Camilla") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                elif (seller == "Caesar") and quantity >= quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
        """
        
        # CROISSANTS
        product = "CROISSANTS"
        if product in state.market_trades:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                
                quantity_threshold = 0
                volume_threshold = 0
                
                if (buyer == "Olivia") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                elif (seller == "Olivia") and quantity >= quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
                    
        """
        # DJEMBES
        product = "DJEMBES"
        if 'PICNIC_BASKET2' in state.market_trades:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades['PICNIC_BASKET2']:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                
                quantity_threshold = 10
                volume_threshold = 0
                
                if (buyer == "Penelope") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask)+10, int(order_quantity))])
                elif (seller == "Penelope") and quantity >= quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
               """
                    
        return orders
    
    def get_r3_orders(self, state):
        items = [
            'VOLCANIC_ROCK',
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500'
        ]
        
        orders = {item: [] for item in items}
        
        options = {item: {} for item in items}
        option_used = "VOLCANIC_ROCK_VOUCHER_10000"
        strike = 9250
        for _, option in options.items():
            if _ == "VOLCANIC_ROCK":
                continue
            S = self.get_vwmp(state.order_depths['VOLCANIC_ROCK']) 
            strike += 250
            K = strike
            T = ((7 - 5) - (state.timestamp / 1000000)) / 365
            m_t = np.log(K / S) / np.sqrt(T)
            m_t_threshold = .3
            option['ATM'] = True if abs(m_t) <= m_t_threshold else False
            if option['ATM']:
                option_used = _
        
        # rock
        product = "VOLCANIC_ROCK"

        if option_used in state.market_trades:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[option_used]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                quantity_threshold = 17
                volume_threshold = 0
                
                if (buyer == "Penelope") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                elif (seller == "Penelope") and quantity == quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
        
        
        # 9500
        product = "VOLCANIC_ROCK_VOUCHER_9500"
        if product in state.market_trades and options[product]['ATM']:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                quantity_threshold = 17
                volume_threshold = 0
                
                if (buyer == "Penelope") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                elif (seller == "Penelope") and quantity == quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
        
        # 9750
        product = "VOLCANIC_ROCK_VOUCHER_9750"
        if product in state.market_trades and options[product]['ATM']:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                quantity_threshold = 17
                volume_threshold = 0
                
                if (buyer == "Penelope") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                elif (seller == "Penelope") and quantity == quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
        
        
        # 10000
        product = "VOLCANIC_ROCK_VOUCHER_10000"
        if product in state.market_trades and options[product]['ATM']:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                quantity_threshold = 17
                volume_threshold = 0
                
                if (buyer == "Penelope") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                elif (seller == "Penelope") and quantity == quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
        
        # 10250
        product = "VOLCANIC_ROCK_VOUCHER_10250"
        if product in state.market_trades and options[product]['ATM']:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                quantity_threshold = 17
                volume_threshold = 0
                
                if (buyer == "Penelope") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                elif (seller == "Penelope") and quantity == quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
        
        # 10500
        product = "VOLCANIC_ROCK_VOUCHER_10500"
        if product in state.market_trades and options[product]['ATM']:
            depth = state.order_depths[product]
            best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume = self.get_prices(depth)
            for trade in state.market_trades[product]:
                buyer = trade.buyer
                seller = trade.seller
                quantity = abs(trade.quantity)
                quantity_threshold = 17
                volume_threshold = 0
                
                if (buyer == "Penelope") and quantity >= quantity_threshold and best_ask_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'BUY')
                    order_quantity = min(abs(availability), best_ask_volume)
                    orders[product].extend([Order(product, int(best_ask), int(order_quantity))])
                elif (seller == "Penelope") and quantity == quantity_threshold and best_bid_volume >= volume_threshold:
                    availability = self.get_liquidity(state, product, 0, 'SELL')
                    order_quantity = min(abs(availability), best_bid_volume)
                    orders[product].extend([Order(product, int(best_bid), int(-order_quantity))])
        
        return orders
    
    def get_r4_orders(self, state):
        orders = {'MAGNIFICENT_MACARONS': []}
        
        # MAGNIFICENT_MACARONS
        product = "MAGNIFICENT_MACARONS"
        
        if not product in state.observations.conversionObservations:
            return {}

        return orders
        
        
    def get_prices(self, depth):
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
        midpoint = (best_bid + best_ask) / 2 if best_bid != 0 and best_ask != 0 else 0 
        best_bid_volume = abs(depth.buy_orders[best_bid]) if depth.buy_orders else 0
        best_ask_volume = abs(depth.sell_orders[best_ask]) if depth.sell_orders else 0
        
        return best_bid, best_ask, midpoint, best_bid_volume, best_ask_volume
        
    def run(self, state: TradingState):
        result = {}
        self.conversions = 0
        trader_data = ""
        self.traderObject = {}
        
        if state.timestamp == 0:
            self.data = {}
            self.variables = {}
        else:
            self.deserialize(state.traderData) # added after the fact... cant believe I forgot this in the submission
        
        
        for key in self.data.keys():
            self.data[key] = self.data[key][-100:]
        
        #set limits for products
        self.get_limits()


        #R1
        if "RAINFOREST_RESIN" in state.order_depths:
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
            
            if "SQUID_INK" in state.order_depths:
                orders = self.squid_ink_returns(state)
                if orders is not None:
                    if 'SQUID_INK' in orders:
                        result['SQUID_INK'] = orders['SQUID_INK']
            
            if "KELP" in state.order_depths:
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

        #R2
        orders =  self.get_r2_orders(state)
        items = ['PICNIC_BASKET1', 'PICNIC_BASKET2', 'JAMS', 'CROISSANTS', 'DJEMBES']
        for item in items:
            if item in state.order_depths.keys():
                if orders is not None:
                    if item in orders:
                        result.setdefault(item, []).extend(orders[item])
                        
        #R3
        items = [
            'VOLCANIC_ROCK',
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500'
        ]
        orders = self.get_r3_orders(state)
        for item in items:
            if item in state.order_depths.keys():
                if orders is not None:
                    if item in orders:
                        result.setdefault(item, []).extend(orders[item])
    

        #R4
        orders = None#self.get_macaron_orders(state)
        orders = self.get_r4_orders(state)
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
