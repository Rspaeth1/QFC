# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:33:54 2025

@author: rjsyo
"""

import pandas as pd

prices1 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\round-2-island-data-bottle\\prices_round_2_day_-1.csv", sep=';', header=0)
prices2 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\round-2-island-data-bottle\\prices_round_2_day_0.csv", sep=';', header=0)
prices3 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\round-2-island-data-bottle\\prices_round_2_day_1.csv", sep=';', header=0)

prices = [prices1, prices2, prices3]

day = 0
price_df = prices[day]

items = ['CROISSANTS', 'JAMS', 'DJEMBES', 'PICNIC_BASKET1', 'PICNIC_BASKET2']
baskets = ['PICNIC_BASKET1', 'PICNIC_BASKET2']

weights = {
    'PICNIC_BASKET1': {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 1},
    'PICNIC_BASKET2': {'CROISSANTS': 4, 'JAMS': 2, 'DJEMBES': 0}
}


def get_vwmp(row):    
    best_bid = row['bid_price_1']
    best_ask = row['ask_price_1']
    bid_volume = row['bid_volume_1']
    ask_volume = row['ask_volume_1']
    
    return (best_bid * ask_volume + best_ask * bid_volume)/(bid_volume + ask_volume)

def mm_mid_basket(row, volume_cutoff=10):
    last_volume = 0
    for i in range(1,4):
        if row[f'bid_volume_{i}'] + last_volume >= volume_cutoff:
            best_bid = row[f'bid_price_{i}']
            break
        else:
            last_volume += row[f'bid_volume_{i}']
    else:
        best_bid = None
        
    
    for i in range(1,4):
        if row[f'ask_volume_{i}'] + last_volume >= volume_cutoff:
            best_ask = row[f'ask_price_{i}']
            break
        else:
            last_volume += row[f'ask_volume_{i}']
    else:
        best_ask = None
        
    if best_bid is not None and best_ask is not None:
        mid_price = (best_bid + best_ask) / 2
        return mid_price
    else:
        return row['mid_price']
    
def fair_price(row):
    if row['product'] in baskets:
        return mm_mid_basket(row, volume_cutoff=10)
    else:
        return get_vwmp(row)

price_df['fair'] = price_df.apply(fair_price, axis=1)
products = price_df['product'].unique()

columns = ['timestamp'] + list(products) + ['SYNTHETIC']
df_fairs = pd.DataFrame(columns=columns)

for timestamp in price_df['timestamp'].unique():
    rows = price_df[price_df['timestamp'] == timestamp]
    
    fairs = {}
    
    for product in products:
        fair = rows.loc[rows['product'] == product, 'fair'].values[0]
        fairs[product] = fair
        
    synthetic_fair1 = sum(fairs[product] * weights['PICNIC_BASKET1'].get(product, 0) for product in products)
    synthetic_fair2 = sum(fairs[product] * weights['PICNIC_BASKET2'].get(product, 0) for product in products)
    
    new_row = pd.DataFrame({'timestamp': [timestamp], **{product: [fairs[product]] for product in products}, 'SYNTHETIC_1': [synthetic_fair1], 'SYNTHETIC_2': [synthetic_fair2]})
    df_fairs = pd.concat([df_fairs, new_row], ignore_index=True)
    
df_fairs = df_fairs.reset_index(drop=True)

print(df_fairs.head(20))
