# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:21:36 2025

@author: rspaeth1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

if (__name__ == "__main__"):
    prices = pd.read_csv("C:\\Users\\rspaeth1\\Documents\\Prosperity\\prices_round_0.csv", sep=';', header=0)
    trades = pd.read_csv('C:\\Users\\rspaeth1\\Documents\\Prosperity\\trades_round_0.csv', sep=';', header=0)
    
    
    # Calculate trade value (VWP = price * quantity)
    trades['VWP'] = trades['price'] * trades['quantity']
    
    # Aggregate VWAP per timestamp
    VWAPs = trades.groupby('timestamp').agg({
        'VWP': 'sum',   # Sum of price * quantity
        'quantity': 'sum'  # Sum of quantities
    }).reset_index()
    
    # Compute VWAP
    VWAPs['VWAP'] = VWAPs['VWP'] / VWAPs['quantity']
    
    # Merge back to the original trades dataframe if needed
    trades = trades.merge(VWAPs[['timestamp', 'VWAP']], on='timestamp', how='left')
    
    #save as excel
    #prices.to_excel('prices.xlsx', sheet_name='Prices', index=False)
    #trades.to_excel('trades.xlsx', sheet_name='Trades', index=False)
    
    #show top 5
    '''print("\nPrices\n")
    print(prices.head())
    print("\nTrades\n")
    print(trades.head())'''
    
    #find unbiased estimator for resin price (stable throughout history)
    trade_dict = trades.to_dict(orient='index')
    #print(trade_dict)
    
    #get results for resin
    resin = []
    resin_vwp = []
    resin_volume = 0
    
    for trade in trade_dict.values():
        if trade['symbol'] == 'RAINFOREST_RESIN':
            resin.append(trade['price'])
            resin_vwp.append(trade['VWP'])
            resin_volume += trade['quantity']
    
    resin = np.array(resin)
    mean = np.sum(resin) / (len(resin)) #get mean estimator
    vwap = np.sum(resin_vwp) / resin_volume
    std = np.std(resin)
    
    #print(f"Sum: {np.sum(resin)}")
    print(f"Resin Mean: {mean}")
    print(f"Resin VWAP: {vwap}")
    print(f"Resin STD: {std}")
    
    resin = prices[prices['product'] == "RAINFOREST_RESIN"].copy()
    
    resin['price_100'] = resin['mid_price']/100
    x_values = resin['timestamp']
    y_values = resin['price_100']
    plt.ylim(99.94,100.06)
    plt.plot(x_values, y_values)
    plt.title("Resin_Prices_100")
    plt.show()
    
    
    # Parameters for the OU process
    theta = .8      # Speed of mean reversion
    mu = 100        # Long-term mean
    sigma = 1.158/100      # Volatility
    X0 = 100         # Initial value
    T = 2000        # Total time
    dt = 1       # Time step
    N = int(T / dt)  # Number of time steps
    
    # Pre-allocate array for efficiency
    X = np.zeros(N)
    X[0] = X0
    
    # Generate the OU process
    for t in range(1, N):
        dW = np.sqrt(dt) * np.random.normal(0, 1)
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW
    
    # Plot the result
    
    plt.clf()
    plt.plot(np.linspace(0, T, N), X)
    plt.title("Ornstein-Uhlenbeck Process Simulation")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.show()
    
    
    
    #something for kelp
    
    
    #get results for kelp
    kelp = []
    kelp_vwp = []
    kelp_up = 0
    kelp_avg_up = 0
    kelp_up_up = 0
    kelp_down = 0
    kelp_avg_down = 0
    kelp_down_down = 0
    kelp_up_down = 0
    kelp_down_up = 0
    kelp_total = 0
    kelp_volume = 0
    
    trade = list(trade_dict.values())
    vwps = {}
    qtys = {}
    vwaps = {}
    
    for index in range(0,len(trade)):
        if trade[index]['symbol'] == 'KELP':
            kelp.append(trade[index]['price'])
            kelp_vwp.append(trade[index]['VWP'])
            kelp_volume += trade[index]['quantity']
            if index != 0:
                if trade[index-1]['price'] < trade[index]['price']:
                    kelp_up += 1
                    kelp_avg_up += trade[index]['price'] - trade[index-1]['price']
                elif trade[index-1]['price'] > trade[index]['price']:
                    kelp_avg_down += abs(trade[index]['price'] - trade[index-1]['price'])
                    kelp_down += 1
            if index != 1:
                if trade[index-2]['price'] < trade[index-1]['price'] and trade[index-1]['price'] < trade[index]['price']:
                    kelp_down_down += 1
                if trade[index-2]['price'] < trade[index-1]['price'] and trade[index-1]['price'] > trade[index]['price']:
                    kelp_down_up += 1
                if trade[index-2]['price'] > trade[index-1]['price'] and trade[index-1]['price'] > trade[index]['price']:
                    kelp_up_up += 1
                if trade[index-2]['price'] > trade[index-1]['price'] and trade[index-1]['price'] < trade[index]['price']:
                    kelp_up_down += 1
            dict_index = trade[index]['timestamp']
            vwps[dict_index] = vwps.get(dict_index,0) + trade[index]['VWP']
            qtys[dict_index] = qtys.get(dict_index,0) + trade[index]['quantity']
        
    for key, value in vwps.items():
        vwaps[key] = value / qtys[key]
        
                    
                    
    kelp = np.array(kelp)
    mean = np.sum(kelp) / (len(kelp)) #get mean estimator
    vwap = np.sum(kelp_vwp) / kelp_volume
    var = np.sum((kelp - mean)**2)/(len(kelp)-1)
    std = var**.5
    kelp_total = kelp_up + kelp_down
    print(kelp_avg_up)
    kelp_avg_up /= kelp_up
    
    kelp_avg_down /= kelp_down
 
    kelp_up = kelp_up / kelp_total
    kelp_down = kelp_down / kelp_total
    kelp_up_total = kelp_up_up + kelp_up_down
    kelp_down_total = kelp_down_up + kelp_down_down
    kelp_up_up /= kelp_up_total
    kelp_up_down /= kelp_up_total
    kelp_down_up /= kelp_down_total
    kelp_down_down /= kelp_down_total
    
    #print(f"Sum: {np.sum(kelp)}")
    print(f"Kelp Mean: {mean}")
    print(f"Kelp VWAP: {vwap}")
    print(f"Kelp var: {var}")
    print(f"Kelp std: {std}")
    print(f"Kelp up: {kelp_up}")
    print(f"Kelp down: {kelp_down}")
    print(f"Kelp avg-up: {kelp_avg_up}")
    print(f"Kelp avg-down: {kelp_avg_down}")
    print(f"Kelp u-u: {kelp_up_up}")
    print(f"Kelp u-d: {kelp_up_down}")
    print(f"Kelp d-u: {kelp_down_up}")
    print(f"Kelp d-d: {kelp_down_down}")
    
    prices = prices[prices['product'] == "KELP"]
    trades = trades[trades['symbol'] == "KELP"]
    prices['returns'] = prices['mid_price'].diff()
    prices['pct_returns'] = prices['mid_price'].pct_change() + 1
    prices['log_returns'] = prices.apply(lambda row: math.log(row['pct_returns']), axis=1)
    prices['rolling_returns'] = prices['returns'].rolling(10, win_type=None).mean()
    
    vwap_prices = pd.DataFrame(list(vwaps.items()), columns=['timestamp', 'vwap'])
    vwap_prices['returns'] = vwap_prices['vwap'].diff()
    vwap_prices['rolling_returns'] = vwap_prices['returns'].rolling(10, win_type=None).mean()
    
    
    vwap_prices.plot(x='timestamp', y='vwap', kind='line', figsize=(25,15), label="vwap", color="green")
    vwap_prices.plot(x='timestamp', y='returns', kind='line', figsize=(25,15), label="vwap returns", color="green")
    vwap_prices.plot(x='timestamp', y='rolling_returns', kind='line', figsize=(25,15), label="vwap rolling", color="green")
    
    prices.plot(x='timestamp', y='mid_price', kind='line', figsize=(25,15), label="prices")
    prices.plot(x='timestamp', y='rolling_returns', kind='line', label='rolling_returns', color='purple', figsize=(25,15))
    prices.plot(x='timestamp', y='returns', kind='line', label='returns', color='red', figsize=(25,15))
    prices.plot(x='timestamp', y='pct_returns', kind='line', label='pct_returns', color='red', figsize=(25,15))
    prices.plot(x='timestamp', y='log_returns', kind='line', label='log_returns', color='green', figsize=(25,15))

    #plt.xlabel("Timestamp")
    #plt.ylabel("Sales")
    #plt.ylim(2014,2030)
    #plt.xlim()
    #plt.show()
    
    #price
    kelp_returns_mean = np.array(prices['returns'])
    kelp_returns_mean = kelp_returns_mean[~np.isnan(kelp_returns_mean)]
    kelp_returns_mean = kelp_returns_mean[kelp_returns_mean != 0]
    
    kelp_returns_mean_rolling = np.array(prices['rolling_returns'])
    kelp_returns_mean_rolling = kelp_returns_mean_rolling[~np.isnan(kelp_returns_mean_rolling)]
    kelp_returns_mean_rolling = kelp_returns_mean_rolling[kelp_returns_mean_rolling != 0]
    
    kelp_returns_mean_arr = kelp_returns_mean
    kelp_returns_mean_rolling_arr = kelp_returns_mean_rolling
    
    kelp_returns_mean = np.mean(kelp_returns_mean_arr)
    kelp_returns_std = np.std(kelp_returns_mean_arr)
    kelp_returns_mean_rolling = np.mean(kelp_returns_mean_rolling)
    kelp_returns_std_rolling = np.std(kelp_returns_mean_rolling_arr)
    
    #vwap
    kelp_returns_mean_vwap = np.array(vwap_prices['returns'])
    kelp_returns_mean_vwap = kelp_returns_mean_vwap[~np.isnan(kelp_returns_mean_vwap)]
    kelp_returns_mean_vwap = kelp_returns_mean_vwap[kelp_returns_mean_vwap != 0]
    
    kelp_returns_mean_rolling_vwap = np.array(vwap_prices['rolling_returns'])
    kelp_returns_mean_rolling_vwap = kelp_returns_mean_rolling_vwap[~np.isnan(kelp_returns_mean_rolling_vwap)]
    #kelp_returns_mean_rolling_vwap = kelp_returns_mean_rolling_vwap[kelp_returns_mean_rolling_vwap != 0]
    
    kelp_returns_mean_arr_vwap = kelp_returns_mean_vwap
    kelp_returns_mean_rolling_arr_vwap = kelp_returns_mean_rolling_vwap
    
    kelp_returns_mean_vwap = np.mean(kelp_returns_mean_arr_vwap)
    kelp_returns_std_vwap = np.std(kelp_returns_mean_arr_vwap)
    kelp_returns_mean_rolling_vwap = np.mean(kelp_returns_mean_rolling_vwap)
    kelp_returns_std_rolling_vwap = np.std(kelp_returns_mean_rolling_arr_vwap)
    
    print(f"Kelp returns mean: {kelp_returns_mean}")
    print(f"Kelp returns std: {kelp_returns_std}")
    print(f"Kelp returns mean rolling: {kelp_returns_mean_rolling}")
    print(f"Kelp returns std rolling: {kelp_returns_std_rolling}")
    
    print("---------VWAP based---------")
    print(f"Kelp returns mean: {kelp_returns_mean_vwap}")
    print(f"Kelp returns std: {kelp_returns_std_vwap}")
    print(f"Kelp returns mean rolling: {kelp_returns_mean_rolling_vwap}")
    print(f"Kelp returns std rolling: {kelp_returns_std_rolling_vwap}")
    
    