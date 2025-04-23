# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:37:23 2025

@author: rjsyo
"""

from scipy.fft import fft, fftfreq
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

trades1 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\trades_round_1_day_-2_nn.csv", sep=';', header=0)
trades2 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\trades_round_1_day_-1_nn.csv", sep=';', header=0)
trades3 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\trades_round_1_day_0_nn.csv", sep=';', header=0)

#drop headers
trades2.drop(0)
trades3.drop(0)

trades = [trades1] #, trades2, trades3]
trades = pd.concat(trades, axis=0)

prices1 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\prices_round_1_day_-2.csv", sep=';', header=0)
prices2 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\prices_round_1_day_-1.csv", sep=';', header=0)
prices3 = pd.read_csv("C:\\Users\\rjsyo\\Downloads\\Prosperity\\prices_round_1_day_0.csv", sep=';', header=0)

#drop headers
prices2.drop(0)
prices3.drop(0)

prices = [prices2] #, prices2, prices3]
prices = pd.concat(prices, axis=0)

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


trade_dict = trades.to_dict(orient='index')

trade = list(trade_dict.values())
vwps = {'KELP': {}, 'SQUID_INK': {}}
qtys = {'KELP': {}, 'SQUID_INK': {}}
vwaps = {'KELP': {}, 'SQUID_INK': {}}

products = ['KELP', 'SQUID_INK']

"""
for index in range(0,len(trade)):
    for product in products:
        dict_index = trade[index]['timestamp']
        vwps[product][dict_index] = vwps[product].get(dict_index,0) + trade[index]['VWP']
        qtys[product][dict_index] = qtys[product].get(dict_index,0) + trade[index]['quantity']


for product in products:
    for key, value in vwps[product].items():
        vwaps[product][key] = value / qtys[product][key]
        
    vwap_prices = pd.DataFrame(list(vwaps[product].items()), columns=['timestamp', 'vwap'])
    vwap_prices['returns'] = vwap_prices['vwap'].diff()
    vwap_prices['rolling_returns'] = vwap_prices['returns'].rolling(10, win_type=None).mean()
    
    
    vwap_prices.plot(x='timestamp', y='vwap', kind='line', figsize=(25,15), label="vwap "+product, color="green")
    vwap_prices.plot(x='timestamp', y='returns', kind='line', figsize=(25,15), label="vwap returns "+product, color="green")
    vwap_prices.plot(x='timestamp', y='rolling_returns', kind='line', figsize=(25,15), label="vwap rolling "+product, color="green")
    """

"""
for product in products:
    temp = prices[prices['product'] == product]
    temp['returns'] = temp['mid_price'].diff()
    temp['rolling_returns'] = temp['returns'].rolling(100, win_type=None).mean()
    temp['rolling_dev'] = temp['returns'].rolling(100, win_type=None).std()
    temp['rolling_z'] = (temp['returns']-temp['rolling_returns'])/temp['rolling_dev']
    
    temp.plot(x='timestamp', y='mid_price', kind='line', figsize=(25,15), label="prices " +product)
    temp.plot(x='timestamp', y='rolling_z', kind='line', label='rolling_z '+product, color='purple', figsize=(25,15))
    temp.plot(x='timestamp', y='returns', kind='line', label='returns '+product, color='red', figsize=(25,15))
    
     # ----- FFT part -----
    returns = temp['returns'].dropna().values
    n = len(returns)
    timestep = 1  # assume 1 unit per step (e.g., 1 timestamp per row)
    
    # Apply Fourier Transform
    fft_vals = fft(returns)
    fft_freq = fftfreq(n, d=timestep)
    
    # Only take the positive half of frequencies (real-valued signals are symmetric)
    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    powers = np.abs(fft_vals[pos_mask])
    
    # Find dominant frequency
    dominant_freq = freqs[np.argmax(powers)]
    cycle_length = int(1 / dominant_freq)  # in number of data points
    print(f"{product}: Dominant freq: {dominant_freq}, Cycle length: {cycle_length}")
    
    # Plot the frequency domain
    plt.figure(figsize=(18, 5))
    plt.plot(freqs, powers)
    plt.title(f"{product} - FFT of Returns")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
"""
    
squid_freq = {}
prices1 = prices1[prices1['product'] == "SQUID_INK"]
prices2 = prices2[prices2['product'] == "SQUID_INK"]
prices3 = prices3[prices3['product'] == "SQUID_INK"]

prices1['returns'] = prices1['mid_price'].diff()
prices2['returns'] = prices2['mid_price'].diff()
prices3['returns'] = prices3['mid_price'].diff()



for ret in prices1['returns']:
    if math.isnan(ret):
        continue
    squid_freq[ret] = squid_freq.get(ret, 0) + 1
for ret in prices2['returns']:
    if math.isnan(ret):
        continue
    squid_freq[ret] = squid_freq.get(ret, 0) + 1
for ret in prices2['returns']:
    if math.isnan(ret):
        continue
    squid_freq[ret] = squid_freq.get(ret, 0) + 1
    

print(squid_freq)
"""
# Convert to numpy arrays
x = np.array(list(squid_freq.keys()))
y = np.array(list(squid_freq.values()))

# Normalize frequencies to form a probability distribution
y_normalized = y / np.sum(y)

# Fit normal distribution (mean and std)
mean = np.average(x, weights=y)
std = np.sqrt(np.average((x - mean) ** 2, weights=y))

# Generate smooth curve
x_smooth = np.linspace(min(x), max(x), 500)
pdf = norm.pdf(x_smooth, loc=mean, scale=std)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x_smooth, pdf, label='Fitted Normal Curve')
plt.bar(x, y_normalized, width=0.005, alpha=0.6, label='Historical Distribution')
plt.title("Return Distribution with Normal Fit")
plt.xlabel("Return")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()

print(f"Standard deviation: {std}, Mean: {mean}")
"""

#difference
kelp = prices[prices['product'] == 'KELP']
kelp['returns'] = kelp['mid_price'].diff()
ink = prices[prices['product'] == 'SQUID_INK']
ink['returns'] = ink['mid_price'].diff()
print(kelp.head())

temp = pd.merge(kelp, ink, on='timestamp', how='inner')
temp['mid_price'] = temp['mid_price_x'] - temp['mid_price_y']
temp['returns'] = temp['returns_x'] - temp['returns_y']
print(temp.head())

temp['rolling_returns'] = temp['returns'].rolling(15, win_type=None).mean()
    
temp.plot(x='timestamp', y='mid_price', kind='line', figsize=(25,15), label="prices diff")
temp.plot(x='timestamp', y='rolling_returns', kind='line', label='rolling_returns diff', color='purple', figsize=(25,15))
temp.plot(x='timestamp', y='returns', kind='line', label='returns diff', color='red', figsize=(25,15))
    
temp_rolling = np.array(temp['rolling_returns'])
temp_rolling = temp_rolling[~np.isnan(temp_rolling)]
#temp_rolling = temp_rolling[temp_rolling != 0]
temp_std = np.std(temp_rolling)
print(f"Diff std: {temp_std}")


