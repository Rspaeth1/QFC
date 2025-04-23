# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:06:05 2025

@author: Ryan
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

_round_ = 2
product = ["SQUID_INK"]
"""
# --- Utility function to process a single day of data ---
def process_day(prices_file, trades_file):
    # Load data
    book_df = pd.read_excel(prices_file)#, sep=';', header=0)
    trade_df = pd.read_excel(trades_file)#, sep=';', header=0)

    # Filter for SQUID_INK
    book_df = book_df[book_df['product'].isin(product)].copy()
    trade_df = trade_df[trade_df['symbol'].isin(product)].copy()

    # Compute midprice if not in file already
    if 'mid_price' not in book_df.columns:
        book_df['mid_price'] = (book_df['bid_price_1'] + book_df['ask_price_1']) / 2

    book_df['returns'] = book_df['mid_price'].diff().fillna(0)
    book_df['recent_vol'] = book_df['returns'].rolling(50).std()
    book_df['rolling_returns'] = book_df['returns'].rolling(50).mean()
    book_df['z_score'] = (book_df['returns'] - book_df['rolling_returns'])/book_df['recent_vol']
    #print(book_df.head(20))
    

    # Match book state before each trade
    aggressors = []
    for _, row in trade_df.iterrows():
        trade_time = row['timestamp']
        trade_price = row['price']
        book_row = book_df[book_df['timestamp'] <= trade_time].iloc[-1]
        best_ask = book_row['ask_price_1']
        best_bid = book_row['bid_price_1']
        book_bid_vol = book_row['bid_volume_1']
        book_ask_vol = book_row['ask_volume_1']
        
        #z score testing


        if trade_price >= best_ask:
            aggressor = 'buyer'
        elif trade_price <= best_bid:
            aggressor = 'seller'
        else:
            aggressor = 'unknown'
        aggressors.append(aggressor)

    trade_df['aggressor'] = aggressors

    # Feature engineering
    trade_df['rolling_volume'] = trade_df['quantity'].rolling(10).sum()
    trade_df['current_volume'] = trade_df['quantity']
    trade_df['time_since_last_trade'] = trade_df['timestamp'].diff().fillna(0)
    trade_df['rolling_direction'] = trade_df['aggressor'].eq('buyer').rolling(10).mean()
    #trade_df['book_imbalance'] = (book_bid_vol - book_ask_vol) / (book_bid_vol + book_ask_vol)
    #trade_df['spread_position'] = (trade_price - best_bid) / (best_ask - best_bid)


    




    # Clustering
    
    # Make sure both are sorted by timestamp
    book_df = book_df.sort_values('timestamp')
    trade_df = trade_df.sort_values('timestamp')
    
    # Perform the asof merge — use direction='backward' to get the last known book before the trade
    trade_df = pd.merge_asof(
        trade_df,
        book_df[['timestamp', 'bid_volume_1', 'bid_volume_2', 'bid_volume_3', 
                 'ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'z_score', 'recent_vol', 'mid_price']],
        on='timestamp',
        direction='backward'
    )
    
    trade_df['L1_Consumption'] = np.where(
    trade_df['aggressor'] == 'buyer',
    trade_df['quantity'] / trade_df['ask_volume_1'],
    trade_df['quantity'] / trade_df['bid_volume_1']
    )
    
    trade_df['L2_Consumption'] = np.where(
        trade_df['aggressor'] == 'buyer',
        trade_df['quantity'] / trade_df['ask_volume_2'],
        trade_df['quantity'] / trade_df['bid_volume_2']
    )
    
    trade_df['L3_Consumption'] = np.where(
        trade_df['aggressor'] == 'buyer',
        trade_df['quantity'] / trade_df['ask_volume_3'],
        trade_df['quantity'] / trade_df['bid_volume_3']
    )



    standard_features = [
        'rolling_volume', 
        #'current_volume',
        'rolling_direction',
        #'book_imbalance', 
        'z_score', 
        'bid_volume_1',
        'bid_volume_2',
        'bid_volume_3',
        'ask_volume_1',
        'ask_volume_2',
        'ask_volume_3',
        'L1_Consumption',
        'L2_Consumption',
        'L3_Consumption',
        'time_since_last_trade', 
        'recent_vol',
    ]
    
    log_features = [
        #'time_since_last_trade', 
        #'recent_vol',
        #'mid_price'
    ]
    
    features = standard_features + log_features
    
    plot = False
    if plot:
        for column in features.columns:
            plt.figure()
            plt.hist(features[column])
            plt.title(f"Histogram for {column}")
            plt.show()
    
    
    standard_data = trade_df[standard_features].fillna(0)
    #log_data = trade_df[log_features].fillna(0)
    
    #log_scaled = np.log1p(log_data)

    feature_means = np.mean(standard_data, axis=0)
    feature_stds = np.std(standard_data, axis=0)
    
    # Print them cleanly for copy-paste into your bot
    print("\nFeature Means:")
    print([round(m, 6) for m in feature_means])
    
    print("\nFeature STDs:")
    print([round(s, 6) for s in feature_stds])
   
    #print(trade_df.head(100))
    
    standard_scaler = StandardScaler()
    #log_scaler = StandardScaler()
    
    standard_scaled = standard_scaler.fit_transform(standard_data)
    #log_scaled_std = log_scaler.fit_transform(log_scaled)
    
    #features_scaled = np.hstack([standard_scaled, log_scaled_std])
    
    kmeans = KMeans(n_clusters=5, random_state=42)  # vary seed per day
    trade_df['bot_cluster'] = kmeans.fit_predict(standard_scaled)
    
    
    
    print('\nCluster centers')
    print(kmeans.cluster_centers_)

    return book_df, trade_df



# Process all three days

price_file = "C:\\Users\\Ryan\\Downloads\\Prosperity\\prices_r"+str(_round_)+".xlsx"
trade_file = "C:\\Users\\Ryan\\Downloads\\Prosperity\\trades_r"+str(_round_)+".xlsx"


price_file ="C:\\Users\\Ryan\\Downloads\\Prosperity\\prices_round_1_day_-2.csv"
trade_file ="C:\\Users\\Ryan\\Downloads\\Prosperity\\trades_round_1_day_-2_nn.csv"


price_file ="C:\\Users\\Ryan\\Downloads\\Prosperity\\prices_round_1_day_-1.csv"
trade_file ="C:\\Users\\Ryan\\Downloads\\Prosperity\\trades_round_1_day_-1_nn.csv"


price_file ="C:\\Users\\Ryan\\Downloads\\Prosperity\\prices_round_1_day_0.csv"
trade_file ="C:\\Users\\Ryan\\Downloads\\Prosperity\\trades_round_1_day_0_nn.csv"



book, trades = process_day(price_file, trade_file)






trades.to_csv("clustered_trades.csv", index=False)
book.to_csv("book_data.csv", index=False)
"""
trades = pd.read_csv("clustered_trades.csv")      
book = pd.read_csv("book_data.csv")

"""
# Setup plot
plt.figure(figsize=(14, 6))

# Plot midprice from the book data
#plt.plot(book_df['timestamp'], book_df['mid_price'], label='Midprice', color='black', linewidth=1.5, alpha=0.8)

# Pick colors for clusters
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Plot each cluster's trades as dots
for cluster in trades['bot_cluster'].unique():
    cluster_data = trades[trades['bot_cluster'] == cluster]
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['price'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['bid_volume_1'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['ask_volume_1'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['bid_volume_2'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['ask_volume_2'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['bid_volume_3'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['ask_volume_3'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['recent_vol'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['z_score'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
    plt.show()
    plt.figure()
    plt.scatter(cluster_data['timestamp'], cluster_data['quantity'],
                label=f'Cluster {cluster}', s=20, alpha=0.3, color=colors[cluster % len(colors)])
    plt.show()
 

plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.title("SQUID_INK: Trade Clusters vs Midprice")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

#get sample sizes
print(trades['bot_cluster'].value_counts())


# Analyze average future price movement after each cluster trade
lookaheads = [5, 10, 30, 100]

use_combined_filters = True  

filter_conditions = {
    'quantity': [1, 5, 10, 15],
    #'z_score': [0.5, 1.0, 1.5, 2.0],
    #'recent_vol': [0.8, 1.0, 1.2, 1.5],
    #'L1_Consumption': [0.3, 0.5, 0.7],
    #'L2_Consumption': [0.3, 0.5, 0.7],
    #'L3_Consumption': [0.3, 0.5, 0.7]
}

for lookahead in lookaheads:
    print(f"\n\n=== Lookahead: {lookahead} ===")

    results = []

    for cluster in trades['bot_cluster'].unique():
        cluster_trades = trades[trades['bot_cluster'] == cluster]

        if use_combined_filters:
            #  Use this custom filter block
            combined_mask = (
                (cluster_trades['aggressor'].isin(['buyer', 'seller'])) &
                #(cluster_trades['quantity'] >= 10) &
                #(cluster_trades['quantity'] <= 15) &
                (cluster_trades['ask_volume_1'] > 30)&
                (cluster_trades['recent_vol'] > 1)
            )
            filtered_trades = cluster_trades[combined_mask]

            impacts = []
            for _, row in filtered_trades.iterrows():
                t = row['timestamp']
                price_now = row['price']
                future_book = book[book['timestamp'] > t]

                if not future_book.empty and len(future_book) >= lookahead + 1:
                    center = lookahead - 1
                    window = future_book.iloc[max(center - 1, 0): center + 2]
                    future_price = window['mid_price'].mean()
                    impacts.append(future_price - price_now)

            if impacts:
                avg_impact = sum(impacts) / len(impacts)
                results.append((cluster, "combined", "custom", avg_impact, len(impacts)))

        else:
            # Loop through each individual feature & threshold
            base_mask = cluster_trades['aggressor'].isin(['buyer', 'seller'])

            for feature, thresholds in filter_conditions.items():
                for threshold in thresholds:
                    feature_mask = cluster_trades[feature] > threshold
                    filtered_trades = cluster_trades[base_mask & feature_mask]

                    impacts = []
                    for _, row in filtered_trades.iterrows():
                        t = row['timestamp']
                        price_now = row['price']
                        future_book = book[book['timestamp'] > t]

                        if not future_book.empty and len(future_book) >= lookahead + 1:
                            center = lookahead - 1
                            window = future_book.iloc[max(center - 1, 0): center + 2]
                            future_price = window['mid_price'].mean()
                            impacts.append(future_price - price_now)

                    if impacts:
                        avg_impact = sum(impacts) / len(impacts)
                        results.append((cluster, feature, threshold, avg_impact, len(impacts)))

    # Sort and print
    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

    print(f"\nTop filter results (lookahead = {lookahead}):")
    for cluster, feature, threshold, impact, count in sorted_results[:10]:
        print(f"Cluster {cluster} | {feature} > {threshold} → Impact: {impact:.4f} (n = {count})")




