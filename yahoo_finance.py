# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 20:54:57 2025

@author: rspaeth1
"""

# This imports the open source yahoo finance library
import yfinance as yf

# This imports pandas, which allows us to manipulate dataframes
import pandas as pd

# This imports numpy, which is useful for calulations on arrays,
# especially with pandas
import numpy as np

# This imports matplotlib, which allows us to graph
import matplotlib.pyplot as plt

# This imports a couple types to make it clear when they are used later
from typing import List, Dict



# Notice this syntax is actually unneccessary, by writing 
# tickers: List we are just preemptively declaring that tickers
# should be a list. This is mostly useful for the readability of code.
def get_data(tickers: List): 
    """ This function downloads data for a list of tickers 
    then it puts it into a dictionary that we can use to 
    access the data afterwards """
    
    
    # You can download all the data in one call, but for simplicity,
    # I am doing it by looping through the tickers to make it clear
    # what exactly is going on here, since more formatting would be
    # required with the single call for the dataframe
    
    # First define data_dictionary where data is stored by ticker
    data_dictionary = {}
    
    # Then loop through each ticker
    for ticker in tickers:
        # Download the data for past 3 months, on the daily scale
        data_dictionary[ticker] = yf.download(ticker, period='3mo', interval='1d')
        
        # If you want to see the top 5 rows of each set of data,
        # run the below code
        # print(data_dictionary[ticker].head())
        
        
    # Send back our results to where we called the function
    return data_dictionary


def graph_data(data_dict: Dict):
    """ This function graphs our data given a dictionary containing data """
    
    # Loop through each ticker in the dict and plot it
    # Notice by using .items() on data_dict we can get
    # both the key and value stored in ticker and data,
    # respectively, which allows us to reference both
    for ticker, data in data_dict.items():
        
        # Establish the plot size
        plt.figure(figsize=(12,6))
        
        # Add the close data to the plot
        plt.plot(data.index, data['Close'])
        
        # Set the title, labels, and other parameters        
        plt.rcParams.update({'font.size': 12})
        plt.title(ticker + " Past 3 Months")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        
        # Finally, output our graph
        plt.show()
        
    
def graph_table(data_dict: Dict):
    """ This function graphs a table with key metrics given a dictionary
    with data about tickers """
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6,2))
    
    # Hide the axes
    ax.axis('off')
    
    # Create a dictionary to store key metrics
    metrics_dict = {
        'Index': [],
        '3m Return': [],
        '3m Volatility': [],
        'Current Price': []
    }
    
    # Loop through each ticker in the dict and get key metrics
    for ticker, data in data_dict.items():
        # Calculate return % and volatility
        
        # To get 3mo return, simply find % change on close price
        # Note that .iloc allows us to get the price at the index
        # that is being referenced, here 0 is first and -1 is last
        close = data['Close']
        close = close[ticker] # The data is 'multi-leveled' so we need to do this to get just the close price
        returns = ( close.iloc[-1] - close.iloc[0] ) / close.iloc[0] * 100
        
        # Formatting using f-string syntax, or string interpolation
        # Here 'f' indicates f-string syntax, and {} allows for a
        # variable to be cast to string, then :.2f rounds to 2 
        # decimal places, and then finally % is just added to the end
        returns = f"{returns:.2f}%"

        
        # To get volatility we will use numpy on close price
        # Convert to numpy array to do calculations
        volatility = np.std(np.array(close))
        
        # Format
        volatility = f"{volatility:.2f}"
        
        # Add metrics to dict
        metrics_dict['Index'].append(ticker)
        metrics_dict['3m Return'].append(returns)
        metrics_dict['3m Volatility'].append(volatility)
        metrics_dict['Current Price'].append(f"${close.iloc[-1]:.2f}") # Notice here we get the data, format, and append all in line. Sometimes that is faster.
    
    # Convert dict to pandas dataframe
    key_metrics = pd.DataFrame(metrics_dict)
    
    # Create the table
    table = ax.table(cellText=key_metrics.values, colLabels=key_metrics.columns, cellLoc='center', loc='center')
    
    # Format table
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    # Ensure nothing overlaps
    plt.tight_layout()
    
    # Finally, show the table
    plt.show()


if __name__ == "__main__":
    """ Notice that this is not a function, but is executed
    when the file is run. This is standard practice to check
    if the file is the main file being ran, and if so, run 
    the following code. This is particularly useful if you
    are making a big project with multiple imports and you
    want to check if something in a particular file works
    properly without running all of the dependencies. """
    
    # Define tickers for indices
    tickers = ['^GSPC', '^IXIC', '^DJI']  # S&P 500, Nasdaq, Dow Jones

    # Get the data from yfinance
    data = get_data(tickers)
    
    # Graph the data
    graph_data(data)
    
    # Graph a table with some key metrics
    graph_table(data)