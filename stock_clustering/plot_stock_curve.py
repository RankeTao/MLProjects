import datetime
import pandas as pd
import numpy as np
import mplfinance as mpf
import pandas_datareader as pdr

# read data from file and plot stock candle curve
data_file = r"stock_clustering/117家在美上市科技类知名公司.csv"
stock_data = pd.read_csv(data_file)
# rename the column with "Symbol"
stock_data = stock_data.rename(columns={'Unnamed: 0': 'Symbol'})
# filter the data with symbol
symbol_data = stock_data[stock_data.Symbol == '微软']
# print(symbol_data)
# drop the symbol column
symbol_data = symbol_data.drop(['Symbol'], axis=1)
# convert the "Date" column to DateTimeIndex type
symbol_data['Date'] = pd.to_datetime(symbol_data['Date'], format = '%Y-%m-%d')
# Set the "Date" column as index
symbol_data = symbol_data.set_index('Date')
# plot the curve using "mplfinance"
mpf.plot(symbol_data, type='candle')

# get data from web and plot the curve
startDate = datetime.datetime(2019, 1, 1)
endDate = datetime.datetime(2020, 3, 27)
quote = pdr.get_data_yahoo('GOOG', startDate, endDate)
print(quote)
mpf.plot(quote, type='candle')
