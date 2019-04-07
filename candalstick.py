from pandas_datareader import data
import pandas as pd
import quandl
import matplotlib.pyplot as plt
from matplotlib import style
# import numpy as np
from mpl_finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY
style.use('grayscale')

quandl.ApiConfig.api_key = "9C6dwdEq5st3yzk_U7ZP"
# df_sample = quandl.get("NSE/idea", start_date='2019-01-01', end_date='2019-01-10')
df_sample = quandl.get("NSE/adanipower",start_date='2018-11-30', end_date='2019-01-10')
df_sample['Dates'] = df_sample.index
print(len(df_sample))
print(df_sample[['Open','Close']].head())
#
# #moving average
# moving_average21 = df_sample['Close'].rolling(window=21).mean()
# moving_average50 = df_sample['Close'].rolling(window=50).mean()
# moving_average200 = df_sample['Close'].rolling(window=200).mean()
#
#
#
#
# # Create a new column of numerical "date" values for matplotlib to use
# df_sample['date_ax'] = df_sample['Dates'].apply(lambda date: date2num(date))
# ford_values = [tuple(vals) for vals in df_sample[['date_ax', 'Open', 'High', 'Low', 'Close']].values]
#
# mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
# alldays = DayLocator()              # minor ticks on the days
# weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
# dayFormatter = DateFormatter('%d')      # e.g., 12
#
# #Plot it
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.2)
# ax.xaxis.set_major_locator(mondays)
# ax.xaxis.set_minor_locator(alldays)
# ax.xaxis.set_major_formatter(weekFormatter)
# ax.plot(moving_average21.index, moving_average21, label='21 days rolling')
# ax.plot(moving_average50.index, moving_average50, label='50 days rolling')
# ax.plot(moving_average200.index, moving_average200, label='200 days rolling')
# candlestick_ohlc(ax, ford_values, width=0.6, colorup='y',colordown='k')
# ax.xaxis_date()
# ax.autoscale_view()
# plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
# plt.show()
