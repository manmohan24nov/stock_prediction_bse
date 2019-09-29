import pandas as pd
import matplotlib.pyplot as plt
# print(train.isnull().sum(axis=0))
# print(test.isnull().sum(axis=0))
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
import datetime as dt
import matplotlib.dates as mdates
import time

colnames = ['stock_name','date','time','price','date_time']
stock_df_one = pd.read_csv('adani_power_nse.csv',names=colnames, header=None)
date_list = set(list(stock_df_one['date']))
print(date_list)
for i in date_list:

    if i.startswith('Jul'):
        stock_df = stock_df_one[stock_df_one['date']==i]
        stock_df['date_name'] = stock_df['date'].replace(to_replace=i,value='201906{0}'.format(i[-2:]))
        stock_df['date_time'] = stock_df['date_name'] + stock_df['time']
        stock_df['date_time'] = pd.to_datetime(stock_df['date_time'], format='%Y%m%d %H:%M:%S')
        stock_df = stock_df.drop_duplicates()

        print(stock_df)
        print(dt.datetime.now())

        print(date_list)
        # print(time.now())

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(stock_df['date_time'], stock_df['price'], color = 'black', linewidth = 0.4)
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M')
        # plt.plot(stock_df['time'],stock_df['price'])
        plt.show()