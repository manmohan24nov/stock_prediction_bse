import pandas as pd
import matplotlib.pyplot as plt
# print(train.isnull().sum(axis=0))
# print(test.isnull().sum(axis=0))
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)
import datetime as dt
import time
import matplotlib.dates as mdates

colnames = ['stock_name','date','time','open','high','low','close','volume']
stock_df = pd.read_csv('SAIL.txt',names=colnames, header=None)
stock_df['date'] = stock_df['date'].astype('str')
stock_df['date_time'] = stock_df['date'] +' '+ stock_df['time']
stock_df['date_time']  = pd.to_datetime(stock_df['date_time'], format='%Y%m%d %H:%M:%S')
print(stock_df)
# stock_df['date_time'] = stock_df['date'] + stock_df['time']
# stock_df = stock_df.drop_duplicates()
date_list = set(list(stock_df['date']))
print(date_list)
for i in date_list:

    stock_df_temp = stock_df[stock_df['date']==i]
    print(i)
    print(stock_df_temp)
    # print(stock_df)
    # print(dt.datetime.now())
    # print(time.now())

    # plotting with ticks
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(stock_df_temp['date_time'], stock_df_temp['open'], color = 'black', linewidth = 0.4)
    # Then tick and format with matplotlib:
    # ax = plt.gca()

    # df = df.reset_index()
    # df = df.rename(columns={"index":"hour"})
    # ax = df.plot(xticks=df.index)
    # ax.set_xticklabels(stock_df["date_time"])

    # plt.setp(ax.get_xticklabels(), rotation=45)
    # ax.xaxis.set_major_locator(hours)
    # ax.xaxis.set_major_formatter(h_fmt)

    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M')

    # plt.plot(stock_df['time'],stock_df['open'])
    plt.show()