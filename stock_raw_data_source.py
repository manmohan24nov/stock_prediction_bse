# this contains raw data of stocks
# unchanged
import pandas as pd


class stock_data(object):

    def __init__(self,stock_name):
        self.stock_name = stock_name

    def stock_data_func(self):
        raw_data = pd.read_csv("""E:\studymaterial\stock_project\stock_raw_data/{0}.csv""".format(self.stock_name), sep=',', header=0)
        raw_data['new_date'] = pd.to_datetime(raw_data['Date'], format='%d-%B-%Y')
        raw_data.index = raw_data['new_date']
        stock_data = raw_data.sort_index()
        return stock_data
