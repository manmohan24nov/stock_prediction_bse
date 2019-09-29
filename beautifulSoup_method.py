import requests
from bs4 import BeautifulSoup
import time
import sys


list_of_stocks = ['https://www.moneycontrol.com/stock-charts/unionbankindia/charts/UBI01#UBI01',
                  'https://www.moneycontrol.com/stock-charts/punjabnationalbank/charts/PNB05#PNB05',
                  'https://www.moneycontrol.com/stock-charts/adanipower/charts/AP11#AP11',
                  'https://www.moneycontrol.com/stock-charts/steelauthorityindia/charts/SAI#SAI']
stocks_name = ['union_bank','punjab_national_bank','adani_power','sail']

def save_data(platform_name,stock,time_of_trading,price):
    file_name = stock + '_' + platform_name +'.csv'
    with open(file_name, 'a') as file_line:
        stock_to_append_data = stock+','+time_of_trading + ',' + price+'\n'
        file_line.write(stock_to_append_data)
    file_line.close()


for time_in_sec in range(400):

    for i,j in enumerate(list_of_stocks):
        try :
            page = requests.get(j)
            soup = BeautifulSoup(page.text, 'html.parser')
            # print(soup.find_all(id='Bse_Prc_tick'))
            # print(soup.find_all(id='bse_upd_time'))
            # print(soup.find_all(id='Nse_Prc_tick'))
            # print(soup.find_all(id='nse_upd_time'))
            print(stocks_name[i],soup.find(id='bse_upd_time').getText(),soup.find(id='Bse_Prc_tick').getText())
            print(stocks_name[i],soup.find(id='nse_upd_time').getText(),soup.find(id='Nse_Prc_tick').getText())
            save_data('bse',stocks_name[i],soup.find(id='bse_upd_time').getText(),soup.find(id='Bse_Prc_tick').getText())
            save_data('nse', stocks_name[i], soup.find(id='nse_upd_time').getText(),
                      soup.find(id='Nse_Prc_tick').getText())

            if str(soup.find(id='bse_upd_time').getText()[-5:-3])=='16':
                sys.exit()

        except Exception:
            continue
    time.sleep(240)

