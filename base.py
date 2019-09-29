from selenium import webdriver
import os
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

os.environ["PATH"] += os.pathsep + '/home/manmohan/projects_on_python/stock_mining/'

from selenium import webdriver
driver = webdriver.Firefox()
driver.get('https://www.moneycontrol.com/india/stockpricequote/banks-public-sector/punjabnationalbank/PNB05')
# element = driver.find_element_by_id('search_str')
# element.send_keys('punjab national bank')
# python_button = driver.find_elements_by_xpath("//a[@class='btn_black btn_search FR' and @title='Submit']")[0]
# python_button.click()
# driver.switch_to_alert().accept()
soup=BeautifulSoup(driver.page_source)
print(soup)
#do something useful
#prints all the links with corresponding text

for link in soup.find_all('a'):
    print(link.get('href',None),link.get_text())
# time.sleep(20)
price = driver.find_elements_by_xpath("//span[@id='Nse_prc_tick' and @class='PA2']")
print(price)
print(driver.find_elements_by_xpath("//strong"))

# price = driver.find_elements_by_xpath("//strong")
# print(price.text)