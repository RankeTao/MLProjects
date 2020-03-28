import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import json
# set the url and get the content
url = 'http://vip.stock.finance.sina.com.cn/usstock/ustotal.php'
page = requests.get(url)
type(page.content)

# make content look better in order
soup = BeautifulSoup(page.content, 'html.parser')
print(soup.prettify())

# find all the stock collection labels
symbol_labels = soup.findAll('label')
# get the label text and truncate it
labels = [symbol_label.text.strip(':') for symbol_label in symbol_labels]
print(labels)

# initialize a dict to store the stock symbols and name  belonging to each label
data = dict(zip(labels, [[] for i in range(len(labels))]))
print(data[labels[0]])
# get all the label's content
col_divs = soup.findAll('div',{'class':'col_div'})

for i, label in enumerate(labels):
    for item in col_divs[i].findAll('a'):
        text = item.text
        t_ls = [i.rstrip(')') for i in text.split('(')]
        data[label].append((t_ls[1], t_ls[0]))
    data[label] = dict(data[label])
print(data)

# dump the data to json file
json_filepath = r"stock_clustering/symbol_map.json"
json.dump(data, open(json_filepath, "w", encoding="utf-8"), ensure_ascii=False)

print("DONE!")