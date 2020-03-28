import json
import datetime
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from sklearn import covariance, cluster


def load_json_from_file(json_filepath):
    """
    :param json_filepath:
    :return: labels, dict type
    """
    with open(json_file, 'r', encoding='utf8') as jf:
        labels = json.load(jf)
    return labels


def delete_stock_symbol_from_array(stock_symbol, stock_symbols, names):
    """Delete the stock symbol from np.array and return.

    :param stock_symbol: such as "MSFT"
    :param stock_symbols: np.array, ["MSFT", "GOOG", "EBAY", ...]
    :param names:  np.array, ["微软", "谷歌", "eBay", ...]
    :return: stock_symbols, names
    """
    index = np.argwhere(stock_symbols == stock_symbol)
    stock_symbols = np.delete(stock_symbols, index)
    names = np.delete(names, index)
    return stock_symbols, names


def get_quotes(stock_symbols, names, startDate, endDate):
    """Get the data from web using "pdr.get_data_yahoo".\n
    Delete the stock symbol at the same symbol index \n
    from the (stock_symbols, names) if an error happens.

    :param stock_symbols: np.array, ["MSFT", "GOOG", "EBAY", ...]
    :param names: np.array, ["微软", "谷歌", "eBay", ...]
    :param startDate: Datetime type
    :param endDate:  Datetime type
    :return: {quotes, list of DataFrame}, stock_symbols, names
    """
    quotes = []
    for stock_symbol in stock_symbols:
        print(stock_symbol)
        try:
            quote = pdr.get_data_yahoo(stock_symbol, startDate, endDate)
            # print(quote)
            quotes.append(quote)
        except Exception:
            print(f"{stock_symbol} is deleted!")
            stock_symbols, names = delete_stocks_ymbol_from_array(stock_symbol, stock_symbols, names)
    return quotes, stock_symbols, names


if __name__ == "__main__":
    # load json file of symbol mapping
    json_file = r"stock_clustering/symbol_map.json"
    labels = load_json_from_file(json_file)
    print(labels.keys())
    # choose one label to analyse
    symbols = labels['117家在美上市科技类知名公司']
    # get the stock symbols and names of some label
    # It takes too long to load all the stock data,
    # so I set it to 10, you can change the number as you like.
    # if delete the "[0:10]", it loads all the data at default.
    stock_symbols, names = np.array(list(symbols.items())[0:10]).T
    # print(stock_symbols, names)

    # convert the time to Datetime
    startDate = datetime.datetime(2019, 1, 1)
    endDate = datetime.datetime(2020, 3, 27)
    # get the stock price data from the web
    print("Loading data....")
    quotes, stock_symbols, names = get_quotes(stock_symbols, names, startDate, endDate)

    df = pd.concat(quotes, axis=0, keys=stock_symbols)
    df.to_csv(r"stock_clustering/117家在美上市科技类知名公司.csv")
    print("Data Saved to CSVFile!")

    # cluster the quotes
    opening_quotes = np.array([quote.Open for quote in quotes]).astype(np.float)
    closing_quotes = np.array([quote.Close for quote in quotes]).astype(np.float)
    print(opening_quotes.shape, closing_quotes.shape)
    # calculate the fluctuation of stock price（closing_quotes - opening_quotes）
    delta_quotes = closing_quotes - opening_quotes
    print(delta_quotes.shape)
    # biuld the covariance model
    edge_model = covariance.GraphicalLassoCV()
    # standardize the data
    X = delta_quotes.copy().T
    X /= X.std(axis=0)
    print(X.shape)
    # train the model
    with np.errstate(invalid='ignore'):
        edge_model.fit(X)
    # build cluster model
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    num_labels = labels.max()
    # print the cluster results
    for i in range(num_labels + 1):
        print(f"Cluster{i + 1}\t-->\t{', '.join(names[labels == i])}")
