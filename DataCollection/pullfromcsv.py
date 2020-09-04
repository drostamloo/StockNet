import pandas as pd
import numpy as np
from yahooquery import Ticker
import pickle

exchange = 'snp'
data_format = 'PRICE5'  # Either 'PRICE', 'PRICE5', or 'PRICE10'

constituents = pd.read_csv('{}_train/{}_constituents.csv'.format(exchange, exchange))
tickers = constituents['Symbol'].tolist()
column_titles = ['open', 'high', 'low', 'adjclose']

#init = Ticker(tickers)
#data = init.history(start='2013-01-02', end='2020-08-22')

with open('{}_train/{}_raw.pickle'.format(exchange, exchange), 'rb') as pickled:
    data = pickle.load(pickled)

# Convert dataframe to dict to process each company individually
if type(data) == type(constituents):
    data = {k: data.loc[k] for k in tickers}

# Need to create dataframe if yahooquery returns a dict and ensure all entries are dataframes,
# are continually traded since 2015, and traded actively enough to not throw "divide by zero" error when standardizing.

print('{} total companies in {}.'.format(len(data), exchange))
num_days = data[('JPM' if exchange == 'nyse' else 'GOOG')].shape[0]
print('{} total trading days.'.format(num_days))
data = {k: v[column_titles].ewm(alpha=0.2, axis=0, adjust=False).mean()[10:]
        for k, v in data.items() if (type(v) == type(constituents)
                                     and str(v.index.array[0])[:10] == '2013-01-02'
                                     and not np.any(pd.isnull(v[column_titles]))
                                     and np.count_nonzero(v['volume'].to_numpy() == 0) < (480 if exchange == 'nyse' else 270)
                                     and v.shape[0] == num_days)
        }
for k, v in data.items():
    data[k] = pd.concat([v], keys=[k], names=['symbol'])

df = pd.concat([v for k, v in data.items()], axis=0)
print('Data sift has produced {} total companies from {} '
      'for consideration.'.format(len(data), exchange))

df.to_csv('{}_train/{}_{}_data.csv'.format(exchange, exchange, data_format))
