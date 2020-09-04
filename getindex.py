import numpy as np
import pandas as pd
from yahooquery import Ticker
import pickle

exchange = 'snp'
data_format = 'PRICE5'

with open('{}_train/{}_{}_variable_dump.pickle'.format(exchange, exchange, data_format), 'rb') as pickled:
    variable_dump = pickle.load(pickled)
    num_companies = variable_dump['num_companies']
    num_days = variable_dump['num_days']
    num_features = variable_dump['num_features']
    window_size = variable_dump['window_size']
    window_stride = variable_dump['window_stride']
    num_windows = variable_dump['num_windows']
    history_size = variable_dump['history_size']
    target_size = variable_dump['target_size']

column_titles = ['open', 'high', 'low', 'adjclose']
#init = Ticker(('^NYA' if exchange == 'nyse' else ('^GSPC' if exchange == 'snp' else '^NDX')))
#data = init.history(start='2013-01-02', end='2020-08-22')

with open('{}_train/{}_index_raw.pickle'.format(exchange, exchange), 'rb') as pickled:
    data = pickle.load(pickled)

index = data[column_titles].ewm(alpha=0.2, axis=0, adjust=False).mean()[10:].to_numpy()[-num_days:, :]
index_array = np.zeros((num_companies, num_windows, history_size, num_features))

std_index_window = np.zeros((history_size, num_features))
index_windows = []
# Stack standardized windows to form training data from index values
for w in range(num_windows):
    average = np.mean(index[w*window_stride: w*window_stride + history_size, :4], axis=-1, keepdims=True)
    index_price_avg = np.mean(index[w*window_stride: w*window_stride + history_size, :4], axis=None)
    index_price_std = np.std(index[w*window_stride: w*window_stride + history_size, :4], axis=None)
    #index_vol_avg = np.mean(index[w*window_stride: w*window_stride + history_size, 4], axis=None)
    #index_vol_std = np.std(index[w*window_stride: w*window_stride + history_size, 4], axis=None)
    std_index_window[:, :] = np.divide(average - index_price_avg, index_price_std)
    #std_index_window[:, 4] = np.divide(index[w*window_stride: w*window_stride + history_size, 4] - index_vol_avg, index_vol_std)
    index_windows.append(std_index_window)
    index_array[:, w, :, :] = std_index_window

print(index_array.shape)
np.savez('{}_train/{}_{}_index.npz'.format(exchange, exchange, data_format), index=index_array)
