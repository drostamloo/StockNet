import numpy as np
import pandas as pd
import pickle

exchange = 'snp'
data_format = 'PRICE5'

df = pd.read_csv('{}_train/{}_{}_data.csv'.format(exchange, exchange, data_format), index_col=[0, 1])
constituents = pd.read_csv('{}_train/{}_constituents.csv'.format(exchange, exchange))
tickers = constituents['Symbol'].values.tolist()

num_companies = 0
num_days = df.loc[('JPM' if exchange == 'nyse' else 'GOOG')].shape[0]
num_features = 1

# Assigning important variables
window_size = 192
window_stride = 2
num_windows = int((num_days - window_size) / window_stride + 1)
num_days = (num_windows - 1) * window_stride + window_size
print('Data covers {} trading days with {} windows.'.format(num_days, num_windows))
history_size = 128
target_size = 64

data_list = []
for company in tickers:
    try:
        single_array = df.loc[company].to_numpy()
    except KeyError:
        continue
    data_list.append(single_array)
    num_companies += 1
data_array = np.stack(data_list, axis=0)
# Shape is (num_companies, num_days, num_features); row-vector format
data_array = data_array[:, -num_days:, :]

print('Data sift has produced {} total companies from {} for consideration.'.format(num_companies, exchange))

one_company_history_list = []
one_company_target_list = []
all_companies_history = []
all_companies_target = []
std_single_window = np.zeros((window_size, num_features))
# Parse all trading history into 192 day windows, 1 day stride between consecutive windows.
# First 128 days of window are the input sequence, last 64 days is forecast target.
# Normalize prices by standardization, normalize volumes by standardization
for c in range(num_companies):
    for w in range(num_windows):
        average = np.mean(data_array[c, w*window_stride: w*window_stride + window_size, :4], axis=-1, keepdims=True)
        price_avg = np.mean(data_array[c, w*window_stride: w*window_stride + history_size, :4], axis=None)
        price_std = np.std(data_array[c, w*window_stride: w*window_stride + history_size, :4], axis=None)
        """
        volume_avg = np.mean(data_array[c, w*window_stride: w*window_stride + history_size, 4], axis=None)
        volume_std = np.std(data_array[c, w*window_stride: w*window_stride + history_size, 4], axis=None)
        if volume_std == 0:
            print("Hey! Too many zeros (in the volume category), resulting in a std of 0 for some windows!")
        """
        std_single_window[:, :] = np.divide(average - price_avg, price_std)
        #std_single_window[:, 4] = np.divide(data_array[c, w*window_stride: w*window_stride + window_size, 4] - volume_avg, volume_std)
        one_company_history_list.append(std_single_window[:history_size, :])
        one_company_target_list.append(std_single_window[history_size:, :])
    one_company_history_array = np.stack(one_company_history_list, axis=0)
    one_company_target_array = np.stack(one_company_target_list, axis=0)
    all_companies_history.append(one_company_history_array)
    all_companies_target.append(one_company_target_array)
    one_company_history_list = []
    one_company_target_list = []

# Stack windows from all companies
history_array = np.stack(all_companies_history, axis=0)
target_array = np.pad(np.stack(all_companies_target, axis=0), ((0, 0), (0, 0), (1, 0), (0, 0)))
# Above for transformer decoder input
#target_array = np.stack(all_companies_target, axis=0)
print(history_array.shape)
print(target_array.shape)
np.savez('{}_train/{}_{}_history_target.npz'.format(exchange, exchange, data_format), history=history_array, target=target_array)

variable_dump = {'num_companies': num_companies, 'num_days': num_days, 'num_features': num_features,
                 'window_size': window_size, 'window_stride': window_stride, 'num_windows': num_windows,
                 'history_size': history_size, 'target_size': target_size}

with open('{}_train/{}_{}_variable_dump.pickle'.format(exchange, exchange, data_format), 'wb') as pickled:
    pickle.dump(variable_dump, pickled)
