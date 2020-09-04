import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from yahooquery import Ticker
import matplotlib.pyplot as plt
import mplfinance as fplt

import datetime
import os
exchange = 'snp'

source_end = datetime.date.today()
source_start = source_end - datetime.timedelta(days=190)
source_end_str = source_end.strftime('%Y-%m-%d')
source_start_str = source_start.strftime('%Y-%m-%d')

pred_start = source_end + datetime.timedelta(days=1)
pred_start_str = pred_start.strftime('%Y-%m-%d')
"""
company = None
column_titles = ['open', 'adjclose', 'high', 'low', 'volume']

if company in pd.read_csv('s&p_data/s&p_constituents.csv')['Symbol'].array:
    exchange = 's&p'
elif company in pd.read_csv('nasdaq_data/nasdaq_constituents.csv')['Symbol'].array:
    exchange = 'nasdaq'
elif company in pd.read_csv('nyse_data/nyse_constituents.csv')['Symbol'].array:
    exchange = 'nyse'

constituents = pd.read_csv('{}_data/{}_constituents.csv'.format(exchange, exchange))
tickers = constituents['Symbol'].tolist()

if not os.path.isfile('{}_data/{}.npz'.format(exchange, source_end_str)):
    os.system('rm -rf {}_data/*.npz'.format(exchange))
    init = Ticker(tickers)
    data = init.history(start=source_start_str)

    if type(data) == type(constituents):
        data = {k: data.loc[k] for k in tickers}

    # Need to create dataframe if yahooquery returns a dict and ensure all entries are dataframes,
    # are continually traded since 2015, and traded actively enough to not throw "divide by zero" error when standardizing.

    print('{} total companies in {}.'.format(len(data), exchange))
    num_days = data[('JPM' if exchange == 'nyse' else 'GOOG')].shape[0]
    data = {k: v[column_titles].ewm(alpha=0.9, axis=0).mean()[-120:0].to_numpy()
            for k, v in data.items() if (type(v) == type(constituents)
                                         and str(v.index.array[0])[:10] == source_start_str
                                         and not np.any(pd.isnull(v[column_titles]))
                                         and np.count_nonzero(v['volume'].to_numpy() == 0) < 120
                                         and v.shape[0] == num_days)
            }

    print('Data sift has produced {} total companies from {} '
          'for consideration.'.format(len(data), exchange))

    all_companies_history = []
    std_single_window = np.zeros((120, 5))
    for k, v in data.items():
        data_price_avg = np.mean(v[:, :4], axis=None)
        data_price_std = np.std(v[:, :4], axis=None)
        data_vol_avg = np.mean(v[:, 4], axis=None)
        data_vol_std = np.std(v[:, 4], axis=None)

        std_single_window[:, :4] = np.divide(v[:, :4] - data_price_avg, data_price_std)
        std_single_window[:, 4] = np.divide(v[:, 4] - data_vol_avg, data_vol_std)
        all_companies_history.append(std_single_window)

    history = np.stack(all_companies_history, axis=0)

    index_init = Ticker(('^NYA' if exchange == 'nyse' else ('^GSPC' if exchange == 's&p' else '^NDX')))
    index_data = index_init.history(start=source_start_str)
    index = index_data[column_titles].ewm(alpha=0.9, axis=0).mean()[-120:].to_numpy()

    index_price_avg = np.mean(index[:, :4], axis=None)
    index_price_std = np.mean(index[:, :4], axis=None)
    index_vol_avg = np.mean(index[:, 4], axis=None)
    index_vol_std = np.mean(index[:, 4], axis=None)

    index[:, :4] = np.divide(index[:, :4] - index_price_avg, index_price_std)
    index[:, 4] = np.divide(index[:, 4] - index_vol_avg, index_vol_std)

    np.savez('{}_data/{}.npz'.format(exchange, source_end_str),
             history=history, index=index)
else:
    with np.load('{}_data/{}.npz'.format(exchange, source_end_str)) as data:
        history = data['history'].astype(np.float32)
        index = data['index'].astype(np.float32)

source_init = Ticker(company)
source_data = source_init.history(start=source_start_str)
source = source_data[column_titles].ewm(alpha=0.9, axis=0).mean()[-120:].to_numpy()

source_price_avg = np.mean(source[:, :4], axis=None)
source_price_std = np.mean(source[:, :4], axis=None)
source_vol_avg = np.mean(source[:, 4], axis=None)
source_vol_std = np.mean(source[:, 4], axis=None)

source[:, :4] = np.divide(source[:, :4] - source_price_avg, source_price_std)
source[:, 4] = np.divide(source[:, 4] - source_vol_avg, source_vol_std)


def get_positive_and_negative(source, data):
    positive = np.zeros((120, 5))
    negative = np.zeros((120, 5))

    correlations = np.corrcoef(np.vstack([source[:, 1], data[:, 1]]))
    idx_positives = np.argpartition(correlations[0, :], -21)[-21:-1]
    idx_negatives = np.argpartition(correlations[0, :], 20)[:20]
    positive[:, :] = np.mean(data[idx_positives, :, :], axis=0)
    negative[:, :] = np.mean(data[idx_negatives, :, :], axis=0)

    return positive, negative


positive, negative = get_positive_and_negative(source, history)

history = history.astype(np.float32)
positive = positive.astype(np.float32)
negative = negative.astype(np.float32)
index = index.astype(np.float32)
"""
# LOAD DATA
with np.load('{}_train/{}_input_ready.npz'.format(exchange, exchange)) as data:
    x_source = data['x_source'].astype(np.float32)
    x_positive = data['x_positive'].astype(np.float32)
    x_negative = data['x_negative'].astype(np.float32)
    x_index = data['x_index'].astype(np.float32)
    x_target = data['x_target'].astype(np.float32)


# Learning rate scheduler
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embed_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.embed_dim = embed_dim
        # JSON cannot serialize a float32, must create a new variable to operate on embed_dim
        self.d_model = tf.cast(embed_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * self.warmup_steps ** -1.5

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = dict(embed_dim=self.embed_dim,
                      warmup_steps=self.warmup_steps)
        return config


model = keras.models.load_model('{}_keras_model/'.format(exchange), custom_objects={'CustomSchedule': CustomSchedule})


def evaluate(source):
    print(source)
    source = tf.expand_dims(source, 0)
    outputs = []
    for i in range(60):

        # predictions.shape == (batch_size (== 1), seq_len, num_features (== 5))
        prediction = model.predict(source)
        outputs.append(prediction)
        # select the last output from the seq_len dimension
        source = np.concatenate([source[:, 1:, :], prediction[None, :, :]], axis=1)  # (batch_size, 1, num_features)

        # concatenate the prediction to the output which is given to the decoder
        # as its input.

    return tf.stack(outputs, axis=0)


def plot_candlestick(source, predicted, true):
    date_range = pd.date_range(end=source_end, periods=120, freq='B').union(
        pd.date_range(start=pred_start, periods=60, freq='B'))

    data = pd.DataFrame(data=np.concatenate([source, predicted], axis=0),
                        index=date_range,
                        columns=['open', 'close', 'high', 'low', 'volume'])
    true = pd.DataFrame(data=np.concatenate([source, true], axis=0),
                        index=date_range,
                        columns=['open', 'close', 'high', 'low', 'volume'])

    true_plt = fplt.make_addplot(true[['close']], type='line')
    fplt.plot(data,
              type='line',
              vlines=dict(vlines=source_end, linestyle='-.'),
              addplot=true_plt
              #style='charles',
              #title='test',
              #ylabel='Price ($)',
              #volume=True,
              #ylabel_lower='Shares\nTraded',
              )


def predict(source):
    result = evaluate(source)
    print(result)
    plot_candlestick(source, result)


predict(x_source[4, :, :])
