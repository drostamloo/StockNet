import tensorflow as tf
import numpy as np
import pandas as pd
from yahooquery import Ticker
import matplotlib.pyplot as plt
from tkinter import *

import datetime
import os

exchange = 'snp'
input_method = 'SINGLE'
data_format = 'PRICE'

source_end = datetime.date.today()
source_start = source_end - datetime.timedelta(days=225)
source_end_str = source_end.strftime('%Y-%m-%d')
source_start_str = source_start.strftime('%Y-%m-%d')

pred_start = source_end + datetime.timedelta(days=1)
pred_start_str = pred_start.strftime('%Y-%m-%d')


class Prompt(Tk):
    def __init__(self):
        Tk.__init__(self)
        input_var = 0
        self.title('StockNet')
        img = Image('photo', file='chart_candlestick.gif')
        self.iconphoto(True, img)
        self.geometry('300x150')
        self.label = Label(self, text='TICKER (ALL CAPS):')
        self.entry = Entry(self, textvariable=input_var)
        self.button = Button(self, text='Enter', command=self.on_button)
        self.label.pack(pady=15)
        self.entry.pack()
        self.button.pack(pady=15)

    def on_button(self):
        self.answer = self.entry.get()
        self.quit()


prompt = Prompt()
prompt.mainloop()
prompt.destroy()
company = prompt.answer

column_titles = ['open', 'adjclose', 'high', 'low']

if company in pd.read_csv('snp_data/snp_constituents.csv')['Symbol'].array:
    exchange = 'snp'
elif company in pd.read_csv('nasdaq_data/nasdaq_constituents.csv')['Symbol'].array:
    exchange = 'nasdaq'
elif company in pd.read_csv('nyse_data/nyse_constituents.csv')['Symbol'].array:
    exchange = 'nyse'

constituents = pd.read_csv('{}_data/{}_constituents.csv'.format(exchange, exchange))
tickers = constituents['Symbol'].tolist()


def fetch_all_data():
    os.system('rm -rf {}_data/*.npz'.format(exchange))
    init = Ticker(tickers)
    data = init.history(start=source_start_str)

    if type(data) == type(constituents):
        data = {k: data.loc[k] for k in tickers}

    # Need to create dataframe if yahooquery returns a dict and ensure all entries are dataframes,
    # are continually traded since 2015, and traded actively enough to not throw "divide by zero" error when standardizing.

    print('{} total companies in {}.'.format(len(data), exchange))
    num_days = data[('JPM' if exchange == 'nyse' else 'GOOG')].shape[0]
    data = {k: v[column_titles].ewm(alpha=1, axis=0, adjust=False).mean().to_numpy()[-128:, :]
            for k, v in data.items() if (type(v) == type(constituents)
                                         #and str(v.index.array[0])[:10] == source_start_str
                                         and not np.any(pd.isnull(v[column_titles]))
                                         and v.shape[0] == num_days)
            }

    print('Data sift has produced {} total companies from {} '
          'for consideration.'.format(len(data), exchange))

    all_companies_history = []
    std_single_window = np.zeros((128, 1))

    for k, v in data.items():
        average = np.mean(v, axis=-1, keepdims=True)
        data_price_avg = np.mean(v[:, :4], axis=None)
        data_price_std = np.std(v[:, :4], axis=None)

        std_single_window[:, :] = np.divide(average - data_price_avg, data_price_std)
        all_companies_history.append(std_single_window)

    history = np.stack(all_companies_history, axis=0)

    index_init = Ticker(('^NYA' if exchange == 'nyse' else ('^GSPC' if exchange == 'snp' else '^NDX')))
    index_data = index_init.history(start=source_start_str)
    index = index_data[column_titles].ewm(alpha=1, axis=0).mean()[-128:].to_numpy()

    index_average = np.mean(index, axis=-1, keepdims=True)
    index_price_avg = np.mean(index[:, :4], axis=None)
    index_price_std = np.mean(index[:, :4], axis=None)

    index = np.divide(index_average - index_price_avg, index_price_std)

    np.savez('{}_data/{}.npz'.format(exchange, source_end_str),
             history=history, index=index)

    # Destroy loading window
    loading.destroy()


if not os.path.isfile('{}_data/{}.npz'.format(exchange, source_end_str)):
    loading = Tk()
    loading.geometry('400x100')
    loading.title('Please wait')
    label = Label(loading, text='Up-to-date data not found for {}, fetching new data.'.format(exchange))
    label.pack(pady=20)
    loading.after(250, fetch_all_data)
    loading.mainloop()

with np.load('{}_data/{}.npz'.format(exchange, source_end_str)) as arrays:
    history = arrays['history'].astype(np.float32)
    index = arrays['index'].astype(np.float32)

source_init = Ticker(company)
source_data = source_init.history(start=source_start_str)
source = source_data[column_titles].ewm(alpha=1, axis=0).mean()[-128:].to_numpy()

source_average = np.mean(source, axis=-1, keepdims=True)
source_price_avg = np.mean(source[:, :4], axis=None)
source_price_std = np.std(source[:, :4], axis=None)

source = np.divide(source_average - source_price_avg, source_price_std).astype(np.float32)


def get_positive_and_negative(source, data):
    positives = np.zeros((128, 1))
    negatives = np.zeros((128, 1))

    correlations = np.corrcoef(np.vstack([source[:, 0], data[:, :, 0]]))
    idx_positives = np.argpartition(correlations[0, :], -21)[-21:-1]
    idx_negatives = np.argpartition(correlations[0, :], 20)[:20]
    positives[:, :] = np.mean(data[idx_positives, :, :], axis=0)
    negatives[:, :] = np.mean(data[idx_negatives, :, :], axis=0)

    return positives, negatives


positive, negative = get_positive_and_negative(source, history)

source = source.astype(np.float32)
positive = positive.astype(np.float32)
negative = negative.astype(np.float32)
index = index.astype(np.float32)

# LOAD DATA
with np.load('{}_train/{}_{}_input_ready.npz'.format(exchange, exchange, data_format)) as data:
    x_source = data['x_source'].astype(np.float32)
    x_positive = data['x_positive'].astype(np.float32)
    x_negative = data['x_negative'].astype(np.float32)
    x_index = data['x_index'].astype(np.float32)
    x_target = data['x_target'].astype(np.float32)
    dev_source = data['dev_source'].astype(np.float32)
    dev_positive = data['dev_positive'].astype(np.float32)
    dev_negative = data['dev_negative'].astype(np.float32)
    dev_index = data['dev_index'].astype(np.float32)
    dev_target = data['dev_target'].astype(np.float32)


# GET POSITIONAL ENCODINGS
def get_angles(pos, i, d_timevec):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_timevec))
    return pos * angle_rates


def positional_encoding(position, d_timevec):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_timevec)[np.newaxis, :], d_timevec)

    pos_encoding = np.zeros((1, position, d_timevec))
    pos_encoding[:, :, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, :, 1::2] = np.sin(angle_rads[:, 1::2])

    return tf.cast(pos_encoding, dtype=tf.float32)


# CREATE LOOK-AHEAD MASK
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


# ATTENTION AND FEED-FORWARD
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add mask to scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax is normalized on the last axis (seq_len_k) to produce row-vector scores sum of 1.
    # Attention weights matrix used to weight value row_vectors and sum to produce attention output.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """ Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # ...
        v = self.wv(v)  # ...

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, d_model)
        k = self.split_heads(k, batch_size)  # ...
        v = self.split_heads(v, batch_size)  # ...

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# ENCODER AND DECODER LAYERS
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x, None)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class SingleInputDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(SingleInputDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask):
        attn_weights = []

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn_weights.append(attn_weights_block1)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, None)  # (batch_size, target_seq_len, d_model)
        attn_weights.append(attn_weights_block2)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights


class SerialDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(SerialDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha3 = MultiHeadAttention(d_model, num_heads)
        #self.mha4 = MultiHeadAttention(d_model, num_heads)
        #self.mha5 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm5 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm6 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)
        #self.dropout5 = tf.keras.layers.Dropout(rate)
        #self.dropout6 = tf.keras.layers.Dropout(rate)

        self.Cross_MultiHeadAttention = [self.mha2, self.mha3]
        self.Cross_Dropout = [self.dropout2, self.dropout3]
        self.Cross_LayerNorm = [self.layernorm2, self.layernorm3]

    def call(self, x, enc_output, training, look_ahead_mask):
        attn_weights = []

        self_attn, self_attn_weights = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn_weights.append(self_attn_weights)
        self_attn = self.dropout1(self_attn, training=training)
        self_attn = self.layernorm1(self_attn + x)
        previous = self_attn

        for count, i in enumerate(enc_output):
            cross_attn, cross_attn_weights = self.Cross_MultiHeadAttention[count](i, i, previous, None)
            attn_weights.append(cross_attn_weights)
            cross_attn = self.Cross_Dropout[count](cross_attn, training=training)
            cross_attn = self.Cross_LayerNorm[count](cross_attn + previous)
            previous = cross_attn
            # cross_attn.shape == (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(cross_attn)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout4(ffn_output)
        out = self.layernorm4(ffn_output + previous)

        return out, attn_weights


class ParallelDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(ParallelDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha3 = MultiHeadAttention(d_model, num_heads)
        # self.mha4 = MultiHeadAttention(d_model, num_heads)
        # self.mha5 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)
        # self.dropout5 = tf.keras.layers.Dropout(rate)
        # self.dropout6 = tf.keras.layers.Dropout(rate)

        self.Cross_MultiHeadAttention = [self.mha2, self.mha3]
        self.Cross_Dropout = [self.dropout2, self.dropout3]

    def call(self, x, enc_output, training, look_ahead_mask):
        attn_weights = []
        cross_attns = []

        self_attn, self_attn_weights = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn_weights.append(self_attn_weights)
        self_attn = self.dropout1(self_attn, training=training)
        self_attn = self.layernorm1(self_attn + x)

        for count, i in enumerate(enc_output):
            cross_attn, cross_attn_weights = self.Cross_MultiHeadAttention[count](i, i, self_attn, None)
            attn_weights.append(cross_attn_weights)
            cross_attn = self.Cross_Dropout[count](cross_attn, training=training)
            cross_attns.append(cross_attn)
            # cross_attn.shape == (batch_size, target_seq_len, d_model)

        parallel_out = self.layernorm2(tf.math.accumulate_n(cross_attns) + self_attn)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(parallel_out)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout4(ffn_output)
        out = self.layernorm3(ffn_output + parallel_out)

        return out, attn_weights


class FlatDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(FlatDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask):
        attn_weights = []

        concatenated = tf.concat(enc_output, axis=1)  # (batch_size, 2 * input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn_weights.append(attn_weights_block1)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            concatenated, concatenated, out1, None)  # (batch_size, target_seq_len, d_model)
        attn_weights.append(attn_weights_block2)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights


class HierarchicalDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(HierarchicalDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.mha3 = MultiHeadAttention(d_model, num_heads)
        #self.mha4 = MultiHeadAttention(d_model, num_heads)
        #self.mha5 = MultiHeadAttention(d_model, num_heads)

        self.concat_mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout4 = tf.keras.layers.Dropout(rate)
        #self.dropout5 = tf.keras.layers.Dropout(rate)
        #self.dropout6 = tf.keras.layers.Dropout(rate)

        self.concat_dropout = tf.keras.layers.Dropout(rate)

        self.Cross_MultiHeadAttention = [self.mha2, self.mha3]
        self.Cross_Dropout = [self.dropout2, self.dropout3]

    def call(self, x, enc_output, training, look_ahead_mask):
        attn_weights = []
        cross_attns = []

        self_attn, self_attn_weights = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn_weights.append(self_attn_weights)
        self_attn = self.dropout1(self_attn, training=training)
        self_out = self.layernorm1(self_attn + x)

        for count, i in enumerate(enc_output):
            cross_attn, cross_attn_weights = self.Cross_MultiHeadAttention[count](i, i, self_out, None)
            attn_weights.append(cross_attn_weights)
            cross_attn = self.Cross_Dropout[count](cross_attn, training=training)
            cross_attns.append(cross_attn)
            #  cross_attn.shape == (batch_size, target_seq_len, d_model)

        concatenated = tf.concat(cross_attns, axis=1)  # (batch_size, 2*target_seq_len, d_model)
        concat_attn, concat_attn_block = self.concat_mha(concatenated, concatenated, self_out, None)
        cross_attns.append(concat_attn_block)
        concat_attn = self.concat_dropout(concat_attn, training=training)
        cross_out = self.layernorm2(concat_attn + self_out)

        ffn_output = self.ffn(cross_out)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout4(ffn_output, training=training)
        out = self.layernorm3(ffn_output + cross_out)  # (batch_size, target_seq_len, d_model)

        return out, attn_weights


# FULL ENCODER AND DECODER
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, d_timevec, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_timevec)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        #self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]

        positions = tf.ones((tf.shape(x)[0], seq_len, tf.shape(self.pos_encoding)[2])) * self.pos_encoding[:, :seq_len, :]
        x = tf.concat([x, positions], axis=-1)

        #x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, d_timevec, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_timevec)

        self.dec_layers = [SingleInputDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        #self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, sources, training, look_ahead_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        positions = tf.ones((tf.shape(x)[0], seq_len, tf.shape(self.pos_encoding)[2])) * self.pos_encoding[:, :seq_len, :]
        x = tf.concat([x, positions[:, :seq_len, :]], axis=-1)

        #x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, blocks = self.dec_layers[i](x, sources, training, look_ahead_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = blocks[0]
            attention_weights['decoder_layer{}_block2'.format(i+1)] = blocks[1]

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# FULL TRANSFORMER, TRAINING, HYPERPARAMETERS
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, d_timevec, num_heads, dff, pe_input, pe_target,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, d_timevec, num_heads, dff, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, d_timevec, num_heads, dff, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, sources, tar, training, look_ahead_mask):
        # dec_output.shape == (batch_size, input_seq_len, d_model)
        enc_output = self.encoder(sources, training)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, 1)

        return final_output, attention_weights


# Hyperparameters
num_layers = 4
d_model = 64
d_timevec = d_model - 1
dff = 256
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(num_layers, d_model, d_timevec, num_heads, dff, pe_input=128,
                          pe_target=64, rate=dropout_rate)

latest = tf.train.latest_checkpoint('{}_{}3_{}_model/'.format(exchange, input_method, data_format))
transformer.load_weights(latest)


def evaluate(source):
    source = tf.expand_dims(source, 0)
    output = tf.zeros((1, 1, 1))

    for i in range(64):
        look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])

        # predictions.shape == (batch_size (== 1), seq_len, num_features (== 5))
        predictions, attention_weights = transformer(source, output, False, look_ahead_mask)

        # select the last output from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, num_features)

        # concatenate the prediction to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predictions], axis=1)

    return tf.squeeze(output, axis=0)


def plot_line_validate(source, predicted, true):
    past = pd.date_range(end=source_end, periods=128, freq='B')
    future = pd.date_range(start=pred_start, periods=64, freq='B')
    date_range = past.union(future)

    data = pd.DataFrame(data=np.concatenate([source, predicted], axis=0),
                        index=date_range,
                        columns=['Predicted'])
    true = pd.DataFrame(data=np.concatenate([source, true], axis=0),
                        index=date_range,
                        columns=['True'])

    ax = data.plot()
    fig = true.plot(ax=ax, title='Validate').get_figure()
    plt.axvline(past[-1], linestyle='--', color='k')
    fig.savefig('figure.png')
    plt.draw()


def plot_line_test(source, predicted, true):
    past = pd.date_range(end=source_end, periods=128, freq='B')
    future = pd.date_range(start=pred_start, periods=64, freq='B')
    date_range = past.union(future)

    data = pd.DataFrame(data=np.concatenate([source, predicted], axis=0),
                        index=date_range,
                        columns=['Predicted'])
    true = pd.DataFrame(data=np.concatenate([source, true], axis=0),
                        index=date_range,
                        columns=['True'])

    ax = data.plot()
    fig = true.plot(ax=ax, title='Test').get_figure()
    plt.axvline(past[-1], linestyle='--', color='k')
    fig.savefig('figure.png')
    plt.draw()


def plot_line(source, predicted):
    past = pd.date_range(end=source_end, periods=128, freq='B')
    future = pd.date_range(start=pred_start, periods=64, freq='B')
    date_range = past.union(future)

    data = pd.DataFrame(data=np.concatenate([source, predicted], axis=0),
                        index=date_range,
                        columns=['Predicted'])

    data = (data * source_price_std) + source_price_avg

    fig = data.plot(title=company).get_figure()
    plt.axvline(past[-1], linestyle='--', color='k')
    fig.savefig(os.path.expanduser('~') + '/Desktop/figure.png')
    plt.draw()


def predict_validate(source, true):
    result = evaluate(source)
    predicted = result[1:, :]
    true = true[1:, :]
    print(np.mean(np.abs(predicted - true)))
    plot_line_validate(source, predicted, true)


def predict_test(source, true):
    result = evaluate(source)
    predicted = result[1:, :]
    print('Normalized: {}'.format(np.mean(np.abs(result[1:, :] - true[1:, :]))))
    result = (result * test_price_std) + test_price_avg
    true = (true * test_price_std) + test_price_avg
    source = (source * test_price_std) + test_price_avg
    predicted = result[1:, :]
    true = true[1:, :]
    print('Dollar: {}'.format(np.mean(np.abs(predicted - true))))
    plot_line_test(source, predicted, true)


def predict(source):
    result = evaluate(source)
    predicted = result[1:, :]
    plot_line(source, predicted)


init = Ticker('AAPL')
test = init.history(start='2007-01-01', end='2008-08-22')
test = test[column_titles].ewm(alpha=1, axis=0).mean()[-192:].to_numpy().astype(np.float32)

test_avg = np.mean(test, axis=-1, keepdims=True)
test_price_avg = np.mean(test[:128, :4], axis=None)
test_price_std = np.std(test[:128, :4], axis=None)

test_source = np.divide(test_avg[:128] - test_price_avg, test_price_std)
test_target = np.pad(np.divide(test_avg[128:] - test_price_avg, test_price_std), ((1, 0), (0, 0)))

predict_validate(dev_source[1050, :, :], dev_target[1050, :, :])
predict_test(test_source, test_target)
predict(source)
plt.show()
