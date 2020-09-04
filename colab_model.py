import tensorflow as tf

import time
import numpy as np

from google.colab import auth

auth.authenticate_user()

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()  # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

exchange = 'snp'
input_method = 'SINGLE'
data_format = 'PRICE5'

# LOAD DATA
with np.load('/content/drive/My Drive/{}_{}_input_ready.npz'.format(exchange, data_format)) as data:
    x_source = data['x_source'].astype(np.float32)
    dev_source = data['dev_source'].astype(np.float32)
    test_source = data['test_source'].astype(np.float32)
    x_target = data['x_target'].astype(np.float32)
    dev_target = data['dev_target'].astype(np.float32)
    test_target = data['test_target'].astype(np.float32)
    x_positive = data['x_positive'].astype(np.float32)
    dev_positive = data['dev_positive'].astype(np.float32)
    test_positive = data['test_positive'].astype(np.float32)
    x_negative = data['x_negative'].astype(np.float32)
    dev_negative = data['dev_negative'].astype(np.float32)
    test_negative = data['test_negative'].astype(np.float32)
    x_index = data['x_index'].astype(np.float32)
    dev_index = data['dev_index'].astype(np.float32)
    test_index = data['test_index'].astype(np.float32)
    # Change input/output shapes from (num_examples, num_features, num_days) to (num_examples, num_days, num_features)
    # Changing to traditional row-vector format for transformers

train_data = tf.data.Dataset.from_tensor_slices((x_source, x_positive, x_negative, x_index, x_target))
dev_data = tf.data.Dataset.from_tensor_slices((dev_source, dev_positive, dev_negative, dev_index, dev_target))
test_data = tf.data.Dataset.from_tensor_slices((test_source, test_positive, test_negative, test_index, test_target))

BATCH_SIZE = 64 * strategy.num_replicas_in_sync

train_data = train_data.cache()
train_data = train_data.shuffle(x_source.shape[0]).batch(BATCH_SIZE, drop_remainder=True)
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
train_data = strategy.experimental_distribute_dataset(train_data)

dev_data = dev_data.batch(BATCH_SIZE, drop_remainder=True)
dev_data = strategy.experimental_distribute_dataset(dev_data)

test_data = test_data.batch(BATCH_SIZE, drop_remainder=True)
test_data = strategy.experimental_distribute_dataset(test_data)


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
            #attention_weights['decoder_layer{}_block3'.format(i+1)] = blocks[2]
            #attention_weights['decoder_layer{}_block4'.format(i+1)] = blocks[3]
            #attention_weights['decoder_layer{}_block5'.format(i+1)] = blocks[4]

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# FULL TRANSFORMER, TRAINING, HYPERPARAMETERS
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, d_timevec, num_heads, dff, pe_input, pe_target,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder1 = Encoder(num_layers, d_model, d_timevec, num_heads, dff, pe_input, rate)
        #self.encoder2 = Encoder(num_layers, d_model, d_timevec, num_heads, dff, pe_input, rate)
        #self.encoder3 = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)
        #self.encoder4 = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

        #self.Encoders = [self.encoder1, self.encoder2]

        self.decoder = Decoder(num_layers, d_model, d_timevec, num_heads, dff, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, sources, tar, training, look_ahead_mask):
        # dec_output.shape == (batch_size, input_seq_len, d_model)
        #enc_output = [self.Encoders[count](source) for count, source in enumerate(sources)]
        enc_output = self.encoder1(sources, training)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, 1)

        return final_output, attention_weights


# Hyperparameters
num_layers = 4
d_model = 128
d_timevec = d_model - 1
dff = 512
num_heads = 8
dropout_rate = 0.1


# Learning rate scheduler
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


with strategy.scope():
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

# Loss and metrics
with strategy.scope():
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


    def loss_function(real, pred):
        loss_ = loss_object(real, pred)

        return tf.reduce_sum(loss_)


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_accuracy')

    dev_accuracy = tf.keras.metrics.MeanAbsoluteError(name='dev_accuracy')
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_accuracy')

# Training and checkpointing
with strategy.scope():
    transformer = Transformer(num_layers, d_model, d_timevec, num_heads, dff, pe_input=128,
                              pe_target=64, rate=dropout_rate)

# Cannot use below until fully implementing GCloud TPU service
checkpoint_path = 'gs://tpu_daniel/{}_{}_{}_checkpoints'.format(exchange, input_method, data_format)

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!')

EPOCHS = 400

train_step_signature = [
    tf.TensorSpec(shape=(64, 128, 1), dtype=tf.float32),
    #tf.TensorSpec(shape=(64, 128, 1), dtype=tf.float32),
    #tf.TensorSpec(shape=(64, 128, 1), dtype=tf.float32),
    #tf.TensorSpec(shape=(64, 128, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(64, 65, 1), dtype=tf.float32)
]

with strategy.scope():
    @tf.function(input_signature=train_step_signature)
    def train_step(source, target):
        tar_inp = target[:, :-1, :]
        tar_real = target[:, 1:, :]

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])

        with tf.GradientTape() as tape:
            predictions, _ = transformer(source, tar_inp, True, look_ahead_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy.update_state(tar_real, predictions)


    @tf.function
    def distributed_train_step(source, target):
        strategy.run(train_step, args=(source, target))


    @tf.function(input_signature=train_step_signature)
    def dev_step(source, target):
        tar_inp = target[:, :-1, :]
        tar_real = target[:, 1:, :]

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])

        predictions, _ = transformer(source, tar_inp, False, look_ahead_mask)

        dev_accuracy.update_state(tar_real, predictions)


    @tf.function
    def distributed_dev_step(source, target):
        strategy.run(dev_step, args=(source, target))


    @tf.function(input_signature=train_step_signature)
    def test_step(source, target):
        tar_inp = target[:, :-1, :]
        tar_real = target[:, 1:, :]

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])

        predictions, _ = transformer(source, tar_inp, False, look_ahead_mask)

        test_accuracy.update_state(tar_real, predictions)


    @tf.function
    def distributed_test_step(source, target):
        strategy.run(test_step, args=(source, target))

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> history, positives, negatives, index; tar -> target
    for batch, (source, positive, negative, index, target) in enumerate(train_data):
        distributed_train_step(source, target)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} MAE: {:.4g}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    print('Epoch {} Loss {:.4f} MAE: {:.4g}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {:.4f} secs\n'.format(time.time() - start))

    # Cannot implement use checkpointing without authentication
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    if (epoch + 1) % 5 == 0:
        for batch, (source, positive, negative, index, target) in enumerate(dev_data):
            distributed_dev_step(source, target)

        print('Validation: Epoch {} MAE: {:.4g}\n'.format(epoch + 1,
                                                              dev_accuracy.result()))
        dev_accuracy.reset_states()

for batch, (source, positive, negative, index, target) in enumerate(test_data):
    distributed_test_step(source, target)

print('\nTest performance: MAE: {:.4g}'.format(test_accuracy.result()))
transformer.save_weights('gs://tpu_daniel/{}_{}_{}_model/{}_{}_{}'.format(exchange, input_method, data_format, exchange, input_method, data_format))
print('Training complete!')