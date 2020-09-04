import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import os.path as path
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

# LOAD DATA
with np.load('/content/drive/My Drive/{}_input_ready.npz'.format(exchange)) as data:
    x_source = data['x_source'].astype(np.float32)
    dev_source = data['dev_source'].astype(np.float32)
    test_source = data['test_source'].astype(np.float32)
    x_target = data['x_target'].squeeze().astype(np.float32)
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

train_data = tf.data.Dataset.from_tensor_slices((x_source, x_target))
dev_data = tf.data.Dataset.from_tensor_slices((dev_source, dev_target))
test_data = tf.data.Dataset.from_tensor_slices((test_source, test_target))

BATCH_SIZE = 64 * strategy.num_replicas_in_sync

train_data = train_data.cache()
train_data = train_data.shuffle(x_source.shape[0]).batch(BATCH_SIZE, drop_remainder=True)
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

dev_data = dev_data.batch(BATCH_SIZE, drop_remainder=True)

test_data = test_data.batch(BATCH_SIZE, drop_remainder=True)


# GET POSITIONAL ENCODINGS
class PositionalEncoding(layers.Layer):
    def __init__(self, timevec_dim, seq_len):
        super(PositionalEncoding, self).__init__()
        self.timevec_dim = timevec_dim
        self.seq_len = seq_len

    def get_angles(self, pos, i, timevec_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(timevec_dim))
        return pos * angle_rates

    def positional_encoding(self, position, d_timevec):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(timevec_dim)[np.newaxis, :],
                                     timevec_dim)

        pos_encoding = np.zeros((1, position, timevec_dim))
        pos_encoding[:, :, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, :, 1::2] = np.sin(angle_rads[:, 1::2])

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        encodings = tf.ones([tf.shape(x)[0], self.seq_len, self.timevec_dim]) * self.positional_encoding(self.seq_len,
                                                                                                         self.timevec_dim)
        return tf.cast(tf.concat([x, encodings], axis=-1), dtype=tf.float32)


class TimeVec(layers.Layer):
    def __init__(self, embed_dim, timevec_dim, seq_len):
        super(TimeVec, self).__init__()
        self.embed_dim = embed_dim
        self.timevec_dim = timevec_dim
        self.seq_len = seq_len
        self.range = tf.range(seq_len, dtype=tf.float32)[:, None]

    def build(self, input_shape):
        self.W = tf.Variable(tf.keras.backend.random_normal(shape=(1, self.timevec_dim - 1), dtype=tf.float32),
                             trainable=True)
        self.P = tf.Variable(tf.keras.backend.random_normal(shape=(1, self.timevec_dim - 1), dtype=tf.float32),
                             trainable=True)
        self.w = tf.Variable(tf.keras.backend.random_normal(shape=(1, 1), dtype=tf.float32), trainable=True)
        self.p = tf.Variable(tf.keras.backend.random_normal(shape=(1, 1), dtype=tf.float32), trainable=True)
        self.dense = layers.Dense(self.embed_dim - self.timevec_dim)

    def call(self, x):
        first = self.range * self.w + self.p
        periodic = tf.sin(self.range * self.W + self.P)
        time_vec = tf.concat([first, periodic], axis=-1)
        expanded = self.dense(x)
        encodings = tf.ones([tf.shape(expanded)[0], self.seq_len, self.timevec_dim]) * time_vec
        return tf.cast(tf.concat([expanded, encodings], axis=-1), dtype=tf.float32)


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}'
            )
        self.projection_dim = embed_dim // num_heads

    def build(self, input_shape):
        self.query_dense = layers.Dense(self.embed_dim, input_shape=input_shape)
        self.key_dense = layers.Dense(self.embed_dim, input_shape=input_shape)
        self.value_dense = layers.Dense(self.embed_dim, input_shape=input_shape)
        self.combine_heads = layers.Dense(self.embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # ...
        value = self.value_dense(inputs)  # ...
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # ...
        value = self.separate_heads(
            value, batch_size)
        # ...
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.attn(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class FinalBlock(layers.Layer):
    def __init__(self, rate):
        super(FinalBlock, self).__init__()
        self.rate = dropout_rate

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(filters=8, kernel_size=3, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')
        self.conv3 = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.conv4 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')

        self.pool1 = layers.MaxPool1D(pool_size=2, strides=2)
        self.pool2 = layers.MaxPool1D(pool_size=2, strides=2)
        self.pool3 = layers.MaxPool1D(pool_size=2, strides=2)
        self.pool4 = layers.MaxPool1D(pool_size=2, strides=2)

        self.global_avg = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(32, activation='relu')
        self.dropout = layers.Dropout(self.rate)
        self.dense2 = layers.Dense(1, activation='linear')

    def call(self, inputs, training):
        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)

        x = self.global_avg(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        outputs = self.dense2(x)

        return outputs


def create_model(embed_dim, num_heads, ff_dim, timevec_dim=3, seq_len=128, rate=0.1):
    source_input = layers.Input(shape=(seq_len, 1))
    source_embedding_layer = PositionalEncoding(timevec_dim, seq_len)
    source = source_embedding_layer(source_input)

    source_transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    source_transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    source_transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    source_transformer_block4 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)

    source = source_transformer_block1(source)
    source = source_transformer_block2(source)
    source = source_transformer_block3(source)
    source = source_transformer_block4(source)

    """
    pos_input = layers.Input(shape=(seq_len, 5))
    pos_embedding_layer = TimeVec(embed_dim, timevec_dim, seq_len)
    pos = pos_embedding_layer(pos_input)

    pos_transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    pos_transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    pos_transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    pos_transformer_block4 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)

    pos = pos_transformer_block1(pos)
    pos = pos_transformer_block2(pos)
    pos = pos_transformer_block3(pos)
    pos = pos_transformer_block4(pos)

    neg_input = layers.Input(shape=(seq_len, 5))
    neg_embedding_layer = TimeVec(embed_dim, timevec_dim, seq_len)
    neg = neg_embedding_layer(neg_input)

    neg_transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    neg_transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    neg_transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    neg_transformer_block4 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)

    neg = neg_transformer_block1(neg)
    neg = neg_transformer_block2(neg)
    neg = neg_transformer_block3(neg)
    neg = neg_transformer_block4(neg)

    index_input = layers.Input(shape=(seq_len, 5))
    index_embedding_layer = TimeVec(embed_dim, timevec_dim, seq_len)
    index = index_embedding_layer(index_input)

    index_transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    index_transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    index_transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
    index_transformer_block4 = TransformerBlock(embed_dim, num_heads, ff_dim, rate)

    index = index_transformer_block1(index)
    index = index_transformer_block2(index)
    index = index_transformer_block3(index)
    index = index_transformer_block4(index)

    combined = tf.stack([source, pos, neg, index], axis=1)
    """

    final_block = FinalBlock(rate)
    outputs = final_block(source)

    model = keras.Model(inputs=source_input, outputs=outputs)

    return model


# Hyperparameters
embed_dim = 128
timevec_dim = 127
ff_dim = 1024
num_heads = 8
dropout_rate = 0.1
seq_len = 128


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


with strategy.scope():
    learning_rate = CustomSchedule(embed_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

# Cannot use below until fully implementing GCloud TPU service
checkpoint_path = 'gs://tpu_daniel/{}_checkpoints/'.format(exchange)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq=2410
)

# Training and checkpointing
with strategy.scope():
    try:
        model = keras.models.load_model(checkpoint_path, custom_objects={'CustomSchedule': CustomSchedule})
        print('Checkpoint restored!')
    except OSError:
        model = create_model(embed_dim, num_heads, ff_dim,
                             timevec_dim=timevec_dim, seq_len=seq_len, rate=dropout_rate)
        model.compile(loss='mse', optimizer=optimizer,
                      metrics=['mean_squared_error', 'mean_absolute_error'])
        print('No checkpoint found. New model created!')

EPOCHS = 80

model.fit(train_data, epochs=EPOCHS, callbacks=[checkpoint_callback], validation_data=dev_data)
model.evaluate(test_data)
model.save('gs://tpu_daniel/{}_model/'.format(exchange, exchange))
