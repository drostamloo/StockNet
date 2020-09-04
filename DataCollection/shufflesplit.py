import numpy as np

exchange = 'snp'
data_format = 'PRICE5'

with np.load('{}_train/{}_{}_history_target.npz'.format(exchange, exchange, data_format)) as history_target:
    history = history_target['history']
    target = history_target['target']

with np.load('{}_train/{}_{}_pn.npz'.format(exchange, exchange, data_format)) as positives_negatives:
    positives = positives_negatives['positives']
    negatives = positives_negatives['negatives']

with np.load('{}_train/{}_{}_index.npz'.format(exchange, exchange, data_format)) as index:
    index = index['index']

# Reshape data and shuffle
originals = [history, target, positives, negatives, index]

# Squeeze target array only for second type of transformer (encoder only) implementation
#reshaped[1] = np.squeeze(reshaped[1], axis=1)
num_examples = history.shape[0] * history.shape[1]
num_windows = history.shape[1]
history_window_size = history.shape[2]
target_window_size = target.shape[2]
num_features = history.shape[3]
print('{} total examples generated.'.format(num_examples))

assert(index.shape[0] == history.shape[0])

x_source, _, devtest_source = np.split(originals[0], [int(0.8 * num_windows), int(0.92 * num_windows)], axis=1)
x_target, _, devtest_target = np.split(originals[1], [int(0.8 * num_windows), int(0.92 * num_windows)], axis=1)
x_positive, _, devtest_positive = np.split(originals[2], [int(0.8 * num_windows), int(0.92 * num_windows)], axis=1)
x_negative, _, devtest_negative = np.split(originals[3], [int(0.8 * num_windows), int(0.92 * num_windows)], axis=1)
x_index, _, devtest_index = np.split(originals[4], [int(0.8 * num_windows), int(0.92 * num_windows)], axis=1)

x_source = x_source.reshape((-1, history_window_size, num_features))
x_target = x_target.reshape((-1, target_window_size, num_features))
x_positive = x_positive.reshape((-1, history_window_size, num_features))
x_negative = x_negative.reshape((-1, history_window_size, num_features))
x_index = x_index.reshape((-1, history_window_size, num_features))
devtest_source = devtest_source.reshape((-1, history_window_size, num_features))
devtest_target = devtest_target.reshape((-1, target_window_size, num_features))
devtest_positive = devtest_positive.reshape((-1, history_window_size, num_features))
devtest_negative = devtest_negative.reshape((-1, history_window_size, num_features))
devtest_index = devtest_index.reshape((-1, history_window_size, num_features))

state = np.random.get_state()
np.random.set_state(state)
np.random.shuffle(devtest_source)
np.random.set_state(state)
np.random.shuffle(devtest_target)
np.random.set_state(state)
np.random.shuffle(devtest_positive)
np.random.set_state(state)
np.random.shuffle(devtest_negative)
np.random.set_state(state)
np.random.shuffle(devtest_index)
np.random.set_state(state)

dev_source, test_source = np.split(devtest_source, [devtest_source.shape[0] // 2])
dev_target, test_target = np.split(devtest_target, [devtest_target.shape[0] // 2])
dev_positive, test_positive = np.split(devtest_positive, [devtest_positive.shape[0] // 2])
dev_negative, test_negative = np.split(devtest_negative, [devtest_negative.shape[0] // 2])
dev_index, test_index = np.split(devtest_index, [devtest_index.shape[0] // 2])

print(x_source.shape)
print(x_target.shape)
print(dev_source.shape)
print(dev_target.shape)
print(test_source.shape)
print(test_target.shape)

print('Number of windows removed to isolate dev and test sets: {}'.format(_.shape[1]))

np.savez('{}_train/{}_{}_input_ready.npz'.format(exchange, exchange, data_format),
         x_source=x_source, dev_source=dev_source, test_source=test_source,
         x_target=x_target, dev_target=dev_target, test_target=test_target,
         x_positive=x_positive, dev_positive=dev_positive, test_positive=test_positive,
         x_negative=x_negative, dev_negative=dev_negative, test_negative=test_negative,
         x_index=x_index, dev_index=dev_index, test_index=test_index)
