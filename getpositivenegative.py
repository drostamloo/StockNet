import numpy as np
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

with np.load('{}_train/{}_{}_history_target.npz'.format(exchange, exchange, data_format)) as arrays:
    history = arrays['history']


# Fetch averages of most and least correlated stocks belonging to a given time window
def get_positive_and_negative(source_array):
    # Compute average prices
    #source_averages = np.mean(history[:, :, :4], axis=-1, keepdims=True)

    positives = np.zeros((num_companies, num_windows, history_size, num_features))
    negatives = np.zeros((num_companies, num_windows, history_size, num_features))

    for w in range(num_windows):
        correlations = np.corrcoef(source_array[:, w, :, 0])

        for c in range(num_companies):
            idx_positives = np.argpartition(correlations[c, :], -21)[-21:-1]
            idx_negatives = np.argpartition(correlations[c, :], 20)[:20]
            positives[c, w, :, :] = np.mean(source_array[idx_positives, w, :, :], axis=0)
            negatives[c, w, :, :] = np.mean(source_array[idx_negatives, w, :, :], axis=0)

    return positives, negatives


positives, negatives = get_positive_and_negative(history)
print(positives.shape)
print(negatives.shape)

np.savez('{}_train/{}_{}_pn.npz'.format(exchange, exchange, data_format), positives=positives, negatives=negatives)
