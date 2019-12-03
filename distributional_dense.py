"""
Reframe regression problem as a distributional problem
in the vein of "Improving Regression Performance with Distributional Losses"
https://arxiv.org/abs/1806.04613
- converts the regression target into bucketized trunc-norm centered around the regression
  and with stdev = radius of bin
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import truncnorm
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from preprocessors import preprocess_csv
from keras_util import ExponentialMovingAverage
from postprocessors import generate_stacking_inputs
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 1337  # seed for k-fold split
NUM_FOLDS = 8  # k-fold num splits
BATCH_SIZE = 64
NUM_EPOCHS = 30
LOW = 11  # lowest training price (log1p) = 11.51
HIGH = 19  # highest training price = 18.53
NUM_BINS = 10


def split_to_folds(input_df):
    """ Split input df into kfolds returning (train, val) index tuples """
    kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [x for x in kf.split(input_df)]


def build_dense_model():
    """ Simple two layer MLP """
    inputs = layers.Input(shape=(277,))
    output = layers.GaussianDropout(0.1)(inputs)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Dropout(0.1)(output)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Dense(NUM_BINS, activation='softmax')(output)
    model = Model(inputs=inputs,
                  outputs=output)

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy')
    return model


def train_fold(fold, train_df, test_df, train_labels, supports):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_labels[train_idx]

    val_x = train_df.iloc[val_idx].drop('price_doc', axis=1)
    val_y = train_labels[val_idx]

    std_scaler = StandardScaler().fit(train_x)
    train_x = std_scaler.transform(train_x)
    val_x = std_scaler.transform(val_x)

    model = build_dense_model()
    results = model.fit(x=train_x,
                        y=train_y,
                        batch_size=BATCH_SIZE,
                        validation_data=(val_x, val_y),
                        epochs=NUM_EPOCHS,
                        verbose=0,
                        callbacks=[ExponentialMovingAverage()])

    test_df = std_scaler.transform(test_df)
    raw_val_prob = model.predict(val_x)
    raw_test_prob = model.predict(test_df)
    test_pred = np.expm1(np.dot(raw_test_prob, supports))

    return results.history, test_pred, val_idx, raw_val_prob, raw_test_prob


def generate_target_dist(mean, num_bins=NUM_BINS, low=LOW, high=HIGH):
    """
    Generate discretized truncated norm prob distribution centered around mean
    :param mean: center of truncated norm
    :param num_bins: number of bins
    :param low: low end of truncated range
    :param high: top end of truncated range
    :return: (support, probabilities for support) tuple
    """
    radius = 0.5 * (high - low) / num_bins

    def trunc_norm_prob(center):
        """ get probability mass """
        return (truncnorm.cdf(center + radius,
                              a=(low - mean) / radius,
                              b=(high - mean) / radius,
                              loc=mean, scale=radius) -
                truncnorm.cdf(center - radius,
                              a=(low - mean) / radius,
                              b=(high - mean) / radius,
                              loc=mean, scale=radius))

    supports = np.array([x * (2 * radius) + radius + low for x in range(num_bins)])
    probs = np.array([trunc_norm_prob(support) for support in supports])
    return supports, probs


if __name__ == '__main__':
    start_time = time.time()
    train_ids, test_ids, processed_train_df, processed_test_df = preprocess_csv()

    # generate distribution labels
    with Pool(8) as p:
        train_labels = np.stack([x[1] for x in p.map(generate_target_dist, np.log1p(processed_train_df['price_doc']))])
    supports = generate_target_dist(-1)[0]

    folds = split_to_folds(processed_train_df)
    all_fold_results = []
    all_fold_preds = []

    with Pool(NUM_FOLDS) as p:
        combined_results = p.starmap(train_fold, ((curr_fold,
                                                   processed_train_df,
                                                   processed_test_df,
                                                   train_labels,
                                                   supports) for curr_fold in folds))

    losses = np.mean([x[0]['loss'] for x in combined_results], axis=0)
    val_losses = np.mean([x[0]['val_loss'] for x in combined_results], axis=0)
    print('Loss: {}'.format(losses))
    print('Val loss: {}'.format(val_losses))
    plt.plot(losses, label='train_losses')
    plt.plot(val_losses, label='val_losses')
    plt.title('Distributional Dense')
    plt.legend(loc='best')
    plt.show()

    mean_pred = np.squeeze(np.mean(np.stack([x[1] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/dist_dense_output.csv',
                                                  index=False)

    generate_stacking_inputs(filename='dist_dense_input',
                             oof_indices=np.concatenate([x[2] for x in combined_results]),
                             oof_preds=np.concatenate([x[3] for x in combined_results]),
                             test_preds=np.squeeze(np.mean(np.stack([x[4] for x in combined_results]), axis=0)),
                             train_ids=train_ids,
                             test_ids=test_ids,
                             train_df=processed_train_df)

    print('Elapsed time: {}'.format(time.time() - start_time))
