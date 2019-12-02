"""
Apply multi-output Random Forest regression on the distributional targets
"""
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from preprocessors import preprocess_csv
from scipy.stats import truncnorm
from multiprocessing import Pool


SEED = 1337  # seed for k-fold split
NUM_FOLDS = 4  # k-fold num splits
LOW = 11  # lowest training price (log1p) = 11.51
HIGH = 19  # highest training price = 18.53
NUM_BINS = 10


def split_to_folds(input_df):
    """ Split input df into kfolds returning (train, val) index tuples """
    kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [x for x in kf.split(input_df)]


def train_fold(fold, train_df, test_df, train_labels, supports):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_labels[train_idx]

    model = RandomForestRegressor(n_estimators=300,
                                  n_jobs=8)
    model.fit(train_x, train_y)

    test_pred = np.expm1(np.dot(model.predict(test_df),
                                supports))

    return test_pred


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

    mean_pred = np.squeeze(np.mean(np.stack([x for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/rfr_dist_output.csv',
                                                  index=False)
    print('Elapsed time: {}'.format(time.time() - start_time))
