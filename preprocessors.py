""" Pre-processing helper functions """
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import truncnorm

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
MACRO_PATH = 'data/macro.csv'
# used to generate "days since" feature (2010-01-01 first date on macro.csv)
DATE_START_POINT = pd.Timestamp('2010-01-01')


def mixup_generator(train_features, train_labels, batch_size, alpha=0.2):
    """ Generates linear mixes of train_features and targets with the proportion derived from the beta dist"""
    assert len(train_features) == len(train_labels)

    while True:
        p = np.random.beta(alpha, alpha, size=(batch_size, 1))
        indices_a = np.random.choice(list(range(len(train_features))), size=(batch_size,))
        indices_b = np.random.choice(list(range(len(train_features))), size=(batch_size,))

        combined_features = p * train_features[indices_a] + (1 - p) * train_features[indices_b]

        if train_labels.ndim == 1:
            p = p.reshape(-1)
        combined_targets = p * train_labels[indices_a] + (1 - p) * train_labels[indices_b]
        yield combined_features, combined_targets


def split_to_folds(input_df, num_folds, seed=None):
    """ Split input df into kfolds returning (train, val) index tuples """
    kf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    return [x for x in kf.split(input_df)]


def window_stack(a, step_size=1, width=3):
    """
    Returns sliding window of numpy matrix by <width> timesteps stacked
    e.g., for input matrix of (X, 33) with width 100
    returns (X-99, 100, 33)
    """
    return np.stack((a[i:1 + i - width or None:step_size] for i in range(0, width)),
                    axis=1)


def generate_simple_macro(input_df):
    """ Pulls macro data to the input df by timestamp """
    macro_df = pd.read_csv(MACRO_PATH)
    macro_df = macro_df.iloc[:-1]  # last row garbage
    macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')
    ts_df = pd.DataFrame(macro_df['timestamp'].str.split('-', expand=True), dtype='int')
    numeric_cols = [col for col in macro_df.columns if macro_df[col].dtype in [np.float,
                                                                               np.int]]
    macro_df[['ts_year', 'ts_month', 'ts_day']] = ts_df
    macro_df = macro_df.set_index('timestamp')[list(numeric_cols) +
                                               ['ts_year', 'ts_month', 'ts_day']]

    combined_df = input_df.set_index('timestamp').join(macro_df)
    combined_df = combined_df[macro_df.columns]

    # downcast DP to SP
    for col in combined_df.columns:
        if combined_df[col].dtype == np.float64:
            combined_df[col] = combined_df[col].astype(np.float32)
        elif combined_df[col].dtype == np.int64:
            combined_df[col] = combined_df[col].astype(np.int32)

    return combined_df


def generate_macro_windows(min_unique=100, lookback_period=100, monthly_resampling=False):
    """
    Converts raw macro csv into stacked matrix over the lookback period
    and for macro columns that have >= min_unique values
    returns a dates df (for matching), and their corresponding lookback data
    """
    macro_df = pd.read_csv(MACRO_PATH).iloc[:-1]
    macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')

    if monthly_resampling:
        macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'])
        macro_df = macro_df.set_index('timestamp').resample('M').mean().reset_index()

    unique_vals = macro_df.nunique()
    macro_df = macro_df[unique_vals[unique_vals >= min_unique].index.values]

    # Normalize all columns except for timestamp
    macro_df[[x for x in macro_df.columns if x != 'timestamp']] = StandardScaler().fit_transform(
        macro_df[[x for x in macro_df.columns if x != 'timestamp']])

    rolling_matrix = window_stack(macro_df, width=lookback_period)
    rolling_dates = pd.DataFrame(rolling_matrix[:, -1, 0]).reset_index()
    rolling_dates.columns = ['rolling_id', 'timestamp']

    # drop dates from matrix
    rolling_matrix = rolling_matrix[:, :, 1:]

    return rolling_dates, rolling_matrix


def handle_ohe_columns(processed_df, ohe_features, ohe_card):
    """ OHE non-numeric columns and numeric columns with low cardinality """
    start_cols = list(processed_df.columns)
    # do not OHE datetime fields
    [start_cols.remove(x) for x in ['timestamp', 'ts_year', 'ts_month', 'ts_day']]
    non_numeric_cols = []
    for col in start_cols:
        if processed_df[col].dtype not in [np.int, np.float] \
                or (ohe_features and processed_df[col].nunique() <= ohe_card and col != 'is_train'):
            if ohe_features:
                oh_df = pd.get_dummies(processed_df[col])
                oh_df.columns = [col + '_' + str(x) for x in oh_df.columns]
                processed_df = pd.concat([processed_df, oh_df], axis=1)

            non_numeric_cols.append(col)

    return processed_df.drop(non_numeric_cols, axis=1)


def generate_target_dist(mean, num_bins, low, high):
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


def demean_by_macro_indicator(train_df, indicator='cpi'):
    """
    divide price_doc by CPI for that given date
    """
    macro_df = pd.read_csv(MACRO_PATH).iloc[:-1].fillna(method='ffill').fillna(method='bfill')
    macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')
    # norm against their values on the 1st date of the macro time series
    numeric_cols = [col for col in macro_df.columns if macro_df[col].dtype in [np.int, np.float]]
    macro_df[numeric_cols] /= macro_df[numeric_cols].iloc[0]

    macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'])
    macro_df = macro_df.set_index('timestamp')[indicator]
    train_df = train_df.set_index('timestamp').join(macro_df)
    train_df['price_doc'] = train_df['price_doc'] / train_df[indicator]
    train_df = train_df.drop(indicator, axis=1).reset_index()
    train_df['timestamp'] = train_df['timestamp'].dt.strftime('%Y-%m-%d')
    return train_df


def remean_price_by_indicator(output_df, indicator):
    """ for a model trained by a demeaned target, need to remean the predictions"""
    timestamp_df = pd.read_csv(TRAIN_PATH).set_index('id')['timestamp']
    timestamp_df = timestamp_df.append(pd.read_csv(TEST_PATH).set_index('id')['timestamp'])
    output_df = output_df.set_index('id').join(timestamp_df).reset_index()
    output_df['timestamp'] = pd.to_datetime(output_df['timestamp'])
    output_df = output_df.set_index('timestamp')

    indicator_df = pd.read_csv('data/macro.csv').iloc[:-1].fillna(method='ffill').fillna(method='bfill')
    # norm against their values on the 1st date of the macro time series
    numeric_cols = [col for col in indicator_df.columns if indicator_df[col].dtype in [np.int, np.float]]
    indicator_df[numeric_cols] /= indicator_df[numeric_cols].iloc[0]
    indicator_df['timestamp'] = pd.to_datetime(indicator_df['timestamp'])
    indicator_df = indicator_df.set_index('timestamp')[indicator]

    output_df = output_df.join(indicator_df)
    output_df['price_doc'] = output_df['price_doc'] * output_df[indicator]
    return output_df.reset_index(drop=True).drop(indicator, axis=1)


def preprocess_csv(ohe_features=False,
                   ohe_card=10,
                   rolling_macro=None,
                   demeaning_indicator=None,
                   simple_macro=False):
    """
    Transforms raw data in input CSVs into features ready for modelling
    1. If flagged, demean price targets by specified macro indicator
    2. Drop id column
    3.
    4. Timestamp col to year, month, day columns
    5. Drop any non-numeric column & if ohe_features,
        ohe all non-numeric columns + numeric columns w/ distinct < ohe_card
    6. If flagged, generate rolling windows of macro data
    """
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # infer arg needed otherwise int columns get casted as float
    train_df = train_df.fillna(-1, downcast='infer')
    train_df['is_train'] = 1

    # 1. Demean by Indicator
    if demeaning_indicator:
        train_df = demean_by_macro_indicator(train_df, indicator=demeaning_indicator)

    test_df = test_df.fillna(-1, downcast='infer')
    test_df['is_train'] = 0

    processed_df = train_df.append(test_df,
                                   ignore_index=True,
                                   sort=False)

    # 2. Drop ID col
    train_ids = processed_df[processed_df['is_train'] == 1]['id'].values
    test_ids = processed_df[processed_df['is_train'] == 0]['id'].values
    processed_df.drop(['id'], axis=1, inplace=True)

    # 3. Generate macro data aligned with the processed train/test data on the timestamp
    train_macro, test_macro = None, None
    if simple_macro:
        train_macro = generate_simple_macro(train_df)
        test_macro = generate_simple_macro(test_df)

    # 4. Split timestamp
    ts_df = pd.DataFrame(processed_df['timestamp'].str.split('-', expand=True), dtype='int')
    processed_df[['ts_year', 'ts_month', 'ts_day']] = ts_df

    # 5. Drop non-numeric columns and if flag set, OH object columns
    processed_df = handle_ohe_columns(processed_df, ohe_features, ohe_card)

    # reduce memory usage by downcasting all DP values to SP
    for col in processed_df.columns:
        if processed_df[col].dtype == np.float64:
            processed_df[col] = processed_df[col].astype(np.float32)
        elif processed_df[col].dtype == np.int64:
            processed_df[col] = processed_df[col].astype(np.int32)

    processed_train = processed_df[processed_df['is_train'] == 1].drop('is_train', axis=1)
    processed_test = processed_df[processed_df['is_train'] == 0].drop(['is_train', 'price_doc'],
                                                                      axis=1)

    train_rolling, test_rolling = None, None
    # 6. Generate lookback data
    if rolling_macro:
        # Convert all dates to end-of-month to allow for join with the macro data
        if 'monthly_resampling' in rolling_macro:
            processed_train['timestamp'] = pd.to_datetime(processed_train['timestamp']) + pd.offsets.MonthEnd(0)
            processed_test['timestamp'] = pd.to_datetime(processed_test['timestamp']) + pd.offsets.MonthEnd(0)

        rolling_dates, rolling_matrix = generate_macro_windows(**rolling_macro)
        train_rolling = rolling_matrix[
            processed_train.set_index('timestamp').join(rolling_dates.set_index('timestamp'))[
                'rolling_id'].values].astype(np.float32)
        test_rolling = rolling_matrix[
            processed_test.set_index('timestamp').join(rolling_dates.set_index('timestamp'))[
                'rolling_id'].values].astype(np.float32)

    # Drop timestamp column after generating lookups
    # Be careful:- they've been converted to last day of month
    processed_train.drop('timestamp', axis=1, inplace=True)
    processed_test.drop('timestamp', axis=1, inplace=True)

    return {'train_ids': train_ids,
            'test_ids': test_ids,
            'processed_train': processed_train,
            'processed_test': processed_test,
            'train_rolling': train_rolling,
            'test_rolling': test_rolling,
            'train_macro': train_macro,
            'test_macro': test_macro}


if __name__ == '__main__':
    preprocess_dict = preprocess_csv(ohe_features=True)
    (train_ids,
     test_ids,
     processed_train_df,
     processed_test_df) = [preprocess_dict[key] for key in ['train_ids',
                                                            'test_ids',
                                                            'processed_train',
                                                            'processed_test']]

    numerator = processed_train_df.sample(n=1000000, replace=True).values
    denominator = processed_train_df.sample(n=1000000, replace=True).values
    output_df = pd.DataFrame((numerator / denominator), columns=processed_train_df.columns).astype(np.float32)
    print(output_df.info())
    print(output_df.sample(n=10))
