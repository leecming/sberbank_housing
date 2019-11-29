""" Pre-processing helper functions """
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
MACRO_PATH = 'data/macro.csv'


def window_stack(a, step_size=1, width=3):
    """
    Returns sliding window of numpy matrix by <width> timesteps stacked
    e.g., for input matrix of (X, 33) with width 100
    returns (X-99, 100, 33)
    """
    return np.stack((a[i:1 + i - width or None:step_size] for i in range(0, width)),
                    axis=1)


def generate_macro_windows(min_unique=100, lookback_period=100):
    """
    Converts raw macro csv into stacked matrix over the lookback period
    and for macro columns that have >= min_unique values
    returns a dates df (for matching), and their corresponding lookback data
    """
    macro_df = pd.read_csv(MACRO_PATH)
    macro_df = macro_df.fillna(method='bfill').fillna('ffill')

    unique_vals = macro_df.nunique()
    macro_df = macro_df[unique_vals[unique_vals >= min_unique].index.values]
    macro_df = macro_df.iloc[:-1]  # last row garbage
    # Normalize all columns except for timestamp
    macro_df[[x for x in macro_df.columns if x != 'timestamp']] = StandardScaler().fit_transform(
        macro_df[[x for x in macro_df.columns if x != 'timestamp']])

    rolling_matrix = window_stack(macro_df, width=100)
    rolling_dates = pd.DataFrame(rolling_matrix[:, -1, 0]).reset_index()
    rolling_dates.columns = ['rolling_id', 'timestamp']

    # drop dates from matrix
    rolling_matrix = rolling_matrix[:, :, 1:]

    return rolling_dates, rolling_matrix


def preprocess_csv(ohe_features=False,
                   ohe_card=10,
                   generate_rolling=False):
    """
    Transforms raw data in input CSVs into features ready for modelling
    1. Drop id column
    2. Timestamp col to year, month, day columns
    3. Drop any non-numeric column
    4, if ohe_features, ohe all non-numeric columns + numeric columns w/ distinct < ohe_card
    """
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # infer arg needed otherwise int columns get casted as float
    train_df = train_df.fillna(-1, downcast='infer')
    train_df['is_train'] = 1

    test_df = test_df.fillna(-1, downcast='infer')
    test_df['is_train'] = 0

    processed_df = train_df.append(test_df,
                                   ignore_index=True,
                                   sort=False)

    # 1. Drop ID col
    train_ids = processed_df[processed_df['is_train'] == 1]['id'].values
    test_ids = processed_df[processed_df['is_train'] == 0]['id'].values
    processed_df.drop(['id'], axis=1, inplace=True)

    # 2. Split timestamp
    ts_df = pd.DataFrame(processed_df['timestamp'].str.split('-', expand=True), dtype='int')
    processed_df[['ts_year', 'ts_month', 'ts_day']] = ts_df
    # processed_df.drop('timestamp', axis=1, inplace=True)

    # 3. Drop non-numeric columns and if flag set, OH object columns
    start_cols = list(processed_df.columns)
    start_cols.remove('timestamp')
    non_numeric_cols = []
    for col in start_cols:
        if processed_df[col].dtype not in [np.int, np.float] \
                or (ohe_features and processed_df[col].nunique() <= ohe_card and col != 'is_train'):
            if ohe_features:
                oh_df = pd.get_dummies(processed_df[col])
                oh_df.columns = [col + '_' + str(x) for x in oh_df.columns]
                processed_df = pd.concat([processed_df, oh_df], axis=1)

            non_numeric_cols.append(col)
    processed_df.drop(non_numeric_cols, axis=1, inplace=True)

    processed_train = processed_df[processed_df['is_train'] == 1].drop('is_train', axis=1)
    processed_test = processed_df[processed_df['is_train'] == 0].drop(['is_train', 'price_doc'],
                                                                      axis=1)

    # Generate lookback data
    if generate_rolling:
        rolling_dates, rolling_matrix = generate_macro_windows()
        train_rolling = rolling_matrix[
            processed_train.set_index('timestamp').join(rolling_dates.set_index('timestamp'))['rolling_id'].values]
        test_rolling = rolling_matrix[
            processed_test.set_index('timestamp').join(rolling_dates.set_index('timestamp'))['rolling_id'].values]

    # Drop timestamp column after generating lookups
    processed_train.drop('timestamp', axis=1, inplace=True)
    processed_test.drop('timestamp', axis=1, inplace=True)

    if generate_rolling:
        return train_ids, test_ids, processed_train, processed_test, train_rolling, test_rolling
    else:
        return train_ids, test_ids, processed_train, processed_test