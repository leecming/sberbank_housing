""" Pre-processing helper functions """
import numpy as np
import pandas as pd

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
MACRO_PATH = 'data/macro.csv'


def preprocess_csv(ohe_features=False,
                   ohe_card=10):
    """
    Transforms raw data in input CSVs into features ready for modelling
    1. Drop id column
    2. Timestamp col to year, month, day columns
    3. Drop any non-numeric column
    4, if ohe_features, ohe all non-numeric columns + numeric columns w/ distinct < ohe_card
    """
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    macro_df = pd.read_csv(MACRO_PATH)

    macro_columns_interest = [col for col in macro_df.columns if macro_df[col].nunique() > 1000]
    macro_df = macro_df[macro_columns_interest]

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
    processed_df.drop('timestamp', axis=1, inplace=True)

    # 3. Drop non-numeric columns and if flag set, OH object columns
    start_cols = list(processed_df.columns)
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

    return train_ids, test_ids, processed_train, processed_test
