""" Pre-processing helper functions """
import numpy as np
import pandas as pd


def preprocess_csv(train_df, test_df):
    """
    Transforms raw data in input CSVs into features ready for modelling
    1. Drop id column
    2. Timestamp col to year, month, day columns
    3. Drop any non-numeric column
    """
    # infer arg needed otherwise int columns get casted as float
    temp_train_df = train_df.copy().fillna(-1, downcast='infer')
    temp_train_df['is_train'] = 1
    temp_test_df = test_df.copy().fillna(-1, downcast='infer')
    temp_test_df['is_train'] = 0
    processed_df = temp_train_df.append(temp_test_df,
                                        ignore_index=True,
                                        sort=False)

    # 1. Drop ID col
    processed_df.drop(['id'], axis=1, inplace=True)

    # 2. Split timestamp
    ts_df = pd.DataFrame(processed_df['timestamp'].str.split('-', expand=True), dtype='int')
    processed_df[['ts_year', 'ts_month', 'ts_day']] = ts_df
    processed_df.drop('timestamp', axis=1, inplace=True)

    # 3. Drop non-numeric columns
    for col in processed_df.columns:
        if processed_df[col].dtype not in [np.int, np.float]:
            processed_df.drop(col, axis=1, inplace=True)

    processed_train = processed_df[processed_df['is_train'] == 1].drop('is_train', axis=1)
    processed_test = processed_df[processed_df['is_train'] == 0].drop(['is_train', 'price_doc'],
                                                                      axis=1)

    return processed_train, processed_test
