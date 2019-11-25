"""
Starter sklearn implementation of a Gradient Boosting regression model
on Sberbank Russian Housing Market dataset
(https://www.kaggle.com/c/sberbank-russian-housing-market)
- No pre-processing: just drop all non-numeric columns
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from preprocessors import preprocess_csv

SEED = 1337  # seed for kfold split
NUM_FOLDS = 4  # kfold num splits
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'


def split_to_folds(input_df):
    """ Split input df into kfolds returning (train, val) index tuples """
    kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [x for x in kf.split(input_df)]


def train_fold(fold, train_df, test_df):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_df.iloc[train_idx]['price_doc']

    gbr = GradientBoostingRegressor()

    gbr.fit(train_x,
            train_y)
    test_pred = gbr.predict(test_df)

    return test_pred


if __name__ == '__main__':
    raw_train_df = pd.read_csv(TRAIN_PATH)
    raw_test_df = pd.read_csv(TEST_PATH)

    processed_train_df, processed_test_df = preprocess_csv(raw_train_df, raw_test_df)

    assert raw_train_df.shape[0] == processed_train_df.shape[0]
    assert raw_test_df.shape[0] == processed_test_df.shape[0]

    folds = split_to_folds(processed_train_df)
    all_fold_preds = []
    for curr_fold in folds:
        curr_pred = train_fold(curr_fold,
                               processed_train_df,
                               processed_test_df)
        all_fold_preds.append(curr_pred)

    mean_pred = np.squeeze(np.mean(np.stack(all_fold_preds), axis=0))
    print(mean_pred.shape)

    pd.DataFrame({'id': raw_test_df['id'],
                  'price_doc': mean_pred}).to_csv('data/gb_regressor_output.csv',
                                                  index=False)
