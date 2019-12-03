"""
Starter lgbm implementation of a ExtraTrees regressor
on Sberbank Russian Housing Market dataset
(https://www.kaggle.com/c/sberbank-russian-housing-market)
- Left the MP stubs but LGBM doesn't seem to play well with MP - running in sequence currently
"""
import time
from itertools import starmap
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from preprocessors import preprocess_csv
from postprocessors import generate_stacking_inputs

SEED = 1337  # seed for kfold split
NUM_FOLDS = 4  # kfold num splits


def split_to_folds(input_df):
    """ Split input df into kfolds returning (train, val) index tuples """
    kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [x for x in kf.split(input_df)]


def train_fold(fold, train_df, test_df):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_df.iloc[train_idx]['price_doc']

    val_x = train_df.iloc[val_idx].drop('price_doc', axis=1)

    lgbmr = LGBMRegressor(n_estimators=1000,
                          boosting_type='dart',
                          max_bin=1024,
                          num_leaves=127)

    lgbmr.fit(train_x,
              train_y)
    val_pred = lgbmr.predict(val_x)
    test_pred = lgbmr.predict(test_df)

    return test_pred, val_idx, val_pred


if __name__ == '__main__':
    start_time = time.time()
    train_ids, test_ids, processed_train_df, processed_test_df = preprocess_csv(ohe_features=True,
                                                                                ohe_card=20)

    folds = split_to_folds(processed_train_df)

    combined_results = list(starmap(train_fold, ((curr_fold,
                                                  processed_train_df,
                                                  processed_test_df) for curr_fold in folds)))

    mean_pred = np.squeeze(np.mean(np.stack([x[0] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/lgbm_regressor_output.csv',
                                                  index=False)

    generate_stacking_inputs(filename='lgbm_regressor',
                             oof_indices=np.concatenate([x[1] for x in combined_results]),
                             oof_preds=np.concatenate([x[2] for x in combined_results]),
                             test_preds=mean_pred,
                             train_ids=train_ids,
                             test_ids=test_ids,
                             train_df=processed_train_df)

    print('Elapsed time: {}'.format(time.time() - start_time))
