"""
Starter sklearn implementation of a Random Forest regression model
on Sberbank Russian Housing Market dataset
(https://www.kaggle.com/c/sberbank-russian-housing-market)
- Uses MP to do parallel processing by folds
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from preprocessors import preprocess_csv
from postprocessors import  generate_stacking_inputs
from multiprocessing import Pool

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

    rfr = RandomForestRegressor(n_estimators=300,
                                n_jobs=4)

    rfr.fit(train_x, train_y)
    val_pred = rfr.predict(val_x)
    test_pred = rfr.predict(test_df)

    return test_pred, val_idx, val_pred


if __name__ == '__main__':
    start_time = time.time()
    train_ids, test_ids, processed_train_df, processed_test_df = preprocess_csv(ohe_features=True,
                                                                                ohe_card=20)

    folds = split_to_folds(processed_train_df)

    with Pool(NUM_FOLDS) as p:
        combined_results = p.starmap(train_fold, ((curr_fold,
                                                   processed_train_df,
                                                   processed_test_df) for curr_fold in folds))

    mean_pred = np.squeeze(np.mean(np.stack([x[0] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/rf_regressor_output.csv',
                                                  index=False)

    generate_stacking_inputs(filename='rf_regressor',
                             oof_indices=np.concatenate([x[1] for x in combined_results]),
                             oof_preds=np.concatenate([x[2] for x in combined_results]),
                             test_preds=mean_pred,
                             train_ids=train_ids,
                             test_ids=test_ids,
                             train_df=processed_train_df)

    print('Elapsed time: {}'.format(time.time() - start_time))