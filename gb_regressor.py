"""
Starter sklearn implementation of a Gradient Boosting regression model
on Sberbank Russian Housing Market dataset
(https://www.kaggle.com/c/sberbank-russian-housing-market)
- Uses MP to do parallel processing by folds
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from preprocessors import preprocess_csv, split_to_folds
from postprocessors import  generate_stacking_inputs
from multiprocessing import Pool

SEED = 1337  # seed for kfold split
NUM_FOLDS = 4  # kfold num splits


def train_fold(fold, train_df, test_df):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_df.iloc[train_idx]['price_doc']

    all_preds = []
    for curr_divisor in [None, 'full_sq']:
        gbr = GradientBoostingRegressor(n_estimators=300)

        if curr_divisor:
            gbr.fit(train_x, train_y / train_df.iloc[train_idx][curr_divisor])
            all_preds.append(gbr.predict(test_df) * test_df[curr_divisor])
        else:
            gbr.fit(train_x, train_y)
            all_preds.append(gbr.predict(test_df))

    stacked_pred = np.stack(all_preds, axis=0)
    test_pred = np.nanmean(np.where(stacked_pred > 0, stacked_pred, np.nan), axis=0)

    return test_pred, val_idx, None


if __name__ == '__main__':
    start_time = time.time()
    preprocess_dict = preprocess_csv()
    (train_ids,
     test_ids,
     processed_train_df,
     processed_test_df) = [preprocess_dict[key] for key in ['train_ids',
                                                            'test_ids',
                                                            'processed_train',
                                                            'processed_test']]

    folds = split_to_folds(processed_train_df,
                           num_folds=NUM_FOLDS,
                           seed=SEED,
                           shuffle=False)

    with Pool(NUM_FOLDS) as p:
        combined_results = p.starmap(train_fold, ((curr_fold,
                                                   processed_train_df,
                                                   processed_test_df) for curr_fold in folds))

    mean_pred = np.squeeze(np.mean(np.stack([x[0] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/gb_regressor_output.csv',
                                                  index=False)

    # generate_stacking_inputs(filename='gb_regressor',
    #                          oof_indices=np.concatenate([x[1] for x in combined_results]),
    #                          oof_preds=np.concatenate([x[2] for x in combined_results]),
    #                          test_preds=mean_pred,
    #                          train_ids=train_ids,
    #                          test_ids=test_ids,
    #                          train_df=processed_train_df)

    print('Elapsed time: {}'.format(time.time() - start_time))
