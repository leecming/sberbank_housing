"""
SVR (kernel=RBF) -
1. StandardScaler on all features
2. log1p on output
3. parallel processing using mp where num_cores = num_folds
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from preprocessors import preprocess_csv, split_to_folds
from postprocessors import generate_stacking_inputs
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 1337  # seed for k-fold split
NUM_FOLDS = 8  # k-fold num splits

def train_fold(fold, train_df, test_df):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_df.iloc[train_idx]['price_doc']
    train_y = np.log1p(train_y).values

    val_x = train_df.iloc[val_idx].drop('price_doc', axis=1)
    val_y = train_df.iloc[val_idx]['price_doc']
    val_y = np.log1p(val_y).values

    std_scaler = StandardScaler().fit(train_x)
    train_x = std_scaler.transform(train_x)
    val_x = std_scaler.transform(val_x)

    model = SVR(kernel='rbf',
                C=0.1,
                cache_size=2000)
    model.fit(train_x,
              train_y)
    test_df = std_scaler.transform(test_df)
    raw_val_prob = model.predict(val_x)
    val_rmsle = (np.mean((raw_val_prob - val_y)**2))**0.5
    raw_test_prob = model.predict(test_df)
    test_pred = np.expm1(raw_test_prob)

    return val_rmsle, test_pred, val_idx, raw_val_prob, raw_test_prob


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
    all_fold_results = []
    all_fold_preds = []

    with Pool(NUM_FOLDS) as p:
        combined_results = p.starmap(train_fold, ((curr_fold,
                                                   processed_train_df,
                                                   processed_test_df) for curr_fold in folds))

    val_losses = np.mean([x[0] for x in combined_results])
    print('Mean val loss: {}'.format(val_losses))

    mean_pred = np.squeeze(np.mean(np.stack([x[1] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/scaled_trad_output.csv',
                                                  index=False)

    generate_stacking_inputs(filename='scaled_trad_input',
                             oof_indices=np.concatenate([x[2] for x in combined_results]),
                             oof_preds=np.concatenate([x[3] for x in combined_results]),
                             test_preds=np.squeeze(np.mean(np.stack([x[4] for x in combined_results]), axis=0)),
                             train_ids=train_ids,
                             test_ids=test_ids,
                             train_df=processed_train_df)

    print('Elapsed time: {}'.format(time.time() - start_time))