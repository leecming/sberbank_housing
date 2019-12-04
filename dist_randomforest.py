"""
Apply multi-output Random Forest regression on the distributional targets
"""
import time
from functools import partial
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from preprocessors import preprocess_csv, split_to_folds, generate_target_dist
from postprocessors import generate_stacking_inputs
from multiprocessing import Pool


SEED = 1337  # seed for k-fold split
NUM_FOLDS = 4  # k-fold num splits
LOW = 11  # lowest training price (log1p) = 11.51
HIGH = 19  # highest training price = 18.53
NUM_BINS = 10


def train_fold(fold, train_df, test_df, train_labels, supports):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_labels[train_idx]

    val_x = train_df.iloc[val_idx].drop('price_doc', axis=1)
    val_y = train_labels[val_idx]

    model = RandomForestRegressor(n_estimators=300,
                                  n_jobs=8)
    model.fit(train_x, train_y)

    raw_val_prob = model.predict(val_x)
    raw_test_prob = model.predict(test_df)

    test_pred = np.expm1(np.dot(raw_test_prob, supports))

    return test_pred, val_idx, raw_val_prob, raw_test_prob


if __name__ == '__main__':
    start_time = time.time()
    train_ids, test_ids, processed_train_df, processed_test_df = preprocess_csv()

    # generate distribution labels
    generate_target_partial = partial(generate_target_dist, num_bins=NUM_BINS, low=LOW, high=HIGH)
    with Pool(8) as p:
        train_labels = np.stack([x[1] for x in p.map(generate_target_partial,
                                                     np.log1p(processed_train_df['price_doc']))])
    supports = generate_target_partial(-1)[0]

    folds = split_to_folds(processed_train_df, NUM_FOLDS, SEED)
    all_fold_results = []
    all_fold_preds = []

    with Pool(NUM_FOLDS) as p:
        combined_results = p.starmap(train_fold, ((curr_fold,
                                                   processed_train_df,
                                                   processed_test_df,
                                                   train_labels,
                                                   supports) for curr_fold in folds))

    mean_pred = np.squeeze(np.mean(np.stack([x[0] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/rfr_dist_output.csv',
                                                  index=False)

    generate_stacking_inputs(filename='rfr_dist_input',
                             oof_indices=np.concatenate([x[1] for x in combined_results]),
                             oof_preds=np.concatenate([x[2] for x in combined_results]),
                             test_preds=np.squeeze(np.mean(np.stack([x[3] for x in combined_results]), axis=0)),
                             train_ids=train_ids,
                             test_ids=test_ids,
                             train_df=processed_train_df)

    print('Elapsed time: {}'.format(time.time() - start_time))
