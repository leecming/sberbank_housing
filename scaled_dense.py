"""
Modified version of basic_dense -
1. StandardScaler on all features
2. log1p on output
3. BatchNorm
4. EMA on model weights
5. parallel processing using mp where num_cores = num_folds
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from preprocessors import preprocess_csv
from keras_util import ExponentialMovingAverage
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 1337  # seed for k-fold split
NUM_FOLDS = 4  # k-fold num splits
BATCH_SIZE = 64
NUM_EPOCHS = 100


def split_to_folds(input_df):
    """ Split input df into kfolds returning (train, val) index tuples """
    kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [x for x in kf.split(input_df)]


def build_dense_model():
    """ Simple two layer MLP """
    inputs = layers.Input(shape=(277,))
    output = layers.GaussianDropout(0.1)(inputs)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Dropout(0.1)(output)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Dense(1)(output)
    model = Model(inputs=inputs,
                  outputs=output)

    model.compile(optimizer=Adam(),
                  loss='mean_squared_error')
    return model


def train_fold(fold, train_df, test_df):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_df.iloc[train_idx]['price_doc']
    train_y = np.log1p(train_y)

    val_x = train_df.iloc[val_idx].drop('price_doc', axis=1)
    val_y = train_df.iloc[val_idx]['price_doc']
    val_y = np.log1p(val_y)

    std_scaler = StandardScaler().fit(train_x)
    train_x = std_scaler.transform(train_x)
    val_x = std_scaler.transform(val_x)

    model = build_dense_model()
    results = model.fit(x=train_x,
                        y=train_y,
                        batch_size=BATCH_SIZE,
                        validation_data=(val_x, val_y),
                        epochs=NUM_EPOCHS,
                        verbose=0,
                        callbacks=[ExponentialMovingAverage()])

    test_df = std_scaler.transform(test_df)
    test_pred = np.expm1(model.predict(test_df))

    return results.history, test_pred


if __name__ == '__main__':
    train_ids, test_ids, processed_train_df, processed_test_df = preprocess_csv()

    folds = split_to_folds(processed_train_df)
    all_fold_results = []
    all_fold_preds = []

    with Pool(NUM_FOLDS) as p:
        combined_results = p.starmap(train_fold, ((curr_fold,
                                                   processed_train_df,
                                                   processed_test_df) for curr_fold in folds))

    print('Loss: {}'.format(np.mean([x[0]['loss'] for x in combined_results], axis=0)))
    print('Val loss: {}'.format(np.mean([x[0]['val_loss'] for x in combined_results], axis=0)))
    print('Val RMSLE: {}'.format(np.sqrt(np.mean([x[0]['val_loss'] for x in combined_results], axis=0))))
    mean_pred = np.squeeze(np.mean(np.stack([x[1] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/scaled_dense_output.csv',
                                                  index=False)
