"""
Modified version of basic_dense -
1. StandardScaler on all features
2. log1p on output
3. BatchNorm
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from preprocessors import preprocess_csv
from keras_util import ExponentialMovingAverage

SEED = 1337  # seed for kfold split
NUM_FOLDS = 4  # kfold num splits
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
BATCH_SIZE = 64
NUM_EPOCHS = 30


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


def train_fold(model, fold, train_df, test_df):
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

    results = model.fit(x=train_x,
                        y=train_y,
                        batch_size=BATCH_SIZE,
                        validation_data=(val_x, val_y),
                        epochs=NUM_EPOCHS,
                        callbacks=[ExponentialMovingAverage()])

    test_df = std_scaler.transform(test_df)
    test_pred = np.expm1(model.predict(test_df))

    return model, results.history, test_pred


if __name__ == '__main__':
    raw_train_df = pd.read_csv(TRAIN_PATH)
    raw_test_df = pd.read_csv(TEST_PATH)

    processed_train_df, processed_test_df = preprocess_csv(raw_train_df, raw_test_df)

    assert raw_train_df.shape[0] == processed_train_df.shape[0]
    assert raw_test_df.shape[0] == processed_test_df.shape[0]

    folds = split_to_folds(processed_train_df)
    all_fold_results = []
    all_fold_preds = []
    for curr_fold in folds:
        _, curr_results, curr_pred = train_fold(build_dense_model(),
                                                curr_fold,
                                                processed_train_df,
                                                processed_test_df)

        all_fold_results.append(curr_results)
        all_fold_preds.append(curr_pred)

    print('Loss: {}'.format(np.mean([x['loss'] for x in all_fold_results], axis=0)))
    print('Val loss: {}'.format(np.mean([x['val_loss'] for x in all_fold_results], axis=0)))
    print('Val RMSLE: {}'.format(np.sqrt(np.mean([x['val_loss'] for x in all_fold_results], axis=0))))

    mean_pred = np.squeeze(np.mean(np.stack(all_fold_preds), axis=0))
    print(mean_pred.shape)

    pd.DataFrame({'id': raw_test_df['id'],
                  'price_doc': mean_pred}).to_csv('data/scaled_dense_output.csv',
                                                  index=False)
