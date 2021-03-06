"""
Starter Keras implementation of an MLP regression model
on Sberbank Russian Housing Market dataset
(https://www.kaggle.com/c/sberbank-russian-housing-market)
- Loss: MSE,
- Optimizer: Adam
- No pre-processing: just drop all non-numeric columns
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from preprocessors import preprocess_csv, split_to_folds

SEED = 1337  # seed for kfold split
NUM_FOLDS = 4  # kfold num splits
BATCH_SIZE = 64
NUM_EPOCHS = 5


def rmsle(y_true, y_pred):
    """ RMS log-error """
    return K.sqrt(K.mean(K.square(tf.math.log1p(y_true) - tf.math.log1p(y_pred))))


def build_dense_model(num_features):
    """ Simple two layer MLP """
    inputs = layers.Input(shape=(num_features,))
    output = layers.Dense(64, activation='relu')(inputs)
    output = layers.Dense(1)(output)
    model = Model(inputs=inputs,
                  outputs=output)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[rmsle])
    return model


def train_fold(model, fold, train_df, test_df):
    train_idx, val_idx = fold
    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_df.iloc[train_idx]['price_doc']

    val_x = train_df.iloc[val_idx].drop('price_doc', axis=1)
    val_y = train_df.iloc[val_idx]['price_doc']

    results = model.fit(x=train_x,
                        y=train_y,
                        batch_size=BATCH_SIZE,
                        validation_data=(val_x, val_y),
                        epochs=NUM_EPOCHS)

    test_pred = model.predict(test_df)

    return model, results.history, test_pred


if __name__ == '__main__':
    preprocess_dict = preprocess_csv()
    (train_ids,
     test_ids,
     processed_train_df,
     processed_test_df) = [preprocess_dict[key] for key in ['train_ids',
                                                            'test_ids',
                                                            'processed_train',
                                                            'processed_test']]

    folds = split_to_folds(processed_train_df, NUM_FOLDS, SEED)
    all_fold_results = []
    all_fold_preds = []
    for curr_fold in folds:
        _, curr_results, curr_pred = train_fold(build_dense_model(num_features=processed_train_df.shape[1]-1),
                                                curr_fold,
                                                processed_train_df,
                                                processed_test_df)

        all_fold_results.append(curr_results)
        all_fold_preds.append(curr_pred)

    print('Val loss: {}'.format(np.mean([x['val_loss'] for x in all_fold_results], axis=0)))
    print('Val RMSLE: {}'.format(np.mean([x['val_rmsle'] for x in all_fold_results], axis=0)))

    mean_pred = np.squeeze(np.mean(np.stack(all_fold_preds), axis=0))
    print(mean_pred.shape)

    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output.csv',
                                                  index=False)
