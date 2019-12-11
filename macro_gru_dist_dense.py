"""
Modified version of the Distributional dense model (distributional_dense.py)
that has a second input of a sliding window of macro data
- uses BiGRU to interpret the macro sliding window data
"""
import os
import time
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from preprocessors import preprocess_csv, split_to_folds, generate_target_dist
from keras_util import ExponentialMovingAverage
from postprocessors import generate_stacking_inputs
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 1337  # seed for k-fold split
NUM_FOLDS = 4  # k-fold num splits
BATCH_SIZE = 64
NUM_EPOCHS = 30
LOW = 11  # lowest training price (log1p) = 11.51
HIGH = 19  # highest training price = 18.53
NUM_BINS = 10


def build_dense_model():
    """ Simple two layer MLP """
    inputs_1 = layers.Input(shape=(277,))
    output_1 = layers.GaussianDropout(0.1)(inputs_1)

    inputs_2 = layers.Input(shape=(None, 7))
    output_2 = layers.Bidirectional(layers.GRU(5))(inputs_2)

    output = layers.concatenate([output_1, output_2])
    output = layers.Dense(64, activation='relu')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Dropout(0.1)(output)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.BatchNormalization()(output)
    output = layers.Dense(NUM_BINS, activation='softmax')(output)
    model = Model(inputs=[inputs_1, inputs_2],
                  outputs=output)

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy')
    return model


def train_fold(fold,
               train_df,
               test_df,
               train_labels,
               supports,
               train_rolling,
               test_rolling):
    train_idx, val_idx = fold
    train_x_1 = train_df.iloc[train_idx].drop('price_doc', axis=1).astype('float32')
    train_x_2 = train_rolling[train_idx].astype('float32')
    train_y = train_labels[train_idx]

    val_x_1 = train_df.iloc[val_idx].drop('price_doc', axis=1).astype('float32')
    val_x_2 = train_rolling[val_idx].astype('float32')
    val_y = train_labels[val_idx]

    std_scaler = StandardScaler().fit(train_x_1)
    train_x_1 = std_scaler.transform(train_x_1)
    val_x_1 = std_scaler.transform(val_x_1)

    model = build_dense_model()
    results = model.fit(x=[train_x_1, train_x_2],
                        y=train_y,
                        batch_size=BATCH_SIZE,
                        validation_data=([val_x_1, val_x_2], val_y),
                        epochs=NUM_EPOCHS,
                        verbose=0,
                        callbacks=[ExponentialMovingAverage()])

    test_df_1 = std_scaler.transform(test_df)
    raw_val_prob = model.predict([val_x_1, val_x_2])
    raw_test_prob = model.predict([test_df_1.astype('float32'),
                                   test_rolling.astype('float32')])
    test_pred = np.expm1(np.dot(raw_test_prob, supports))

    return results.history, test_pred, val_idx, raw_val_prob, raw_test_prob


if __name__ == '__main__':
    start_time = time.time()
    preprocess_dict = preprocess_csv(rolling_macro={'min_unique': 100,
                                                    'lookback_period': 100})
    (train_ids,
     test_ids,
     processed_train_df,
     processed_test_df,
     train_rolling,
     test_rolling) = [preprocess_dict[key] for key in ['train_ids',
                                                       'test_ids',
                                                       'processed_train',
                                                       'processed_test',
                                                       'train_rolling',
                                                       'test_rolling']]
    # generate distribution labels
    generate_target_partial = partial(generate_target_dist, num_bins=NUM_BINS, low=LOW, high=HIGH)
    with Pool(8) as p:
        train_labels = np.stack([x[1] for x in p.map(generate_target_partial,
                                                     np.log1p(processed_train_df['price_doc']))])
    supports = generate_target_partial(-1)[0]

    folds = split_to_folds(processed_train_df,
                           num_folds=NUM_FOLDS,
                           seed=SEED,
                           shuffle=False)
    all_fold_results = []
    all_fold_preds = []

    with Pool(NUM_FOLDS) as p:
        combined_results = p.starmap(train_fold, ((curr_fold,
                                                   processed_train_df,
                                                   processed_test_df,
                                                   train_labels,
                                                   supports,
                                                   train_rolling,
                                                   test_rolling) for curr_fold in folds))

    losses = np.mean([x[0]['loss'] for x in combined_results], axis=0)
    val_losses = np.mean([x[0]['val_loss'] for x in combined_results], axis=0)
    print('Loss: {}'.format(losses))
    print('Val loss: {}'.format(val_losses))
    plt.plot(losses, label='train_losses')
    plt.plot(val_losses, label='val_losses')
    plt.title('Distributional GRU Dense + macros')
    plt.legend(loc='best')
    plt.show()

    mean_pred = np.squeeze(np.mean(np.stack([x[1] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/macro_gru_dist_dense_output.csv',
                                                  index=False)

    generate_stacking_inputs(filename='macro_gru_dist_dense_output',
                             oof_indices=np.concatenate([x[2] for x in combined_results]),
                             oof_preds=np.concatenate([x[3] for x in combined_results]),
                             test_preds=np.squeeze(np.mean(np.stack([x[4] for x in combined_results]), axis=0)),
                             train_ids=train_ids,
                             test_ids=test_ids,
                             train_df=processed_train_df)

    print('Elapsed time: {}'.format(time.time() - start_time))