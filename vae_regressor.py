"""
VAE Regression model based on :- https://arxiv.org/abs/1904.05948
- Modified VAE encoder-decoder with regression (hist-loss) head in the encoder
- With encoder encoding to a latent that's conditioned on the regression target
- Both encoder/decoder are MLP - Dense(64)->Dense(64)
- Combined loss: KL divergence (target not 0,1 but conditioned on regression) + reconstruction loss + regression loss
"""
import os
import time
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, categorical_crossentropy
from tensorflow.keras import constraints, layers
import tensorflow.keras.backend as K
from preprocessors import preprocess_csv, split_to_folds, generate_target_dist
from postprocessors import generate_stacking_inputs
from keras_util import ExponentialMovingAverage
from multiprocessing import Pool

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 1337  # seed for k-fold split
NUM_FOLDS = 8  # k-fold num splits
BATCH_SIZE = 64
LATENT_DIM = 8
NUM_EPOCHS = 100


def sampling(args):
    """Re-parameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments: args (tensor): mean and log of variance of Q(z|X)
    # Returns: z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_dense_model(num_features):
    """ Simple two layer MLP """
    regression_target = layers.Input(shape=(1,), name='ground_truth')
    feature_input = layers.Input(shape=(num_features,), name='feature_input')
    encoder_output = layers.GaussianDropout(0.1)(feature_input)
    encoder_output = layers.Dense(64, activation='tanh', name='encoder_hidden')(encoder_output)
    encoder_output = layers.Dense(64, activation='tanh', name='encoder_hidden_2')(encoder_output)
    z_mean, z_log_var = layers.Dense(LATENT_DIM, name='z_mean')(encoder_output), \
                        layers.Dense(LATENT_DIM, name='z_log_var')(encoder_output)
    r_mean, r_log_var = layers.Dense(1, name='r_mean')(encoder_output), \
                        layers.Dense(1, name='r_log_var')(encoder_output)

    # Sample latent and regression target
    z = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])
    r = layers.Lambda(sampling, output_shape=(1,), name='r')([r_mean, r_log_var])

    # Latent generator
    pz_mean = layers.Dense(LATENT_DIM,
                           kernel_constraint=constraints.unit_norm(),
                           name='pz_mean')(r)

    encoder = Model([feature_input, regression_target],
                    [z_mean, z_log_var, z,
                     r_mean, r_log_var, r,
                     pz_mean],
                    name='encoder')

    latent_input = layers.Input(shape=(LATENT_DIM,), name='decoder_input')
    decoder_output = layers.Dense(64, activation='tanh', name='decoder_hidden')(latent_input)
    decoder_output = layers.Dense(64, activation='tanh', name='decoder_hidden_2')(decoder_output)
    decoder_output = layers.Dense(num_features, name='decoder_output')(decoder_output)

    decoder = Model(latent_input, decoder_output, name='decoder')

    encoder_decoder_output = decoder(encoder([feature_input, regression_target])[2])
    vae = Model([feature_input, regression_target], encoder_decoder_output, name='vae')

    # Manually write up losses
    reconstruction_loss = mse(feature_input, encoder_decoder_output)
    kl_loss = 1 + z_log_var - K.square(z_mean - pz_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    label_loss = tf.divide(0.5 * K.square(r_mean - regression_target), K.exp(r_log_var)) + 0.5 * r_log_var
    vae_loss = K.mean(reconstruction_loss + kl_loss + label_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam())

    regressor = Model(feature_input, r_mean, name='regressor')

    return vae, regressor


def train_fold(fold,
               train_df,
               test_df):
    train_idx, val_idx = fold
    train_df.drop([col for col in train_df.columns if 'median_' in col], axis=1, inplace=True)
    test_df.drop([col for col in test_df.columns if 'median_' in col], axis=1, inplace=True)

    train_x = train_df.iloc[train_idx].drop('price_doc', axis=1)
    train_y = train_df.iloc[train_idx]['price_doc']
    train_y = np.log1p(train_y).values

    val_x = train_df.iloc[val_idx].drop('price_doc', axis=1)
    val_y = train_df.iloc[val_idx]['price_doc']
    val_y = np.log1p(val_y).values

    std_scaler = StandardScaler().fit(train_x)
    train_x = std_scaler.transform(train_x)
    val_x = std_scaler.transform(val_x)

    vae_model, regressor_model = build_dense_model(num_features=train_x.shape[-1])
    vae_model.fit([train_x, train_y],
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  verbose=0,
                  callbacks=[ExponentialMovingAverage()])

    test_df = std_scaler.transform(test_df)
    # raw_val_prob = model.predict(val_x)
    raw_test_prob = regressor_model.predict(test_df)
    test_pred = np.expm1(raw_test_prob)

    return None, test_pred, val_idx, None, raw_test_prob
    # return results.history, test_pred, val_idx, raw_val_prob, raw_test_prob


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

    # losses = np.mean([x[0]['loss'] for x in combined_results], axis=0)
    # val_losses = np.mean([x[0]['val_loss'] for x in combined_results], axis=0)
    # print('Loss: {}'.format(losses))
    # print('Val loss: {}'.format(val_losses))
    # plt.plot(losses, label='train_losses')
    # plt.plot(val_losses, label='val_losses')
    # plt.title('Scaled Dense')
    # plt.legend(loc='best')
    # plt.show()
    #
    mean_pred = np.squeeze(np.mean(np.stack([x[1] for x in combined_results]), axis=0))
    pd.DataFrame({'id': test_ids,
                  'price_doc': mean_pred}).to_csv('data/output/vae_regressor_output.csv',
                                                  index=False)
    #
    # generate_stacking_inputs(filename='scaled_dense_input',
    #                          oof_indices=np.concatenate([x[2] for x in combined_results]),
    #                          oof_preds=np.concatenate([x[3] for x in combined_results]),
    #                          test_preds=np.squeeze(np.mean(np.stack([x[4] for x in combined_results]), axis=0)),
    #                          train_ids=train_ids,
    #                          test_ids=test_ids,
    #                          train_df=processed_train_df)

    print('Elapsed time: {}'.format(time.time() - start_time))
