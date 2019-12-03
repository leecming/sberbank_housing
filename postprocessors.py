""" Post-processing e.g., ensembling, pseudo-label generation"""
import os
import glob
from functools import reduce
import numpy as np
import pandas as pd

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
MACRO_PATH = 'data/macro.csv'


def ensemble_csvs(input_path='data/output/*.csv',
                  output_path='data/ensemble/'):
    """
    Averages out result CSVs found in data/output and places its own output in data/ensemble
    """
    all_pds = [pd.read_csv(curr_csv) for curr_csv in glob.glob(input_path)]
    mean_price_doc = np.mean([df['price_doc'] for df in all_pds], axis=0)
    pd.DataFrame({'id': all_pds[0]['id'].values,
                  'price_doc': mean_price_doc}).to_csv(output_path + 'ensemble{}.csv'.format(len(all_pds)),
                                                       index=False)


def combine_stacking_inputs(output_path='data/'):
    """
    Combines train and test stack inputs with original train csv
    into an output stacking inputs for a stacking model to feed in
    """
    # make sure train and test files match exactly
    assert [x[2] for x in os.walk('data/stacking_input/train/')] \
           == [x[2] for x in os.walk('data/stacking_input/test/')]

    # combine train inputs
    train_df = pd.read_csv(TRAIN_PATH).set_index('id')
    train_stack_inputs = [pd.read_csv(x).drop('price_doc', axis=1).set_index('id')
                          for x in glob.glob('data/stacking_input/train/*.csv')]
    train_out_df = reduce(lambda left, right: left.join(right), train_stack_inputs, train_df)
    train_out_df.reset_index().to_csv(output_path + 'stacking_train_{}.csv'.format(len(train_stack_inputs)))

    # combine test_inputs
    test_df = pd.read_csv(TEST_PATH).set_index('id')
    test_stack_inputs = [pd.read_csv(x).set_index('id')
                         for x in glob.glob('data/stacking_input/test/*.csv')]
    test_out_df = reduce(lambda left, right: left.join(right), test_stack_inputs, test_df)
    test_out_df.reset_index().to_csv(output_path + 'stacking_test_{}.csv'.format(len(test_stack_inputs)))
    assert [x for x in train_out_df.columns if x != 'price_doc'] == [x for x in test_out_df.columns]


def generate_pseudolabels(ensemble_path=None,
                          output='data/pseudolabels.csv'):
    """ Generate pseudo labels, combining test set predictions with train set """
    test_labels = pd.read_csv(ensemble_path)
    test_df = pd.read_csv(TEST_PATH)
    test_df['price_doc'] = test_labels['price_doc']
    train_df = pd.read_csv(TRAIN_PATH)
    train_df.append(test_df).to_csv(output, index=False)


def generate_stacking_inputs(filename,
                             oof_indices,
                             oof_preds,
                             test_preds,
                             train_ids,
                             test_ids,
                             train_df):
    """
    Given oof predictions and meaned-test predictions, write train and test stacking inputs
    - prediction columns are prefixed by filename
    """
    oof_df = pd.DataFrame(oof_preds)
    output_columns = [filename + '_' + str(x) for x in oof_df.columns.values]
    oof_df.columns = output_columns
    oof_df['id'] = train_ids[oof_indices]
    oof_df['price_doc'] = train_df.iloc[oof_indices]['price_doc']
    oof_df = oof_df[['id', 'price_doc'] + output_columns]
    oof_df.sort_values(by='id').to_csv('data/stacking_input/train/{}.csv'.format(filename), index=False)
    test_df = pd.DataFrame(test_preds)
    test_df.columns = [filename + '_' + str(x) for x in test_df.columns.values]
    test_df['id'] = test_ids
    test_df = test_df[['id'] + output_columns]
    test_df.to_csv('data/stacking_input/test/{}.csv'.format(filename), index=False)


if __name__ == '__main__':
    combine_stacking_inputs()
