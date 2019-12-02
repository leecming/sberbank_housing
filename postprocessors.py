""" Post-processing e.g., ensembling, pseudo-label generation"""
import glob
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


def generate_pseudolabels(ensemble_path=None,
                          output='data/pseudolabels.csv'):
    """ Generate pseudo labels, combining test set predictions with train set """
    test_labels = pd.read_csv(ensemble_path)
    test_df = pd.read_csv(TEST_PATH)
    test_df['price_doc'] = test_labels['price_doc']
    train_df = pd.read_csv(TRAIN_PATH)
    train_df.append(test_df).to_csv(output, index=False)


if __name__ == '__main__':
    # generate_pseudolabels(ensemble_path='data/ensemble/ensemble8.csv')
    ensemble_csvs()
