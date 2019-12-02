"""
Averages out result CSVs found in data/output and places its own output in data/ensemble
"""

import glob
import numpy as np
import pandas as pd


def ensemble_csvs(input_path='data/output/*.csv',
                  output_path='data/ensemble/'):
    all_pds = [pd.read_csv(curr_csv) for curr_csv in glob.glob(input_path)]
    mean_price_doc = np.mean([df['price_doc'] for df in all_pds], axis=0)
    pd.DataFrame({'id':all_pds[0]['id'].values,
                  'price_doc':mean_price_doc}).to_csv(output_path + 'ensemble{}.csv'.format(len(all_pds)),
                                                      index=False)


if __name__ == '__main__':
    ensemble_csvs()