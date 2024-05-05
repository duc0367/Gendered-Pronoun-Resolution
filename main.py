import pandas as pd
from sklearn.model_selection import KFold

from config import CFG
from trainer import train_with_fold

kf = KFold(n_splits=CFG.n_split)

dataframe = pd.read_csv('data/feature_extract.csv', sep='\t')

for i, (train_index, test_index) in enumerate(kf.split(dataframe)):
    dataframe.loc[test_index, 'fold'] = i

for i in range(CFG.n_split):
    train_df = dataframe[dataframe['fold'] != i]
    val_df = dataframe[dataframe['fold'] == i]
    train_with_fold(i, train_df, val_df)
