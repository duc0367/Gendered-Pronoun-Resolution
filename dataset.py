import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from ast import literal_eval

LABEL_TO_ID = {
    'A': 0,
    'B': 1,
    'NEITHER': 2
}


class GAPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = 'train'):
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]
        a_feature = np.array(literal_eval(item['A_feature']))
        b_feature = np.array(literal_eval(item['B_feature']))
        p_feature = np.array(literal_eval(item['P_feature']))
        inputs = np.concatenate([p_feature, a_feature, b_feature])

        assert inputs.shape == (3*768,)

        if self.mode != 'train':
            return inputs, None

        label = LABEL_TO_ID['NEITHER']
        if item['A-coref']:
            label = LABEL_TO_ID['A']
        elif item['B-coref']:
            label = LABEL_TO_ID['B']

        return inputs, label
