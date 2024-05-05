import os

import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

from config import CFG


class Preprocessing:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")
        self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base",
                                               config=self.config)
        self.file_name = 'data/gap-development.tsv'

    def test(self):
        encoder = self.tokenizer(['Ducaa is a superman'], max_length=CFG.max_length, truncation=True,
                                 return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoder["input_ids"][0])
        print(encoder, tokens)

    def encode(self, row):
        encoder = self.tokenizer([row['Text']], max_length=CFG.max_length, truncation=True,
                                 return_offsets_mapping=True)
        offset_mapping = encoder['offset_mapping'][0]
        inputs = {k: torch.tensor(v) for k, v in encoder.items() if k != 'offset_mapping'}
        with torch.no_grad():
            outputs = self.model(**inputs)
        row['A_feature'] = torch.zeros(768)
        row['B_feature'] = torch.zeros(768)
        row['P_feature'] = torch.zeros(768)

        a_offset = [max(0, row['A-offset'] - 1), row['A-offset'] + len(row['A'])]
        b_offset = [max(0, row['B-offset'] - 1), row['B-offset'] + len(row['B'])]
        p_offset = [max(0, row['Pronoun-offset'] - 1), row['Pronoun-offset'] + len(row['Pronoun'])]
        last_hidden_state = outputs[0]  # (1, L, 768)
        features = last_hidden_state[0]  # (L, 768)
        for idx, feature in enumerate(features):
            if a_offset[0] <= offset_mapping[idx][0] and a_offset[1] >= offset_mapping[idx][1]:
                row['A_feature'] += feature

            if b_offset[0] <= offset_mapping[idx][0] and b_offset[1] >= offset_mapping[idx][1]:
                row['B_feature'] += feature

            if p_offset[0] <= offset_mapping[idx][0] and p_offset[1] >= offset_mapping[idx][1]:
                row['P_feature'] += feature

        row['A_feature'] = row['A_feature'].numpy().tolist()
        row['B_feature'] = row['B_feature'].numpy().tolist()
        row['P_feature'] = row['P_feature'].numpy().tolist()
        return row

    def preprocess(self):
        df = pd.read_csv(self.file_name, sep='\t')
        df = df.apply(self.encode, axis=1)
        df = df.drop(columns=['Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL'])
        file_name = os.path.join('data', 'feature_extract.csv')
        df.to_csv(file_name, sep='\t', index=False)


preprocessing = Preprocessing()
preprocessing.preprocess()
