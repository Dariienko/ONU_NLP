import torch
import torchtext

from typing import Any

import pandas as pd
import re


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_data: str,
        vectorizer: Any = torchtext.vocab.GloVe(name='6B'),  # https://pytorch.org/text/stable/vocab.html
        max_sequence_length: int=7
    ):
        super().__init__()
        self.path_to_data = path_to_data
        self.data = pd.read_csv(path_to_data)

        self.max_sequence_length = max_sequence_length
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data.iloc[idx] # -> Series

        is_sarctastic = int(item['is_sarcastic'])
        sentence = item.headline

        sentence = sentence.lower().strip()
        sentence_words = re.sub('\?|\!|\.|\,|\;|\'|\"', '', sentence).split(' ')

        zero_vector = torch.zeros_like(self.vectorizer['i'])
        vectors = [self.vectorizer[x] if x in self.vectorizer.stoi else zero_vector for x in sentence_words]

        for _ in range(max(0, self.max_sequence_length - len(vectors))):
            vectors.append(zero_vector)
        vectors = vectors[:self.max_sequence_length]

        vectors = torch.stack(vectors) # list (vectors with shape 100) -> Tensor(n_weords, 100)

        return {
            'class': is_sarctastic,
            'vectors': vectors
        }


    @staticmethod
    def collate_fn(list_of_elements):
        return {
            'class': torch.stack([x['class'] for x in list_of_elements]),
            'vectors': [x['vectors'] for x in list_of_elements]
        }