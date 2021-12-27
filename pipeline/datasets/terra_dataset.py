from typing import Dict, List

import torch
from torch.utils.data import Dataset


class TERRaDataset(Dataset):
    """
    Custom PyTorch dataset class
    """

    def __init__(self, data: List, tokenizer, **kwarg: Dict):
        self.data = data
        self.tokenizer = tokenizer
        self.labels_map = {
            "not_entailment": 0,
            "entailment": 1,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        premise = self.data[idx][0]
        hypothesis = self.data[idx][1]
        label = self.labels_map[self.data[idx][-1]]

        premise_len = len(self.tokenizer.encode(premise))
        hypothesis_len = len(self.tokenizer.encode(hypothesis))
        segment_ids = torch.tensor([0] * premise_len + [1] * hypothesis_len)

        encoding = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids, attention_mask = encoding.input_ids.squeeze(), encoding.attention_mask.squeeze()

        label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": segment_ids,
            "label": label,
        }


class TERRaTestDataset(Dataset):
    """
    Custom PyTorch dataset class
    """

    def __init__(self, data: List, tokenizer, **kwarg: Dict):
        self.data = data
        self.tokenizer = tokenizer
        self.labels_map = {
            "not_entailment": 0,
            "entailment": 1,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        premise = self.data[idx][0]
        hypothesis = self.data[idx][1]

        premise_len = len(self.tokenizer.encode(premise))
        hypothesis_len = len(self.tokenizer.encode(hypothesis))
        segment_ids = torch.tensor([0] * premise_len + [1] * hypothesis_len)

        encoding = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids, attention_mask = encoding.input_ids.squeeze(), encoding.attention_mask.squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": segment_ids,
        }
