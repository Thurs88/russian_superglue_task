from typing import Dict, List

import torch
from torch.utils.data import Dataset


class RUSSEDataset(Dataset):
    """
    Custom PyTorch dataset class
    """

    def __init__(self, data: List, tokenizer, **kwarg: Dict):
        self.data = data
        self.tokenizer = tokenizer
        self.labels_map = {
            False: 0,
            True: 1,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sentence1, sentence2, _, _, _, _, label = self.data[idx]
        label = self.labels_map[label]

        sentence1_len = len(self.tokenizer.encode(sentence1))
        sentence2_len = len(self.tokenizer.encode(sentence2))
        segment_ids = torch.tensor([0] * sentence1_len + [1] * sentence2_len)

        encoding = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
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


class RUSSeTestDataset(Dataset):
    """
    Custom PyTorch dataset class
    """

    def __init__(self, data: List, tokenizer, **kwarg: Dict):
        self.data = data
        self.tokenizer = tokenizer
        self.labels_map = {
            False: 0,
            True: 1,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        _, sentence1, sentence2, _, _, _, _, label = self.data[idx]

        sentence1_len = len(self.tokenizer.encode(sentence1))
        sentence2_len = len(self.tokenizer.encode(sentence2))
        segment_ids = torch.tensor([0] * sentence1_len + [1] * sentence2_len)

        encoding = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
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


class RUSSeInferenceDataset(Dataset):
    """
    Custom PyTorch dataset class
    """

    def __init__(self, data: List, tokenizer, **kwarg: Dict):
        self.data = data
        self.tokenizer = tokenizer
        self.labels_map = {
            False: 0,
            True: 1,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sentence1, sentence2 = self.data[idx]
        sentence1_len = len(self.tokenizer.encode(sentence1))
        sentence2_len = len(self.tokenizer.encode(sentence2))
        segment_ids = torch.tensor([0] * sentence1_len + [1] * sentence2_len)

        encoding = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
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
