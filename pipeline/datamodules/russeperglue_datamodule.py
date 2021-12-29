from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from transformers import DataCollatorWithPadding

from src.technical_utils import load_obj
from src.text_utils import prepare_data


class RusSuperGLUEDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_obj(cfg.datamodule.tokenizer).from_pretrained(cfg.datamodule.pretrained_tokenizer)
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding='longest',
        )
        self.train_data: List = []
        self.val_data: List = []
        self.test_data: List = []

    def prepare_data(self):
        # load datasets
        self.train_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.data_path + 'train.jsonl',
            lines=True
        ).set_index('idx')
        self.val_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.data_path + 'val.jsonl',
            lines=True
        ).set_index('idx')
        self.test_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.data_path + 'test.jsonl',
            lines=True
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # train_data = prepare_data(self.train_data)
            # val_data = prepare_data(self.val_data)
            dataset_class = load_obj(self.cfg.datamodule.class_name)
            self.train_dataset = dataset_class(
                data=self.train_data.values,
                tokenizer=self.tokenizer,
            )
            self.valid_dataset = dataset_class(
                data=self.val_data.values,
                tokenizer=self.tokenizer,
            )

        if stage == "test":
            dataset_class = load_obj(self.cfg.test.dataset_class)
            # test_data = prepare_data(self.test_data)
            self.test_dataset = dataset_class(
                data=self.test_data.values,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )
        return test_loader
