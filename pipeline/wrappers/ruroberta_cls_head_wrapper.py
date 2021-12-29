from functools import lru_cache

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from src.technical_utils import load_obj


class ruRoBERTaFreezeFineTuner(pl.LightningModule):
    def __init__(self, cfg: DictConfig, train_dataloader_size: int = 100):
        super(ruRoBERTaFreezeFineTuner, self).__init__()
        self.cfg = cfg
        self.train_dataloader_size = train_dataloader_size
        self.model = load_obj(cfg.model.class_name)(cfg)
        if cfg.model.freeze_params:
            print('load model from checkpoint...')
            checkpoint = torch.load(cfg.model.checkpoint)
            self.model.load_state_dict(checkpoint)
            print('freeze backbone...')
            self.freeze_params()

            # set new classifier head
            hidden_size = self.model.model.config.hidden_size
            self.model.classifier = nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(p=self.cfg.model.params.dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(p=self.cfg.model.params.dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(p=self.cfg.model.params.dropout),
                nn.Linear(hidden_size, 1),
            )

        self.criterion = load_obj(cfg.loss.class_name)(**cfg.loss.params)
        self.metrics = nn.ModuleDict(
            {
                self.cfg.metric.metric.metric_name: load_obj(self.cfg.metric.metric.class_name)(
                    **cfg.metric.metric.params
                ).to(self.cfg.general.device)
            }
        )
        if 'other_metrics' in self.cfg.metric.keys():
            for metric in self.cfg.metric.other_metrics:
                self.metrics[metric] = load_obj(metric.class_name)(**metric.params).to(self.cfg.general.device)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def freeze_params(self):
        for param in self.model.model.parameters():
            param.requires_grad = False

    @lru_cache()
    def total_steps(self):
        return (self.train_dataloader_size // self.cfg.trainer.accumulate_grad_batches) * self.cfg.trainer.max_epochs

    def configure_optimizers(self):
        if 'num_warmup_steps' in self.cfg.scheduler.params:
            if not self.cfg.scheduler.params.num_warmup_steps:
                num_warmup_steps = int(self.total_steps() * 0.06)
            else:
                num_warmup_steps = self.cfg.scheduler.params.num_warmup_steps

        if self.cfg.model.freeze_params:
            classifier_optimizer = load_obj(self.cfg.optimizer.class_name)(
                self.model.classifier.parameters(), **self.cfg.optimizer.params
            )
            optimizers = [classifier_optimizer]

            classifier_scheduler = load_obj(self.cfg.scheduler.class_name)(
                classifier_optimizer, **self.cfg.scheduler.params
            )
            schedulers = [
                {
                    'scheduler': classifier_scheduler,
                    'interval': self.cfg.scheduler.step,
                    'monitor': self.cfg.scheduler.monitor,
                }
            ]

        else:
            classifier_optimizer = load_obj(self.cfg.optimizer.class_name)(
                self.model.classifier.parameters(), **self.cfg.optimizer.params.classifier
            )
            backbone_optimizer = load_obj(self.cfg.optimizer.class_name)(
                self.model.model.parameters(), **self.cfg.optimizer.params.backbone
            )
            optimizers = [backbone_optimizer, classifier_optimizer]

            backbone_scheduler = load_obj(self.cfg.scheduler.backbone.class_name)(
                backbone_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps(),
            )
            classifier_scheduler = load_obj(self.cfg.scheduler.classifier.class_name)(
                classifier_optimizer, **self.cfg.scheduler.classifier.params
            )
            schedulers = [
                {
                    'scheduler': backbone_scheduler,
                    'interval': self.cfg.scheduler.backbone.step,
                    'monitor': self.cfg.scheduler.backbone.monitor,
                },
                {
                    'scheduler': classifier_scheduler,
                    'interval': self.cfg.scheduler.classifier.step,
                    'monitor': self.cfg.scheduler.classifier.monitor,
                }
            ]
        return (
            optimizers,
            schedulers,
        )

    def training_step(self, batch, batch_idx, *args):

        labels = batch["labels"].unsqueeze(1)

        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch['token_type_ids'],
        )

        loss = self.criterion(logits, labels.float())
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for metric_name, metric in self.metrics.items():
            score = metric(logits, labels)
            self.log(f"train_{metric_name}", score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        labels = batch["labels"].unsqueeze(1)

        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch['token_type_ids'],
        )

        loss = self.criterion(logits, labels.float())
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for metric_name, metric in self.metrics.items():
            score = metric(logits, labels)
            self.log(f"val_{metric_name}", score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx, optimizer_idx):
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch['token_type_ids'],
        )
        return logits
