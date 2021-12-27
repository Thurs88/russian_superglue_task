from functools import lru_cache

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch import nn

from src.technical_utils import load_obj


class ruRoBERTaFineTuner(pl.LightningModule):
    def __init__(self, cfg: DictConfig, train_dataloader_size: int = 100):
        super(ruRoBERTaFineTuner, self).__init__()
        self.cfg = cfg
        self.train_dataloader_size = train_dataloader_size
        self.model = load_obj(cfg.model.class_name)(cfg)
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

    @lru_cache()
    def total_steps(self):
        return (self.train_dataloader_size // self.cfg.trainer.accumulate_grad_batches) * self.cfg.trainer.max_epochs

    def configure_optimizers(self):
        if not self.cfg.scheduler.params.num_warmup_steps:
            num_warmup_steps = int(self.total_steps() * 0.06)
        else:
            num_warmup_steps = self.cfg.scheduler.backbone.params.num_warmup_steps

        optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps(),
        )

        return (
            [optimizer],
            [{'scheduler': scheduler, 'interval': self.cfg.scheduler.step, 'monitor': self.cfg.scheduler.monitor}],
        )

    def training_step(self, batch, batch_idx):

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
