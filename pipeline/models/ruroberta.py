import torch
from omegaconf import DictConfig
from torch import nn
from transformers import RobertaModel


class ruRoBERTa(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ruRoBERTa, self).__init__()
        self.cfg = cfg
        self.model = RobertaModel.from_pretrained(cfg.model.pretrained_model)

        self.add_token_type_embeddings()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.model.config.hidden_size),
            nn.Dropout(p=self.cfg.model.params.dropout),
            # nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.model.config.hidden_size),
            # nn.Dropout(p=self.cfg.model.params.dropout),
            # nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.model.config.hidden_size),
            # nn.Dropout(p=self.cfg.model.params.dropout),
            nn.Linear(self.model.config.hidden_size, 1),
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

    def add_token_type_embeddings(self, num_additional_embeddings=2, scale=0.5):
        token_type_embedding = self.model.embeddings.token_type_embeddings.weight.data
        noise = torch.normal(
            token_type_embedding.mean() * scale,
            token_type_embedding.std() * scale,
            (num_additional_embeddings, token_type_embedding.size(-1)),
        )
        token_type_embeddings = torch.cat((token_type_embedding, token_type_embedding + noise))
        self.model.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(token_type_embeddings, freeze=False)
        return None
