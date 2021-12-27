from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.1,
        use_kl_div: bool = False,
        ignore_index: int = 0,
        reduce: bool = True,
    ):
        super().__init__()

        assert 0 < smoothing < 1

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.use_kl_div = use_kl_div
        self.reduce = reduce

    def smooth_one_hot(self, true_labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:

        confidence = 1.0 - self.smoothing

        with torch.no_grad():
            true_dist = torch.empty(
                size=(
                    true_labels.size(0),
                    num_classes,
                ),
                device=true_labels.device,
            )
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(
                1,
                true_labels.data.unsqueeze(1),
                confidence,
            )

        return true_dist

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        :param logits: [batch_size, num_classes]
        :param targets: [batch_size]
        :param mask: [batch_size] True if need
        :return: scalar
        """

        # logits = F.log_softmax(logits, dim=-1, dtype=torch.float32)

        targets_smoothed_dist = self.smooth_one_hot(targets, num_classes=2)

        if self.use_kl_div:
            loss = -F.kl_div(
                logits,
                targets_smoothed_dist,
                reduction="none",
            ).sum(dim=-1)
        else:
            loss = torch.sum(
                targets_smoothed_dist * logits,
                dim=-1,
            )

        if self.reduce:
            loss = loss.mean()

        return loss
