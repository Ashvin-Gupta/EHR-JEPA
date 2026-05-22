"""AR encoder + configurable head for supervised binary classification."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from evaluation.supervised_perceiver import build_classification_head
from models.sequence_pooling import SequencePoolingMode, parse_pooling_mode
from training.ar_trainer import AREHRModel

HeadType = Literal["linear", "mlp"]


class ARSupervisedClassifier(nn.Module):
    """Pooled AR sequence embedding → linear or MLP → logit (BCEWithLogits)."""

    def __init__(
        self,
        ar: AREHRModel,
        head_type: HeadType = "linear",
        head_dropout: float = 0.1,
        pooling_mode: SequencePoolingMode = "cls",
    ) -> None:
        super().__init__()
        self.ar = ar
        self.pooling_mode = parse_pooling_mode(pooling_mode)
        d_model = ar.output_dim
        self.head = build_classification_head(
            d_model, d_model, head_type, head_dropout
        )

    def forward(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.ar.encode_pooled_embedding(
            codes,
            attention_mask,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask,
            pooling_mode=self.pooling_mode,
        )
        return self.head(pooled).squeeze(-1)
