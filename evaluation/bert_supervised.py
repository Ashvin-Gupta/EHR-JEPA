"""BERT encoder + configurable head for supervised binary classification."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from evaluation.supervised_perceiver import build_classification_head
from training.bert_trainer import BERTEHRModel

HeadType = Literal["linear", "mlp"]


class BERTSupervisedClassifier(nn.Module):
    """CLS embedding → linear or small MLP → logit (BCEWithLogits)."""

    def __init__(
        self,
        bert: BERTEHRModel,
        head_type: HeadType = "linear",
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bert = bert
        d_model = bert.output_dim
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
        cls_emb = self.bert.encode_cls_embedding(
            codes,
            attention_mask,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask,
        )
        return self.head(cls_emb).squeeze(-1)
