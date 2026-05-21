"""Supervised binary classification: embedding → encoder → [CLS] → head."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from evaluation.supervised_perceiver import HeadType, build_classification_head
from models.cls_encoding import encode_cls_from_batch
from models.event_embedding import EventEmbedding
from models.transformer_encoder import EHRTransformerEncoder


class SupervisedCLSClassifier(nn.Module):
    """
    End-to-end classifier using the pretrained [CLS] representation.

    Matches token-branch JEPA pretraining (encoder + cls_token), without
    perceiver pooler or token predictor.
    """

    def __init__(
        self,
        embedding: EventEmbedding,
        encoder: EHRTransformerEncoder,
        cls_token: nn.Parameter,
        head_type: HeadType = "linear",
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.cls_token = cls_token
        d_model = encoder.config.d_model
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
        cls_emb = encode_cls_from_batch(
            self.embedding,
            self.encoder,
            self.cls_token,
            codes,
            attention_mask,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask,
        )
        return self.head(cls_emb).squeeze(-1)
