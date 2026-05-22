"""Supervised binary classification: embedding → encoder → pool → head."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from evaluation.supervised_perceiver import HeadType, build_classification_head
from models.cls_encoding import encode_cls_from_batch
from models.event_embedding import EventEmbedding
from models.sequence_pooling import SequencePoolingMode, mean_pool_sequence, parse_pooling_mode
from models.transformer_encoder import EHRTransformerEncoder


class SupervisedCLSClassifier(nn.Module):
    """
    End-to-end classifier on encoder sequence representations.

    Pooling (``downstream_eval.pooling``):
      cls       — prepend pretrained [CLS] and read position 0
      mean_pool — masked mean over event tokens (no CLS)
    """

    def __init__(
        self,
        embedding: EventEmbedding,
        encoder: EHRTransformerEncoder,
        cls_token: Optional[nn.Parameter],
        head_type: HeadType = "linear",
        head_dropout: float = 0.1,
        pooling_mode: SequencePoolingMode = "cls",
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.cls_token = cls_token
        self.pooling_mode = parse_pooling_mode(pooling_mode)
        if self.pooling_mode == "cls" and cls_token is None:
            raise ValueError("cls_token is required when pooling_mode='cls'")
        d_model = encoder.config.d_model
        self.head = build_classification_head(
            d_model, d_model, head_type, head_dropout
        )

    def _encode_events(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(
            codes,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask.float() if value_mask is not None else None,
        )
        return self.encoder(x, attention_mask=attention_mask)

    def forward(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pooling_mode == "cls":
            pooled = encode_cls_from_batch(
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
        else:
            h = self._encode_events(
                codes,
                attention_mask,
                values,
                z_scores,
                delta_times,
                value_mask,
            )
            pooled = mean_pool_sequence(h, attention_mask)
        return self.head(pooled).squeeze(-1)
