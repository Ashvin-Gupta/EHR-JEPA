"""
FrozenAREncoder — frozen AR backbone feature extractor for linear probing.

Returns the last-event hidden state (full causal context). Config ``pooling: cls``
is mapped to last-token pooling for causal models; ``mean_pool`` averages event
positions (excluding the leading CLS input).

Interface matches FrozenBERTEncoder / FrozenEHREncoder:
    (codes, attention_mask, ...) → (B, d_model)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.sequence_pooling import SequencePoolingMode, parse_pooling_mode


class FrozenAREncoder(nn.Module):
    def __init__(
        self,
        ar_model: "AREHRModel",  # noqa: F821
        pooling_mode: SequencePoolingMode = "cls",
    ) -> None:
        super().__init__()
        self.ar_model = ar_model
        self.pooling_mode = parse_pooling_mode(pooling_mode)
        self.output_dim: int = ar_model.output_dim

    @torch.no_grad()
    def forward(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.ar_model.encode_pooled_embedding(
            codes,
            attention_mask,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask,
            pooling_mode=self.pooling_mode,
        )
