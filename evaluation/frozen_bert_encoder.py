"""
FrozenBERTEncoder — drop-in replacement for FrozenEHREncoder for the BERT
baseline.

Returns the [CLS] token embedding (shape [B, d_model]) from the frozen
BERTEHRModel.  This is used as the feature vector for the downstream linear
probe.

The interface intentionally matches FrozenEHREncoder so the same
train_linear_probe helper and LinearProbe class work unchanged:

    encoder = FrozenBERTEncoder(bert_model)
    probe   = LinearProbe(encoder.output_dim, dropout=0.1)
    train_linear_probe(encoder, probe, train_loader, val_loader, ...)

The linear probe is trained on top of the CLS representation, which is the
standard BERT downstream evaluation protocol.

Note: forward runs under ``torch.no_grad()`` so no gradients are recorded
through the encoder.  We deliberately do **not** set ``requires_grad=False``
on the wrapped ``BERTEHRModel`` because the same module instance may still
be used for MLM pretraining after the probe; mutating ``requires_grad`` would
break the next training step.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.sequence_pooling import SequencePoolingMode, parse_pooling_mode


class FrozenBERTEncoder(nn.Module):
    """
    Wraps a pretrained BERTEHRModel and returns the [CLS] embedding.

    Parameters
    ----------
    bert_model:
        A BERTEHRModel instance whose weights have been loaded from a checkpoint.
        Forward uses ``torch.no_grad()`` only — the wrapped weights are not
        modified and ``requires_grad`` is left unchanged (see module docstring).
    """

    def __init__(
        self,
        bert_model: "BERTEHRModel",  # noqa: F821
        pooling_mode: SequencePoolingMode = "cls",
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.pooling_mode = parse_pooling_mode(pooling_mode)

        self.output_dim: int = bert_model.output_dim

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
        """
        Parameters
        ----------
        codes:          LongTensor  (B, L) — original (unmasked) codes
        attention_mask: LongTensor  (B, L) — 1=real, 0=pad
        values / z_scores / delta_times / value_mask: optional (B, L) tensors

        Returns
        -------
        FloatTensor (B, d_model) — CLS or mean-pooled event embedding.
        """
        return self.bert_model.encode_pooled_embedding(
            codes,
            attention_mask,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask,
            pooling_mode=self.pooling_mode,
        )
