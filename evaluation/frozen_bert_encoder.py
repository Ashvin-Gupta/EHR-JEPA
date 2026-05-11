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

Note: the model is frozen at construction time (requires_grad=False on all
parameters).  Gradients are never computed through this module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FrozenBERTEncoder(nn.Module):
    """
    Wraps a pretrained BERTEHRModel and returns the [CLS] embedding.

    Parameters
    ----------
    bert_model:
        A BERTEHRModel instance whose weights have been loaded from a checkpoint.
        All parameters are frozen inside this wrapper.
    """

    def __init__(self, bert_model: "BERTEHRModel") -> None:  # noqa: F821
        super().__init__()
        self.bert_model = bert_model

        for p in self.parameters():
            p.requires_grad_(False)

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
        FloatTensor (B, d_model) — the [CLS] token embedding.
        """
        # For downstream evaluation we pass the real codes (no masking).
        # mlm_labels is all -100 so the MLM loss is zero and irrelevant.
        B, L = codes.shape
        dummy_labels = codes.new_full((B, L), -100)

        _, cls_embedding = self.bert_model(
            input_codes=codes,
            attention_mask=attention_mask,
            mlm_labels=dummy_labels,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask,
        )
        return cls_embedding  # (B, d_model)
