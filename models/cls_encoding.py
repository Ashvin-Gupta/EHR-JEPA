"""Shared [CLS] prepending for JEPA encoder and downstream evaluation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.event_embedding import EventEmbedding
from models.transformer_encoder import EHRTransformerEncoder


def encode_embeddings_with_cls(
    encoder: EHRTransformerEncoder,
    cls_token: torch.Tensor,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Run the encoder on [CLS | event embeddings].

    Parameters
    ----------
    x: (B, L, d) event embeddings (no CLS yet)
    attention_mask: (B, L) — 1=real event, 0=pad

    Returns
    -------
    h: (B, L + 1, d) — position 0 is CLS
    """
    B = x.shape[0]
    cls = cls_token.view(1, 1, -1).expand(B, 1, -1)
    x_in = torch.cat([cls, x], dim=1)
    cls_mask = attention_mask.new_ones(B, 1)
    full_mask = torch.cat([cls_mask, attention_mask], dim=1)
    return encoder(x_in, attention_mask=full_mask)


def encode_cls_from_embeddings(
    encoder: EHRTransformerEncoder,
    cls_token: torch.Tensor,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """CLS position output (B, d)."""
    return encode_embeddings_with_cls(encoder, cls_token, x, attention_mask)[:, 0, :]


def encode_cls_from_batch(
    embedding: EventEmbedding,
    encoder: EHRTransformerEncoder,
    cls_token: torch.Tensor,
    codes: torch.Tensor,
    attention_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    z_scores: Optional[torch.Tensor] = None,
    delta_times: Optional[torch.Tensor] = None,
    value_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Embed a batch and return the CLS vector (B, d)."""
    x = embedding(
        codes,
        values=values,
        z_scores=z_scores,
        delta_times=delta_times,
        value_mask=value_mask.float() if value_mask is not None else None,
    )
    return encode_cls_from_embeddings(encoder, cls_token, x, attention_mask)
