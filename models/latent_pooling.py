"""
Latent Cross-Attention Pooling (Perceiver-style).

Converts a variable-length sequence of encoder outputs into a fixed-size
representation by cross-attending from N_latent learnable query tokens.

  queries = latent_tokens  (learned, shape [n_latents, d_model])
  keys/values = encoder_out

This module is used for both:
  - Context sequence:   input [B, N_context, d_model]  → output [B, n_latents, d_model]
  - Target spans (batched): input [B*num_spans, max_span_len, d_model]
                             → output [B*num_spans, n_latents, d_model]

The batch dimension is handled uniformly; the caller is responsible for
reshaping before and after.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LatentCrossAttentionPool(nn.Module):
    """
    Perceiver-style cross-attention pooling.

    Parameters
    ----------
    d_model:
        Hidden dimension (must match encoder output).
    n_latents:
        Number of learnable latent query tokens.  Default 16.
    n_heads:
        Number of attention heads.  Default 8.
    dropout:
        Attention dropout.  Default 0.0.
    """

    def __init__(
        self,
        d_model: int,
        n_latents: int = 16,
        n_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents

        # Learnable latent tokens — shared across the batch
        self.latent_tokens = nn.Parameter(torch.randn(n_latents, d_model))

        # Cross-attention: latents attend over encoder outputs
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        encoder_out: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        encoder_out:
            FloatTensor (B, L, d_model) — encoder output sequence.
        key_padding_mask:
            BoolTensor (B, L) — True on positions to ignore (padding).
            Can also be LongTensor (B, L) with 0=pad, 1=real; will be
            converted internally.

        Returns
        -------
        FloatTensor (B, n_latents, d_model).
        """
        B = encoder_out.shape[0]

        # Expand latents to batch dimension: (B, n_latents, d_model)
        latents = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)

        # Convert key_padding_mask from 1=real/0=pad → True=ignore convention
        kpm: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            kpm_t = key_padding_mask
            if kpm_t.dtype != torch.bool:
                kpm_t = kpm_t == 0   # 0=pad → True=ignore
            kpm = kpm_t

        out, _ = self.cross_attn(
            query=latents,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=kpm,
        )  # (B, n_latents, d_model)

        return self.norm(out)
