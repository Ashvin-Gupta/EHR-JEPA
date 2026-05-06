"""
Predictor network and Temporal Span Prompt for EHR-JEPA.

TemporalSpanPrompt
------------------
Encodes (midpoint_hours, duration_hours) per span into d_model vectors.

Architecture:
  Linear(2 → d_model) → GELU → Linear(d_model → d_model)

Input:  FloatTensor [B, num_spans, 2]
Output: FloatTensor [B, num_spans, d_model]


Predictor
---------
A shallow 2-layer Transformer that predicts target latent representations
from time-conditioned context representations.

Forward pass (see Predictor.forward for details):
  1. Expand Z_context [B, 16, d] → [B, num_spans, 16, d]
  2. Add span_prompts (broadcast over 16 latents) + LayerNorm → Z_prompted
  3. Reshape to [B*num_spans, 16, d], run transformer
  4. Reshape back to [B, num_spans, 16, d]

Output: Z_pred [B, num_spans, 16, d_model]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig


class TemporalSpanPrompt(nn.Module):
    """
    Encodes (midpoint_hours, duration_hours) into conditioning vectors.

    Parameters
    ----------
    d_model:
        Hidden dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, span_coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        span_coords:
            FloatTensor (B, num_spans, 2) — columns are [midpoint_hours, duration_hours].

        Returns
        -------
        FloatTensor (B, num_spans, d_model).
        """
        return self.mlp(span_coords)


class Predictor(nn.Module):
    """
    Shallow Transformer predictor conditioned on temporal span prompts.

    Parameters
    ----------
    d_model:
        Hidden dimension.
    n_heads:
        Attention heads.  Default 8.
    n_layers:
        Transformer depth.  Default 2 (keep shallow to force reliance on prompt).
    dropout:
        Dropout rate.  Default 0.0.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        cfg = TransformerEncoderConfig(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=d_model * 4,
            dropout=dropout,
        )
        self.transformer = EHRTransformerEncoder(cfg)
        self.prompt_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_context: torch.Tensor,
        span_prompts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_context:
            FloatTensor (B, n_latents, d_model) — pooled context representation.
        span_prompts:
            FloatTensor (B, num_spans, d_model) — temporal prompts per span.

        Returns
        -------
        FloatTensor (B, num_spans, n_latents, d_model).
        """
        B, n_latents, d = z_context.shape
        num_spans = span_prompts.shape[1]

        # Expand context: (B, 1, n_latents, d) → (B, num_spans, n_latents, d)
        z_exp = z_context.unsqueeze(1).expand(B, num_spans, n_latents, d)

        # Prompts: (B, num_spans, d) → (B, num_spans, 1, d) — broadcast over latents
        prompts_exp = span_prompts.unsqueeze(2)

        # Add prompts and normalize
        z_prompted = self.prompt_norm(z_exp + prompts_exp)  # (B, num_spans, n_latents, d)

        # Flatten spans into batch dim for the transformer
        z_in = z_prompted.reshape(B * num_spans, n_latents, d)

        # No padding mask: all latent positions are real
        z_out = self.transformer(z_in)  # (B*num_spans, n_latents, d)

        # Reshape back
        z_pred = z_out.reshape(B, num_spans, n_latents, d)
        return z_pred
