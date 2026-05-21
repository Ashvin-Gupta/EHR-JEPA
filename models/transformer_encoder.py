"""
Transformer Encoder with Rotary Position Embeddings (RoPE).

Architecture:
  EHRTransformerEncoder
    - n_layers TransformerEncoderLayers (nn.TransformerEncoderLayer)
    - RoPE applied inside each self-attention block via a hook

RoPE position IDs:
  - Default: arange(L) — standard sequential positions.
  - Custom: any LongTensor [B, L] — allows non-contiguous context windows
    to preserve their original temporal positions (extracted_with_positions mode).

Usage
-----
  enc = EHRTransformerEncoder(TransformerEncoderConfig())
  out = enc(x, attention_mask, position_ids=None)    # shape (B, L, d_model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerEncoderConfig:
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Computes sin/cos buffers up to max_seq_len; at forward time the buffers
    are sliced to the actual sequence length or indexed by position_ids.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(
        self, position_ids: torch.Tensor
    ):
        """
        position_ids: LongTensor (B, L) or (L,)
        Returns cos, sin each of shape (B, L, dim) or (1, L, dim).
        """
        # position_ids: (B, L)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)  # (1, L)
        # (B, L) × (dim/2,) → (B, L, dim/2)
        freqs = torch.einsum(
            "bl,d->bld",
            position_ids.float(),
            self.inv_freq,
        )
        emb = torch.cat([freqs, freqs], dim=-1)  # (B, L, dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        q, k: (B, n_heads, L, head_dim)
        position_ids: (B, L)

        Returns rotated q, k of the same shape.
        """
        cos, sin = self._get_cos_sin(position_ids)  # (B, L, dim)
        # Slice to head_dim
        cos = cos[..., : q.shape[-1]]               # (B, L, head_dim)
        sin = sin[..., : q.shape[-1]]

        # Broadcast over heads: (B, 1, L, head_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Single RoPE-aware self-attention layer
# ---------------------------------------------------------------------------

class RoPEMultiheadAttention(nn.Module):
    """Multihead attention that applies RoPE to Q and K before the dot-product."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, d_model)
        key_padding_mask: (B, L) — True on positions to IGNORE (padding convention)
        attn_mask: (L, L) or (B*n_heads, L, L) additive mask
        position_ids: (B, L) or None

        Returns (B, L, d_model).
        """
        B, L, _ = x.shape

        Q = self.q_proj(x)  # (B, L, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (B, n_heads, L, head_dim)
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        Q, K = self.rope.apply_rotary(Q, K, position_ids)

        # Merge key_padding_mask into a float additive attn_mask for compatibility
        # with older PyTorch versions that don't accept key_padding_mask directly.
        merged_mask: Optional[torch.Tensor] = attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, L) True=ignore → additive (B, 1, 1, L) float
            pad_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=Q.dtype)
            pad_mask = pad_mask.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            merged_mask = pad_mask if attn_mask is None else attn_mask + pad_mask

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=merged_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, n_heads, L, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer layer using RoPE attention
# ---------------------------------------------------------------------------

class RoPETransformerLayer(nn.Module):
    """Pre-norm Transformer encoder layer with RoPE self-attention."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask,
                           attn_mask=attn_mask, position_ids=position_ids)
        x = residual + self.drop(x)

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class EHRTransformerEncoder(nn.Module):
    """
    Stacked RoPE Transformer encoder.

    Parameters
    ----------
    config:
        TransformerEncoderConfig.

    Forward
    -------
    x:              FloatTensor (B, L, d_model)   — embeddings from EventEmbedding
    attention_mask: BoolTensor  (B, L)            — 1 = real token, 0 = pad
    position_ids:   LongTensor  (B, L) | None     — if None defaults to arange(L)

    Returns FloatTensor (B, L, d_model).
    """

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            RoPETransformerLayer(
                config.d_model, config.n_heads, config.ffn_dim, config.dropout
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        attention_mask: (B, L) with 1=real, 0=pad.
        attn_bias: optional additive mask, 0 = attend, -inf = block.
            Shapes (L, L), (B, L, L), or (B, 1, L, L).  Combined with padding.
        Internally converted to key_padding_mask (True = ignore).
        """
        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0   # (B, L), True on pad

        attn_mask = attn_bias
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)

        for layer in self.layers:
            x = layer(
                x,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                position_ids=position_ids,
            )

        return self.final_norm(x)
