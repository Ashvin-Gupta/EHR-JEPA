"""Tests for FrozenAREncoder interface."""

from __future__ import annotations

import torch

from evaluation.frozen_ar_encoder import FrozenAREncoder
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.ar_trainer import AREHRModel


def test_frozen_ar_encoder_output_shape():
    d = 16
    vocab_size = 24
    emb = EventEmbedding(
        EmbeddingConfig(
            embedding_type="learned",
            vocab_size=vocab_size,
            d_model=d,
            unk_idx=vocab_size - 1,
            use_value=False,
            use_time=False,
        )
    )
    enc = EHRTransformerEncoder(
        TransformerEncoderConfig(n_layers=1, d_model=d, n_heads=2, ffn_dim=32, dropout=0.0)
    )
    ar = AREHRModel(emb, enc, vocab_size=vocab_size)
    encoder = FrozenAREncoder(ar, pooling_mode="cls")

    B, L = 3, 7
    codes = torch.randint(1, vocab_size - 2, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    out = encoder(codes, attn)
    assert out.shape == (B, d)
    assert encoder.output_dim == d
