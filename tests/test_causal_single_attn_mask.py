"""Quadrant self-attention mask for causal_single Branch B predictor."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.attention_masks import (
    build_causal_single_quadrant_mask,
    build_causal_single_quadrant_mask_batch,
    quadrant_mask_allows,
)
from tests.test_causal_single_forward import (
    MIN_TGT,
    VOCAB,
    _build_trainer,
    _causal_pre_mask,
)


def test_quadrant_mask_four_blocks():
    nc, nt = 4, 3
    m = build_causal_single_quadrant_mask(nc, nt)
    L = nc + nt

    for i in range(nc):
        for j in range(nc):
            assert quadrant_mask_allows(m, i, j), f"ctx-ctx ({i},{j})"
        for j in range(nc, L):
            assert not quadrant_mask_allows(m, i, j), f"ctx-tgt ({i},{j})"

    for i in range(nc, L):
        for j in range(nc):
            assert quadrant_mask_allows(m, i, j), f"tgt-ctx ({i},{j})"
        for j in range(nc, L):
            allowed = i == j
            assert quadrant_mask_allows(m, i, j) == allowed, f"tgt-tgt ({i},{j})"


def test_quadrant_mask_cls_attention():
    nc, nt = 4, 3
    m = build_causal_single_quadrant_mask(nc, nt, include_cls=True)
    off = 1
    ctx_end = off + nc
    tgt_end = ctx_end + nt

    assert quadrant_mask_allows(m, 0, 0)
    for j in range(off, ctx_end):
        assert quadrant_mask_allows(m, 0, j), f"CLS→ctx {j}"
    for j in range(ctx_end, tgt_end):
        assert not quadrant_mask_allows(m, 0, j), f"CLS→tgt {j}"

    for i in range(off, ctx_end):
        assert quadrant_mask_allows(m, i, 0), f"ctx→CLS {i}"

    for i in range(ctx_end, tgt_end):
        assert quadrant_mask_allows(m, i, 0), f"tgt→CLS {i}"


def test_quadrant_mask_batch_padded_row():
    m = build_causal_single_quadrant_mask_batch(
        [3, 5], [2, 1], max_len=10, include_cls=True, device="cpu"
    )
    assert m.shape == (2, 10, 10)
    assert quadrant_mask_allows(m[0], 0, 5) is False
    assert quadrant_mask_allows(m[0], 5, 5) is True
    assert quadrant_mask_allows(m[1], 6, 6) is True
    assert quadrant_mask_allows(m[1], 6, 7) is False


@pytest.mark.parametrize("seq_len", [32, 64, 128, 512, 1024])
def test_quadrant_forward_finite(seq_len: int):
    trainer = _build_trainer(causal_single_predictor_attn="quadrant")
    B = 8
    codes = torch.randint(0, VOCAB, (B, seq_len))
    attn = torch.ones(B, seq_len, dtype=torch.long)
    if seq_len >= 64:
        attn[:, -8:] = 0
    pre = _causal_pre_mask(B, seq_len, cut=20, n_target=max(MIN_TGT + 4, 24))
    l_pred, l_cov, l_total = trainer(
        codes,
        attn,
        torch.rand(B, seq_len),
        torch.randn(B, seq_len).clamp(-5, 5),
        torch.rand(B, seq_len).clamp(0, 5),
        torch.ones(B, seq_len, dtype=torch.long),
        pre_mask=pre,
    )
    assert torch.isfinite(l_pred), f"l_pred nan at L={seq_len}"
    assert torch.isfinite(l_cov), f"l_cov nan at L={seq_len}"
    assert torch.isfinite(l_total)


def test_unknown_attn_mode_raises():
    trainer = _build_trainer(causal_single_predictor_attn="invalid_mode")
    batch_size, seq_len = 2, 64
    codes = torch.randint(0, VOCAB, (batch_size, seq_len))
    attn = torch.ones(batch_size, seq_len, dtype=torch.long)
    pre = _causal_pre_mask(batch_size, seq_len, cut=15, n_target=MIN_TGT + 2)
    with pytest.raises(ValueError, match="causal_single_predictor_attn"):
        trainer(codes, attn, pre_mask=pre)


if __name__ == "__main__":
    test_quadrant_mask_four_blocks()
    test_quadrant_mask_cls_attention()
    test_quadrant_mask_batch_padded_row()
    print("ok")
