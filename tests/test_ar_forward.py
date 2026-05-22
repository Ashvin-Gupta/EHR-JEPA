"""Tests for AREHRModel forward pass, causal mask, and gradients."""

from __future__ import annotations

import torch

from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.ar_trainer import AREHRModel


def _tiny_ar(vocab_size: int = 20, d: int = 16) -> AREHRModel:
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
    return AREHRModel(emb, enc, vocab_size=vocab_size)


def test_ar_forward_finite_loss():
    model = _tiny_ar()
    B, L = 2, 8
    codes = torch.randint(1, 18, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    attn[1, 5:] = 0
    loss, summary = model(codes, attn)
    assert torch.isfinite(loss).all()
    assert summary.shape == (B, 16)


def test_ar_forward_gradient_flow():
    model = _tiny_ar()
    codes = torch.randint(1, 18, (2, 6))
    attn = torch.ones(2, 6, dtype=torch.long)
    loss, _ = model(codes, attn)
    loss.backward()
    assert model.cls_token.grad is not None
    assert model.lm_head[0].weight.grad is not None


def test_ar_shifted_labels_no_cls_ignore():
    """CLS at input pos 0 must predict event_0, not be ignored with identity leak."""
    model = _tiny_ar()
    codes = torch.tensor([[10, 11, 12, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 0, 0]])
    _x, _mask, _pos, labels, last_idx = model._build_sequence_tensors(
        codes, attn, None, None, None, None, None, None
    )
    assert labels[0, 0].item() == 10
    assert labels[0, 1].item() == 11
    assert labels[0, 2].item() == 12
    assert labels[0, 3].item() == model.eos_token_idx
    assert _x.shape[1] == 4
    assert last_idx[0].item() == 3


def test_ar_packed_two_segments():
    model = _tiny_ar()
    codes = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    seg_starts = torch.tensor([[0, 3]])
    seg_lengths = torch.tensor([[3, 3]])
    loss, summary = model(codes, attn, segment_starts=seg_starts, segment_lengths=seg_lengths)
    assert torch.isfinite(loss)
    assert summary.shape == (1, 16)


def test_ar_causal_mask_blocks_cross_segment():
    model = _tiny_ar()
    # Two segments of 3 events → 4 tokens each (CLS + 3 events)
    boundaries = [0, 4]
    mask = model._causal_segment_mask(8, boundaries, torch.device("cpu"))
    assert mask[0, 3].item() == float("-inf")
    assert mask[3, 0].item() == 0.0
    assert mask[4, 3].item() == float("-inf")
    assert mask[7, 4].item() == 0.0


def test_ar_last_token_pooling_matches_forward():
    model = _tiny_ar()
    codes = torch.randint(1, 18, (2, 5))
    attn = torch.ones(2, 5, dtype=torch.long)
    loss, summary_fwd = model(codes, attn)
    summary_enc = model.encode_last_token_embedding(codes, attn)
    assert torch.allclose(summary_fwd, summary_enc, atol=1e-5)


def test_ar_cls_pooling_uses_last_token():
    model = _tiny_ar()
    codes = torch.randint(1, 18, (2, 5))
    attn = torch.ones(2, 5, dtype=torch.long)
    last = model.encode_last_token_embedding(codes, attn)
    cls_mode = model.encode_pooled_embedding(codes, attn, pooling_mode="cls")
    assert torch.allclose(last, cls_mode, atol=1e-5)


if __name__ == "__main__":
    test_ar_forward_finite_loss()
    test_ar_forward_gradient_flow()
    test_ar_shifted_labels_no_cls_ignore()
    test_ar_packed_two_segments()
    test_ar_causal_mask_blocks_cross_segment()
    test_ar_last_token_pooling_matches_forward()
    test_ar_cls_pooling_uses_last_token()
    print("test_ar_forward: all passed")
