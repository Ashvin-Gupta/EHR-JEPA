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
    loss, cls_emb = model(codes, attn)
    assert torch.isfinite(loss).all()
    assert cls_emb.shape == (B, 16)


def test_ar_forward_gradient_flow():
    model = _tiny_ar()
    codes = torch.randint(1, 18, (2, 6))
    attn = torch.ones(2, 6, dtype=torch.long)
    loss, _ = model(codes, attn)
    loss.backward()
    assert model.cls_token.grad is not None
    assert model.eos_token.grad is not None
    assert model.lm_head[0].weight.grad is not None


def test_ar_packed_two_segments():
    model = _tiny_ar()
    codes = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    seg_starts = torch.tensor([[0, 3]])
    seg_lengths = torch.tensor([[3, 3]])
    loss, cls_emb = model(codes, attn, segment_starts=seg_starts, segment_lengths=seg_lengths)
    assert torch.isfinite(loss)
    assert cls_emb.shape == (1, 16)


def test_ar_causal_mask_blocks_cross_segment():
    model = _tiny_ar()
    boundaries = [0, 4]
    mask = model._causal_segment_mask(7, boundaries, torch.device("cpu"))
    # Causal: position 0 cannot attend to future position 3
    assert mask[0, 3].item() == float("-inf")
    # Same segment: position 3 can attend to position 0 (past)
    assert mask[3, 0].item() == 0.0
    # Cross-segment: position 4 (start of seg 2) cannot attend to position 3 (seg 1)
    assert mask[4, 3].item() == float("-inf")
    assert mask[4, 4].item() == 0.0


def test_ar_encode_cls_matches_forward_cls():
    model = _tiny_ar()
    codes = torch.randint(1, 18, (2, 5))
    attn = torch.ones(2, 5, dtype=torch.long)
    loss, cls_fwd = model(codes, attn)
    cls_enc = model.encode_cls_embedding(codes, attn)
    assert torch.allclose(cls_fwd, cls_enc, atol=1e-5)


if __name__ == "__main__":
    test_ar_forward_finite_loss()
    test_ar_forward_gradient_flow()
    test_ar_packed_two_segments()
    test_ar_causal_mask_blocks_cross_segment()
    test_ar_encode_cls_matches_forward_cls()
    print("test_ar_forward: all passed")
