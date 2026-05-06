"""
Tests for models/event_embedding.py — all four MLP modes.

Uses synthetic [B=2, L=8] tensors with d_model=32.
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.event_embedding import EmbeddingConfig, EventEmbedding


D = 32
VOCAB = 20
B, L = 2, 8


def _make_codes():
    return torch.randint(0, VOCAB, (B, L))


def _make_embedding(use_value: bool, use_time: bool) -> EventEmbedding:
    cfg = EmbeddingConfig(
        embedding_type="learned",
        vocab_size=VOCAB,
        d_model=D,
        use_value=use_value,
        use_time=use_time,
    )
    return EventEmbedding(cfg)


# ------------------------------------------------------------------
# Code only
# ------------------------------------------------------------------

def test_code_only():
    model = _make_embedding(use_value=False, use_time=False)
    codes = _make_codes()
    out = model(codes)
    print(f"\n[test_code_only] output shape: {tuple(out.shape)}")
    print(f"  first output vector[:5]: {out[0, 0, :5].detach().tolist()}")
    assert out.shape == (B, L, D)
    assert not hasattr(model, "mlp") or model.mlp is None


def test_code_only_no_mlp():
    model = _make_embedding(use_value=False, use_time=False)
    assert model.mlp is None
    assert model.layer_norm is None


# ------------------------------------------------------------------
# Code + value  (N_extra = 3)
# ------------------------------------------------------------------

def test_code_plus_value():
    model = _make_embedding(use_value=True, use_time=False)
    codes = _make_codes()
    values = torch.rand(B, L)
    z_scores = torch.rand(B, L)
    value_mask = torch.ones(B, L)

    out = model(codes, values=values, z_scores=z_scores, value_mask=value_mask)
    print(f"\n[test_code_plus_value] output shape: {tuple(out.shape)}")
    print(f"  MLP first layer weight shape: {tuple(model.mlp[0].weight.shape)}")
    print(f"  first output vector[:5]: {out[0, 0, :5].detach().tolist()}")

    assert out.shape == (B, L, D)
    # First Linear: d_model + 3 → d_model
    assert model.mlp[0].weight.shape == (D, D + 3)


def test_code_plus_value_residual():
    """Output with MLP should differ from code-only embedding."""
    model_code = _make_embedding(use_value=False, use_time=False)
    model_val = _make_embedding(use_value=True, use_time=False)

    torch.manual_seed(0)
    codes = _make_codes()
    values = torch.rand(B, L)
    z_scores = torch.rand(B, L)
    value_mask = torch.ones(B, L)

    # Ensure they share the same embedding weights for fair comparison
    model_val.embedding.weight.data.copy_(model_code.embedding.weight.data)

    out_code = model_code(codes)
    out_val = model_val(codes, values=values, z_scores=z_scores, value_mask=value_mask)
    print(f"\n[test_code_plus_value_residual] diff norm: {(out_code - out_val).norm().item():.4f}")
    # They should differ since MLP projects value features
    assert not torch.allclose(out_code, out_val)


# ------------------------------------------------------------------
# Code + time  (N_extra = 1)
# ------------------------------------------------------------------

def test_code_plus_time():
    model = _make_embedding(use_value=False, use_time=True)
    codes = _make_codes()
    delta_times = torch.rand(B, L)

    out = model(codes, delta_times=delta_times)
    print(f"\n[test_code_plus_time] output shape: {tuple(out.shape)}")
    print(f"  MLP first layer weight shape: {tuple(model.mlp[0].weight.shape)}")

    assert out.shape == (B, L, D)
    assert model.mlp[0].weight.shape == (D, D + 1)


# ------------------------------------------------------------------
# Code + value + time  (N_extra = 4)
# ------------------------------------------------------------------

def test_code_plus_value_plus_time():
    model = _make_embedding(use_value=True, use_time=True)
    codes = _make_codes()
    values = torch.rand(B, L)
    z_scores = torch.rand(B, L)
    delta_times = torch.rand(B, L)
    value_mask = torch.ones(B, L)

    out = model(codes, values=values, z_scores=z_scores,
                delta_times=delta_times, value_mask=value_mask)
    print(f"\n[test_code_plus_value_plus_time] output shape: {tuple(out.shape)}")
    print(f"  MLP first layer weight shape: {tuple(model.mlp[0].weight.shape)}")
    print(f"  first output vector[:5]: {out[0, 0, :5].detach().tolist()}")

    assert out.shape == (B, L, D)
    assert model.mlp[0].weight.shape == (D, D + 4)


# ------------------------------------------------------------------
# Extra tensors passed to wrong mode are silently ignored
# ------------------------------------------------------------------

def test_extra_tensors_ignored():
    """Passing unused tensors should not raise an error."""
    model = _make_embedding(use_value=False, use_time=False)
    codes = _make_codes()
    values = torch.rand(B, L)
    z_scores = torch.rand(B, L)
    delta_times = torch.rand(B, L)
    value_mask = torch.ones(B, L)

    out = model(codes, values=values, z_scores=z_scores,
                delta_times=delta_times, value_mask=value_mask)
    print(f"\n[test_extra_tensors_ignored] shape: {tuple(out.shape)} — no crash")
    assert out.shape == (B, L, D)


def test_omitted_optional_tensors():
    """Passing None for optional tensors in active modes defaults to zeros."""
    model = _make_embedding(use_value=True, use_time=True)
    codes = _make_codes()
    # Pass all None — should not crash, should fall back to zero tensors
    out = model(codes)
    print(f"\n[test_omitted_optional_tensors] shape: {tuple(out.shape)} — no crash")
    assert out.shape == (B, L, D)


# ------------------------------------------------------------------
# Frozen weights (text_based)
# ------------------------------------------------------------------

def test_text_based_frozen_weights(tmp_path=None):
    import tempfile, torch, pathlib

    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp())

    vocab_size = 10
    hidden_dim = 16
    fake_emb = torch.randn(vocab_size, hidden_dim)
    emb_path = str(tmp_path / "emb.pt")
    torch.save(fake_emb, emb_path)

    cfg = EmbeddingConfig(
        embedding_type="text_based",
        vocab_size=vocab_size,
        d_model=D,
        code_embeddings_path=emb_path,
        encoder_hidden_dim=hidden_dim,
        unk_idx=0,
    )
    model = EventEmbedding(cfg)

    # Embedding weights should be frozen
    for param in model.embedding.parameters():
        assert not param.requires_grad, "Text-based embedding weights should be frozen"

    # Projection weights should be trainable
    assert model.projection is not None
    for param in model.projection.parameters():
        assert param.requires_grad

    codes = torch.randint(0, vocab_size, (B, L))
    out = model(codes)
    print(f"\n[test_text_based_frozen_weights] shape: {tuple(out.shape)}")
    assert out.shape == (B, L, D)


if __name__ == "__main__":
    import traceback
    tests = [
        test_code_only,
        test_code_only_no_mlp,
        test_code_plus_value,
        test_code_plus_value_residual,
        test_code_plus_time,
        test_code_plus_value_plus_time,
        test_extra_tensors_ignored,
        test_omitted_optional_tensors,
        test_text_based_frozen_weights,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*50)
    if failed:
        sys.exit(1)
