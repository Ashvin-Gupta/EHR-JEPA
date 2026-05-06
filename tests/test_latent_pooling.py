"""
Tests for models/latent_pooling.py.
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.latent_pooling import LatentCrossAttentionPool


def _pool(d_model: int = 64, n_latents: int = 16, n_heads: int = 4):
    return LatentCrossAttentionPool(d_model=d_model, n_latents=n_latents, n_heads=n_heads)


D, N_LAT = 64, 16
B, L = 2, 32


def test_output_shape():
    pool = _pool()
    enc_out = torch.randn(B, L, D)
    out = pool(enc_out)
    print(f"\n[test_output_shape] input={tuple(enc_out.shape)}, output={tuple(out.shape)}")
    print(f"  output norm: {out.norm(dim=-1).mean():.4f}")
    assert out.shape == (B, N_LAT, D)


def test_key_padding_mask():
    pool = _pool()
    enc_out = torch.randn(B, L, D)

    # No mask
    out_full = pool(enc_out)

    # Pad second half
    kpm = torch.ones(B, L, dtype=torch.long)
    kpm[:, L // 2 :] = 0
    out_masked = pool(enc_out, key_padding_mask=kpm)

    print(f"\n[test_key_padding_mask]")
    print(f"  output shape (masked): {tuple(out_masked.shape)}")
    print(f"  norm (no mask):   {out_full.norm(dim=-1).mean():.4f}")
    print(f"  norm (with mask): {out_masked.norm(dim=-1).mean():.4f}")

    assert out_masked.shape == (B, N_LAT, D)
    # With different masks the outputs should differ
    assert not torch.allclose(out_full, out_masked, atol=1e-4)


def test_bool_key_padding_mask():
    """Bool mask (True = ignore) should also work."""
    pool = _pool()
    enc_out = torch.randn(B, L, D)
    kpm = torch.zeros(B, L, dtype=torch.bool)
    kpm[:, L // 2 :] = True   # True = pad/ignore
    out = pool(enc_out, key_padding_mask=kpm)
    assert out.shape == (B, N_LAT, D)


def test_latents_are_learned():
    pool = _pool()
    assert hasattr(pool, "latent_tokens")
    assert isinstance(pool.latent_tokens, torch.nn.Parameter)
    assert pool.latent_tokens.requires_grad, "Latent tokens must be learnable"
    print(f"\n[test_latents_are_learned] latent_tokens shape: "
          f"{tuple(pool.latent_tokens.shape)}, requires_grad=True")


def test_batched_target_spans():
    """Test with batched target span shape [B*num_spans, max_span_len, d]."""
    pool = _pool()
    num_spans = 4
    max_span_len = 20

    enc_out = torch.randn(B * num_spans, max_span_len, D)
    out = pool(enc_out)

    print(f"\n[test_batched_target_spans] "
          f"input={tuple(enc_out.shape)}, output={tuple(out.shape)}")
    assert out.shape == (B * num_spans, N_LAT, D)


def test_output_no_nan():
    pool = _pool()
    enc_out = torch.randn(B, L, D)
    out = pool(enc_out)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_gradient_through_latents():
    pool = _pool()
    enc_out = torch.randn(B, L, D)
    out = pool(enc_out)
    loss = out.sum()
    loss.backward()
    assert pool.latent_tokens.grad is not None
    print(f"\n[test_gradient_through_latents] "
          f"latent grad norm: {pool.latent_tokens.grad.norm():.4f}")


if __name__ == "__main__":
    import traceback
    tests = [
        test_output_shape,
        test_key_padding_mask,
        test_bool_key_padding_mask,
        test_latents_are_learned,
        test_batched_target_spans,
        test_output_no_nan,
        test_gradient_through_latents,
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
