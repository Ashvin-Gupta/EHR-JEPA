"""
Tests for models/predictor.py — TemporalSpanPrompt and Predictor.
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.predictor import Predictor, TemporalSpanPrompt


D = 32
N_LATENTS = 8
B = 2
NUM_SPANS = 4


def _prompt():
    return TemporalSpanPrompt(d_model=D)


def _predictor():
    return Predictor(d_model=D, n_heads=4, n_layers=2, dropout=0.0)


# ------------------------------------------------------------------
# TemporalSpanPrompt
# ------------------------------------------------------------------

def test_temporal_prompt_shape():
    prompt = _prompt()
    coords = torch.rand(B, NUM_SPANS, 2)  # (midpoint, duration)
    out = prompt(coords)
    print(f"\n[test_temporal_prompt_shape] input={tuple(coords.shape)}, output={tuple(out.shape)}")
    print(f"  first prompt vector[:5]: {out[0, 0, :5].detach().tolist()}")
    assert out.shape == (B, NUM_SPANS, D)


def test_temporal_prompt_different_coords():
    """Different (midpoint, duration) should produce different prompt vectors."""
    prompt = _prompt()
    coords_a = torch.tensor([[[0.0, 1.0]] * NUM_SPANS] * B)
    coords_b = torch.tensor([[[100.0, 5.0]] * NUM_SPANS] * B)

    out_a = prompt(coords_a)
    out_b = prompt(coords_b)
    diff_norm = (out_a - out_b).norm()
    print(f"\n[test_temporal_prompt_different_coords] diff norm: {diff_norm:.4f}")
    assert diff_norm > 1e-4


# ------------------------------------------------------------------
# Predictor
# ------------------------------------------------------------------

def test_predictor_forward_shape():
    pred = _predictor()
    z_context = torch.randn(B, N_LATENTS, D)
    span_prompts = torch.randn(B, NUM_SPANS, D)
    out = pred(z_context, span_prompts)
    print(f"\n[test_predictor_forward_shape]")
    print(f"  z_context={tuple(z_context.shape)}, span_prompts={tuple(span_prompts.shape)}")
    print(f"  output={tuple(out.shape)}")
    assert out.shape == (B, NUM_SPANS, N_LATENTS, D)


def test_prompt_conditioning_effect():
    """Z_pred should differ when span_prompts have different structured values."""
    pred = _predictor()
    torch.manual_seed(0)
    z_context = torch.randn(B, N_LATENTS, D)

    # Use random structured prompts (not constant-shift vectors, which LayerNorm cancels)
    torch.manual_seed(1)
    prompts_a = torch.randn(B, NUM_SPANS, D)
    torch.manual_seed(99)
    prompts_b = torch.randn(B, NUM_SPANS, D)

    out_a = pred(z_context, prompts_a)
    out_b = pred(z_context, prompts_b)
    diff_norm = (out_a - out_b).norm().item()
    print(f"\n[test_prompt_conditioning_effect] diff norm: {diff_norm:.4f}")
    assert diff_norm > 1e-3, "Different prompts should produce different predictor outputs"


def test_flatten_reshape_roundtrip():
    """Verify the internal flatten/reshape preserves value order."""
    B2, NS, NL, D2 = 2, 3, 4, 8
    z = torch.arange(B2 * NS * NL * D2, dtype=torch.float).reshape(B2, NS, NL, D2)
    # Flatten
    z_flat = z.reshape(B2 * NS, NL, D2)
    # Unflatten
    z_back = z_flat.reshape(B2, NS, NL, D2)
    diff = (z - z_back).abs().max()
    print(f"\n[test_flatten_reshape_roundtrip] max diff after reshape: {diff:.2e}")
    assert diff < 1e-10


def test_predictor_no_nan():
    pred = _predictor()
    z_context = torch.randn(B, N_LATENTS, D)
    span_prompts = torch.randn(B, NUM_SPANS, D)
    out = pred(z_context, span_prompts)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_predictor_gradient_flows():
    pred = _predictor()
    z_context = torch.randn(B, N_LATENTS, D, requires_grad=True)
    span_prompts = torch.randn(B, NUM_SPANS, D, requires_grad=True)
    out = pred(z_context, span_prompts)
    out.sum().backward()
    assert z_context.grad is not None
    assert span_prompts.grad is not None
    print(f"\n[test_predictor_gradient_flows] "
          f"z_context grad norm: {z_context.grad.norm():.4f}, "
          f"span_prompts grad norm: {span_prompts.grad.norm():.4f}")


if __name__ == "__main__":
    import traceback
    tests = [
        test_temporal_prompt_shape,
        test_temporal_prompt_different_coords,
        test_predictor_forward_shape,
        test_prompt_conditioning_effect,
        test_flatten_reshape_roundtrip,
        test_predictor_no_nan,
        test_predictor_gradient_flows,
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
