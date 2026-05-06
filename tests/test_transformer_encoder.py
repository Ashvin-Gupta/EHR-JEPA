"""
Tests for models/transformer_encoder.py.

All tests use small dims (d_model=32, n_heads=4, n_layers=2) for speed.
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig


def _small_cfg(n_layers: int = 2) -> TransformerEncoderConfig:
    return TransformerEncoderConfig(
        n_layers=n_layers, d_model=32, n_heads=4, ffn_dim=64, dropout=0.0
    )


B, L, D = 2, 16, 32


def _encoder(n_layers: int = 2) -> EHRTransformerEncoder:
    return EHRTransformerEncoder(_small_cfg(n_layers)).eval()


def _rand_input():
    return torch.randn(B, L, D)


def test_forward_shape():
    enc = _encoder()
    x = _rand_input()
    out = enc(x)
    print(f"\n[test_forward_shape] input={tuple(x.shape)}, output={tuple(out.shape)}")
    assert out.shape == (B, L, D)


def test_padding_mask():
    enc = _encoder()
    x = _rand_input()

    # No mask
    out_full = enc(x)

    # Mask second half of sequence (positions L//2 ..  L-1 are pad)
    attn_mask = torch.ones(B, L, dtype=torch.long)
    attn_mask[:, L // 2 :] = 0
    out_masked = enc(x, attention_mask=attn_mask)

    # Outputs on real positions should differ because pad positions contribute
    # differently (or not at all) to attention
    print(f"\n[test_padding_mask]")
    print(f"  norm of output (full):    {out_full.norm(dim=-1).mean():.4f}")
    print(f"  norm of output (masked):  {out_masked.norm(dim=-1).mean():.4f}")
    print(f"  norm of difference:       {(out_full - out_masked).norm():.4f}")
    # The outputs should differ when padding mask changes attention
    assert not torch.allclose(out_full, out_masked, atol=1e-4), (
        "Masked and unmasked outputs should differ"
    )
    assert out_masked.shape == (B, L, D)


def test_rope_default_positions():
    """Without position_ids, output is deterministic (same call twice)."""
    enc = _encoder()
    x = _rand_input()
    out1 = enc(x)
    out2 = enc(x)
    print(f"\n[test_rope_default_positions] diff between two identical calls: "
          f"{(out1 - out2).abs().max().item()}")
    assert torch.allclose(out1, out2, atol=1e-6)


def test_rope_custom_positions():
    """Non-contiguous position_ids produce different output from sequential."""
    enc = _encoder()
    x = _rand_input()

    sequential_ids = torch.arange(L).unsqueeze(0).expand(B, L)
    # Skip some positions: 0, 1, 5, 10, 20, 30, ...
    custom_ids = torch.zeros(B, L, dtype=torch.long)
    pos = 0
    step = 3
    for i in range(L):
        custom_ids[:, i] = pos
        pos += step

    out_seq = enc(x, position_ids=sequential_ids)
    out_custom = enc(x, position_ids=custom_ids)

    print(f"\n[test_rope_custom_positions] sequential positions: {sequential_ids[0].tolist()}")
    print(f"  custom positions: {custom_ids[0].tolist()}")
    print(f"  diff norm: {(out_seq - out_custom).norm().item():.4f}")
    print(f"  first output vector (seq)[:5]: {out_seq[0, 0, :5].detach().tolist()}")
    print(f"  first output vector (custom)[:5]: {out_custom[0, 0, :5].detach().tolist()}")

    assert out_seq.shape == (B, L, D)
    assert out_custom.shape == (B, L, D)
    assert not torch.allclose(out_seq, out_custom, atol=1e-3), (
        "Custom positions should produce different output"
    )


def test_full_sequence_vs_extracted_context():
    """
    Demonstrate the two context encoding strategies:
      - full_sequence: pass all positions, use context-only attention mask
      - extracted_with_positions: extract context events, pass original pos IDs

    This test shows they produce different outputs (expected).
    """
    enc = _encoder()
    torch.manual_seed(7)
    x = _rand_input()

    # Define context positions (non-contiguous: skip positions 4-7 and 12-13)
    target_pos = list(range(4, 8)) + list(range(12, 14))
    context_pos = [p for p in range(L) if p not in target_pos]

    # --- Full sequence encoding ---
    attn_mask_full = torch.zeros(B, L, dtype=torch.long)
    for p in context_pos:
        attn_mask_full[:, p] = 1
    out_full = enc(x, attention_mask=attn_mask_full)

    # --- Extracted with original positions ---
    n_ctx = len(context_pos)
    x_ctx = x[:, context_pos, :]   # (B, n_ctx, D)
    pos_ids = torch.tensor(context_pos, dtype=torch.long).unsqueeze(0).expand(B, n_ctx)
    out_extracted = enc(x_ctx, position_ids=pos_ids)

    print(f"\n[test_full_sequence_vs_extracted_context]")
    print(f"  context positions: {context_pos}")
    print(f"  full_seq output norm (context positions, row 0): "
          f"{out_full[0, context_pos, :].norm(dim=-1).mean():.4f}")
    print(f"  extracted output norm (row 0): "
          f"{out_extracted[0, :, :].norm(dim=-1).mean():.4f}")
    print(f"  first context token (full_seq)[:5]: "
          f"{out_full[0, context_pos[0], :5].detach().tolist()}")
    print(f"  first context token (extracted)[:5]: "
          f"{out_extracted[0, 0, :5].detach().tolist()}")

    # Shape checks
    assert out_full.shape == (B, L, D)
    assert out_extracted.shape == (B, n_ctx, D)


def test_no_nan_in_output():
    enc = _encoder()
    x = _rand_input()
    out = enc(x)
    assert not torch.isnan(out).any(), "NaN detected in encoder output"
    assert not torch.isinf(out).any(), "Inf detected in encoder output"


if __name__ == "__main__":
    import traceback
    tests = [
        test_forward_shape,
        test_padding_mask,
        test_rope_default_positions,
        test_rope_custom_positions,
        test_full_sequence_vs_extracted_context,
        test_no_nan_in_output,
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
