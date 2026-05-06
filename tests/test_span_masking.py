"""
Tests for masking/span_masking.py.
"""

import math
import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from masking.span_masking import SpanMasker, SpanMaskResult


def _masker(**kwargs) -> SpanMasker:
    defaults = dict(mask_ratio=0.30, default_num_spans=4, min_span_length=15, seed=42)
    defaults.update(kwargs)
    return SpanMasker(**defaults)


def test_output_structure():
    masker = _masker()
    result = masker(seq_len=100)
    print(f"\n[test_output_structure]")
    print(f"  type: {type(result).__name__}")
    print(f"  context_indices count: {len(result.context_indices)}")
    print(f"  target_spans count: {len(result.target_spans)}")
    print(f"  span_times: {result.span_times}")
    assert isinstance(result, SpanMaskResult)
    assert isinstance(result.context_indices, list)
    assert isinstance(result.target_spans, list)
    assert isinstance(result.span_times, list)


def test_no_overlap():
    masker = _masker()
    result = masker(seq_len=200)
    all_positions = []
    for span in result.target_spans:
        all_positions.extend(span)

    print(f"\n[test_no_overlap] {len(result.target_spans)} spans, "
          f"{len(all_positions)} masked positions")
    assert len(all_positions) == len(set(all_positions)), "Overlapping spans detected!"


def test_mask_ratio():
    """Total masked tokens should be approximately mask_ratio * N."""
    masker = _masker()
    N = 200
    result = masker(seq_len=N)
    B = int(N * 0.30)
    total_masked = sum(len(s) for s in result.target_spans)
    print(f"\n[test_mask_ratio] N={N}, expected~{B}, got={total_masked}")
    # Allow ±1 per span for floor rounding
    assert abs(total_masked - B) <= len(result.target_spans), (
        f"Masked {total_masked}, expected ~{B}"
    )


def test_context_plus_target_equals_N():
    masker = _masker()
    N = 100
    result = masker(seq_len=N)
    masked = {p for span in result.target_spans for p in span}
    assert set(result.context_indices) | masked == set(range(N))
    assert set(result.context_indices) & masked == set()
    print(f"\n[test_context_plus_target_equals_N] context={len(result.context_indices)}, "
          f"masked={len(masked)}, total={len(result.context_indices) + len(masked)}")


def test_dynamic_num_spans_short():
    """Short sequence: B < default_num_spans * min_span_length → fewer spans."""
    # N=30, B=9, default_num_spans=4, min_span_length=15 → num_spans=1
    masker = _masker(mask_ratio=0.30, default_num_spans=4, min_span_length=5, seed=1)
    # Use min_span_length=5 so B=9 >= 1*5 but < 4*5=20 → num_spans=floor(9/5)=1
    result = masker(seq_len=30)
    print(f"\n[test_dynamic_num_spans_short] N=30, B=9, min_span_length=5")
    print(f"  num_spans selected: {len(result.target_spans)}")
    print(f"  span lengths: {[len(s) for s in result.target_spans]}")
    assert len(result.target_spans) <= 2  # should not be 4


def test_dynamic_num_spans_normal():
    """Long sequence: B >= default_num_spans * min_span_length → 4 spans."""
    # N=200, B=60, default_num_spans=4, min_span_length=15 → 4 spans
    masker = _masker(mask_ratio=0.30, default_num_spans=4, min_span_length=15, seed=2)
    result = masker(seq_len=200)
    print(f"\n[test_dynamic_num_spans_normal] N=200, B=60, min_span_length=15")
    print(f"  num_spans selected: {len(result.target_spans)}")
    print(f"  span lengths: {[len(s) for s in result.target_spans]}")
    assert len(result.target_spans) == 4


def test_padding_excluded():
    """Spans should only cover real (non-padding) positions."""
    N = 100
    pad_start = 80
    attention_mask = torch.ones(N, dtype=torch.long)
    attention_mask[pad_start:] = 0

    masker = _masker(seed=10)
    result = masker(seq_len=N, attention_mask=attention_mask)
    all_masked = [p for span in result.target_spans for p in span]

    print(f"\n[test_padding_excluded] padding starts at {pad_start}")
    print(f"  masked positions: {sorted(all_masked)[:10]}...")
    for p in all_masked:
        assert p < pad_start, f"Position {p} is in padding region"


def test_span_times():
    """span_times should reflect correct midpoint and duration from a times array."""
    N = 100
    # Linear timestamps: 0, 1, 2, ... hours
    times = [float(i) for i in range(N)]
    masker = _masker(seed=5)
    result = masker(seq_len=N, times=times)

    print(f"\n[test_span_times]")
    for i, (span, st) in enumerate(zip(result.target_spans, result.span_times)):
        expected_mid = (times[span[0]] + times[span[-1]]) / 2.0
        expected_dur = times[span[-1]] - times[span[0]]
        print(f"  span {i}: positions {span[:3]}..., times midpoint={st[0]:.2f} "
              f"(expected={expected_mid:.2f}), duration={st[1]:.2f}")
        assert st[0] == pytest.approx(expected_mid, abs=1e-6)
        assert st[1] == pytest.approx(expected_dur, abs=1e-6)


def test_t_span_floor():
    """Span lengths should be floor(B/num_spans); last span gets remainder."""
    # N=100, B=30, num_spans=4: T_span=7, remainder=2 → lengths [7,7,7,9]
    masker = _masker(mask_ratio=0.30, default_num_spans=4, min_span_length=5, seed=3)
    N = 100
    result = masker(seq_len=N)
    B = int(N * 0.30)
    num_spans = len(result.target_spans)
    if num_spans == 0:
        pytest.skip("No spans produced")

    T_base = B // num_spans
    remainder = B - T_base * num_spans
    expected_lengths = [T_base] * (num_spans - 1) + [T_base + remainder]
    actual_lengths = sorted([len(s) for s in result.target_spans])
    expected_sorted = sorted(expected_lengths)

    print(f"\n[test_t_span_floor] B={B}, num_spans={num_spans}")
    print(f"  T_span_base={T_base}, remainder={remainder}")
    print(f"  expected lengths (sorted): {expected_sorted}")
    print(f"  actual lengths  (sorted): {actual_lengths}")
    assert actual_lengths == expected_sorted


def test_empty_sequence():
    """When B=0 (seq too short for any masks at given ratio), return empty results."""
    # seq_len=2, mask_ratio=0.30 → B = int(2*0.30) = 0
    masker = _masker(mask_ratio=0.30, min_span_length=15)
    result = masker(seq_len=2)
    print(f"\n[test_empty_sequence] seq_len=2, context={len(result.context_indices)}, "
          f"spans={len(result.target_spans)}  (expected 0 spans since B=0)")
    assert len(result.target_spans) == 0


if __name__ == "__main__":
    import traceback
    tests = [
        test_output_structure,
        test_no_overlap,
        test_mask_ratio,
        test_context_plus_target_equals_N,
        test_dynamic_num_spans_short,
        test_dynamic_num_spans_normal,
        test_padding_excluded,
        test_span_times,
        test_t_span_floor,
        test_empty_sequence,
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
