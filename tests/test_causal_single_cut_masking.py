"""Tests for masking/causal_single_cut_masking.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from masking.causal_single_cut_masking import CausalSingleCutMasker
from training.trainer import _compute_causal_single_monitoring


def _hours_linear(n: int) -> list[float]:
    return [float(i) for i in range(n)]


def test_single_span_output_shape():
    m = CausalSingleCutMasker(
        min_context_events=15,
        min_target_events=15,
        seed=1,
    )
    L = 80
    r = m(seq_len=L, attention_mask=[1] * L, times_hours=_hours_linear(L))
    assert len(r.target_spans) == 1
    assert r.span_times == []


def test_no_future_in_context():
    m = CausalSingleCutMasker(
        min_context_events=10,
        min_target_events=5,
        seed=2,
    )
    L = 60
    times = _hours_linear(L)
    r = m(seq_len=L, attention_mask=[1] * L, times_hours=times)
    tgt = r.target_spans[0]
    if not tgt:
        return
    ctx = r.context_indices
    assert max(ctx) < min(tgt)
    assert set(ctx) & set(tgt) == set()


def test_context_start_and_cut_in_bounds():
    m = CausalSingleCutMasker(
        min_context_events=10,
        min_target_events=10,
        seed=42,
    )
    L = 100
    r = m(seq_len=L, attention_mask=[1] * L, times_hours=_hours_linear(L))
    tgt = r.target_spans[0]
    if not tgt:
        return
    s = r.context_start_index
    t = r.cutpoint_index
    assert s is not None and t is not None
    assert s <= t
    assert min(r.context_indices) == s
    assert max(r.context_indices) == t
    assert min(tgt) > t
    assert max(tgt) == L - 1


def test_tokens_before_context_start_excluded():
    m = CausalSingleCutMasker(
        min_context_events=5,
        min_target_events=5,
        seed=7,
    )
    found_late_start = False
    for seed in range(50):
        m._rng.seed(seed)
        L = 60
        r = m(seq_len=L, attention_mask=[1] * L, times_hours=_hours_linear(L))
        if not r.target_spans[0]:
            continue
        s = r.context_start_index
        if s is not None and s > 0:
            found_late_start = True
            for p in r.context_indices + r.target_spans[0]:
                assert p >= s
            break
    assert found_late_start, "expected some samples with context_start > 0"


def test_target_runs_to_sequence_end():
    """Target is (t, last_real]; not capped by future_max (ignored)."""
    m = CausalSingleCutMasker(
        future_max_events=5,
        future_max_hours=0.01,
        min_context_events=5,
        min_target_events=20,
        seed=3,
    )
    L = 80
    r = m(seq_len=L, attention_mask=[1] * L, times_hours=_hours_linear(L))
    tgt = r.target_spans[0]
    if not tgt:
        return
    assert len(tgt) >= 20
    assert max(tgt) == L - 1


def test_min_context_events():
    m = CausalSingleCutMasker(
        min_context_events=20,
        min_target_events=5,
        seed=3,
    )
    L = 50
    r = m(seq_len=L, attention_mask=[1] * L, times_hours=_hours_linear(L))
    tgt = r.target_spans[0]
    if not tgt:
        return
    assert len(r.context_indices) >= 20


def test_impossible_sequence_returns_empty_target():
    m = CausalSingleCutMasker(
        min_context_events=15,
        min_target_events=15,
        seed=0,
    )
    L = 20
    r = m(seq_len=L, attention_mask=[1] * L, times_hours=_hours_linear(L))
    assert r.target_spans == []
    assert r.span_times == []


def test_target_delta_minutes_from_cut():
    m = CausalSingleCutMasker(
        min_context_events=5,
        min_target_events=5,
        seed=11,
    )
    L = 60
    times_h = [float(i) for i in range(L)]
    r = m(seq_len=L, attention_mask=[1] * L, times_hours=times_h)
    tgt = r.target_spans[0]
    assert r.target_token_delta_minutes is not None
    dm = r.target_token_delta_minutes[0]
    assert len(dm) == len(tgt)
    t_cut = r.cutpoint_index
    assert t_cut is not None
    for j, p in enumerate(tgt):
        assert dm[j] == pytest.approx((times_h[p] - times_h[t_cut]) * 60.0)


def test_cutpoint_index_matches_context_end():
    m = CausalSingleCutMasker(
        min_context_events=10,
        min_target_events=10,
        seed=99,
    )
    L = 70
    r = m(seq_len=L, attention_mask=[1] * L, times_hours=_hours_linear(L))
    if not r.target_spans[0]:
        return
    assert r.cutpoint_index is not None
    assert r.cutpoint_index == max(r.context_indices)


def test_causal_single_monitoring_metrics():
    ctx = [list(range(5, 20))]
    tgt = [[list(range(21, 35))]]
    m = _compute_causal_single_monitoring(
        ctx, tgt, cutpoints=[19], context_starts=[5]
    )
    assert m["causal_cut_position_ratio"] == pytest.approx((19 - 5) / (34 - 5))
    assert m["causal_context_token_fraction"] == pytest.approx(15 / 29)
    assert m["causal_target_token_fraction"] == pytest.approx(14 / 29)


def test_padding_positions_ignored():
    m = CausalSingleCutMasker(
        min_context_events=5,
        min_target_events=5,
        seed=7,
    )
    L = 40
    attn = [1] * 25 + [0] * 15
    times = _hours_linear(L)
    r = m(seq_len=L, attention_mask=attn, times_hours=times)
    for idx in r.context_indices + (r.target_spans[0] or []):
        assert idx < 25


if __name__ == "__main__":
    tests = [
        test_single_span_output_shape,
        test_no_future_in_context,
        test_context_start_and_cut_in_bounds,
        test_tokens_before_context_start_excluded,
        test_target_runs_to_sequence_end,
        test_min_context_events,
        test_impossible_sequence_returns_empty_target,
        test_target_delta_minutes_from_cut,
        test_cutpoint_index_matches_context_end,
        test_causal_single_monitoring_metrics,
        test_padding_positions_ignored,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
