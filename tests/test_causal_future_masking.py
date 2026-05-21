"""Tests for masking/causal_future_masking.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from masking.causal_future_masking import CausalFutureMasker


def _hours_linear(n: int) -> list[float]:
    return [float(i) for i in range(n)]


def test_no_future_in_context():
    m = CausalFutureMasker(
        num_cutpoints_S=2,
        future_max_events=8,
        future_max_hours=100.0,
        min_target_events=1,
        seed=1,
    )
    L = 32
    attn = [1] * L
    times = _hours_linear(L)
    r = m(seq_len=L, attention_mask=attn, times_hours=times)
    assert len(r.contexts) == 2 and len(r.target_spans) == 2
    for s in range(2):
        ctx, tgt = r.contexts[s], r.target_spans[s]
        if not tgt:
            continue
        t_max_ctx = max(ctx) if ctx else -1
        t_min_tgt = min(tgt)
        assert t_max_ctx < t_min_tgt
        assert set(ctx) & set(tgt) == set()


def test_target_respects_64_events():
    m = CausalFutureMasker(
        num_cutpoints_S=1,
        future_max_events=5,
        future_max_hours=1e6,
        min_target_events=1,
        seed=0,
    )
    L = 40
    attn = [1] * L
    times = _hours_linear(L)
    # Force cut early: run many seeds until we get non-empty
    for seed in range(50):
        m._rng.seed(seed)
        r = m(seq_len=L, attention_mask=attn, times_hours=times)
        if r.target_spans[0]:
            assert len(r.target_spans[0]) <= 5
            break
    else:
        raise AssertionError("expected at least one non-empty target")


def test_target_respects_12h():
    """With 1h between events, at most 3 targets fit in a 3h window (+ first event)."""
    m = CausalFutureMasker(
        num_cutpoints_S=1,
        future_max_events=500,
        future_max_hours=3.0,
        min_target_events=1,
        seed=42,
    )
    L = 80
    attn = [1] * L
    times = [float(i) for i in range(L)]
    r = m(seq_len=L, attention_mask=attn, times_hours=times)
    for tgt in r.target_spans:
        if len(tgt) < 2:
            continue
        t_cut = tgt[0] - 1
        for p in tgt:
            assert times[p] - times[t_cut] <= 3.0 + 1e-5


def test_capped_context_shorter_than_prefix():
    m = CausalFutureMasker(
        num_cutpoints_S=1,
        future_max_events=4,
        future_max_hours=2.0,
        context_chunk_mode="capped_64_or_12h",
        min_target_events=1,
        seed=0,
    )
    L = 30
    attn = [1] * L
    times = _hours_linear(L)
    r = m(seq_len=L, attention_mask=attn, times_hours=times)
    ctx, tgt = r.contexts[0], r.target_spans[0]
    if ctx and tgt:
        assert len(ctx) <= 4


def test_min_target_events_skips_pair_when_impossible():
    """If the time/event caps make it impossible to fit min_target_events, all pairs empty."""
    m = CausalFutureMasker(
        num_cutpoints_S=2,
        future_max_events=100,
        future_max_hours=0.0001,
        min_target_events=1,
        max_cutpoint_resamples=8,
        seed=0,
    )
    L = 20
    attn = [1] * L
    times = [float(i) for i in range(L)]  # 1h between events; window too short for any target
    r = m(seq_len=L, attention_mask=attn, times_hours=times)
    for tgt in r.target_spans:
        assert tgt == []


def test_padding_positions_ignored():
    L = 16
    attn = [1] * 10 + [0] * 6
    times = _hours_linear(10) + [0.0] * 6
    m = CausalFutureMasker(num_cutpoints_S=2, min_target_events=1, seed=3)
    r = m(seq_len=L, attention_mask=attn, times_hours=times)
    for s in range(2):
        for idx in r.contexts[s] + r.target_spans[s]:
            assert idx < 10


def test_non_empty_targets_meet_min_events():
    m = CausalFutureMasker(
        num_cutpoints_S=3,
        future_max_events=30,
        future_max_hours=1e6,
        min_target_events=12,
        max_cutpoint_resamples=80,
        seed=7,
    )
    L = 80
    attn = [1] * L
    times = _hours_linear(L)
    r = m(seq_len=L, attention_mask=attn, times_hours=times)
    for tgt in r.target_spans:
        if tgt:
            assert len(tgt) >= 12


if __name__ == "__main__":
    import traceback

    tests = [
        test_no_future_in_context,
        test_target_respects_64_events,
        test_target_respects_12h,
        test_capped_context_shorter_than_prefix,
        test_min_target_events_skips_pair_when_impossible,
        test_non_empty_targets_meet_min_events,
        test_padding_positions_ignored,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception:
            print(f"\n  FAILED: {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 50)
    if failed:
        sys.exit(1)
