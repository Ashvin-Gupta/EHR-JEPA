"""
Tests for data/normalizer.py.

Uses synthetic in-memory data — no parquet files required.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.normalizer import ValueNormalizer


def _make_records(code: str, values: list) -> list:
    return [(code, v) for v in values]


def _fit_known() -> ValueNormalizer:
    """Fit on a controlled dataset for deterministic assertions."""
    rng = np.random.default_rng(42)
    # 100 values from Normal(5, 2) for code "A"
    vals_a = rng.normal(5, 2, 100).tolist()
    # 50 values from Uniform(0, 10) for code "B"
    vals_b = rng.uniform(0, 10, 50).tolist()
    records = _make_records("A", vals_a) + _make_records("B", vals_b)
    norm = ValueNormalizer()
    norm.fit_from_records(records)
    return norm


def test_fit_stores_stats():
    norm = _fit_known()
    assert len(norm) == 2, f"Expected 2 codes, got {len(norm)}"
    stats = norm._stats
    for code in ("A", "B"):
        assert code in stats, f"Code '{code}' missing from stats"
        for key in ("p05", "p95", "mean", "std"):
            assert key in stats[code], f"Key '{key}' missing for code '{code}'"

    print("\n[test_fit_stores_stats] stats:")
    for code, s in stats.items():
        print(f"  {code}: p05={s['p05']:.3f}, p95={s['p95']:.3f}, "
              f"mean={s['mean']:.3f}, std={s['std']:.3f}")


def test_winsorize():
    norm = ValueNormalizer()
    # Extreme outliers: 50 values in [0, 10] plus two extreme outliers
    vals = list(range(10)) * 5 + [1000, -1000]
    norm.fit_from_records(_make_records("C", vals))
    p05 = norm._stats["C"]["p05"]
    p95 = norm._stats["C"]["p95"]

    v_high, _ = norm.transform("C", 1000.0)
    v_low, _ = norm.transform("C", -1000.0)

    print(f"\n[test_winsorize] p05={p05:.3f}, p95={p95:.3f}")
    print(f"  1000 → winsorized={v_high:.3f}  (should == p95={p95:.3f})")
    print(f"  -1000 → winsorized={v_low:.3f}  (should == p05={p05:.3f})")

    assert v_high == pytest.approx(p95, abs=1e-6)
    assert v_low == pytest.approx(p05, abs=1e-6)


def test_zscore():
    norm = ValueNormalizer()
    # 100 values with known mean=10, std≈1
    vals = [10.0] * 50 + [11.0] * 25 + [9.0] * 25  # mean=10, std=0.5
    norm.fit_from_records(_make_records("D", vals))

    mean = norm._stats["D"]["mean"]
    std = norm._stats["D"]["std"]

    # Value equal to mean should give z=0
    _, z_mean = norm.transform("D", mean)
    # Value 1 std above mean should give z≈1
    _, z_one_std = norm.transform("D", mean + std)

    print(f"\n[test_zscore] mean={mean:.4f}, std={std:.4f}")
    print(f"  transform(mean) → z={z_mean:.6f}  (expected 0)")
    print(f"  transform(mean+std) → z={z_one_std:.6f}  (expected 1)")

    assert z_mean == pytest.approx(0.0, abs=1e-5)
    assert z_one_std == pytest.approx(1.0, abs=1e-4)


def test_zscore_std_zero():
    """When all values are identical, std=0 → z_score should be 0."""
    norm = ValueNormalizer()
    norm.fit_from_records(_make_records("E", [5.0] * 20))
    v, z = norm.transform("E", 5.0)
    print(f"\n[test_zscore_std_zero] v={v}, z={z}  (expected z=0)")
    assert z == 0.0


def test_missing_value():
    norm = _fit_known()
    v, z = norm.transform("A", None)
    print(f"\n[test_missing_value] transform('A', None) → (v={v}, z={z})")
    assert v == 0.0
    assert z == 0.0


def test_unseen_code():
    norm = _fit_known()
    v, z = norm.transform("UNSEEN_CODE_XYZ", 42.0)
    print(f"\n[test_unseen_code] transform('UNSEEN_CODE_XYZ', 42.0) → (v={v}, z={z})")
    assert v == 0.0
    assert z == 0.0


def test_save_load_roundtrip():
    norm = _fit_known()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        norm.save(path)
        norm2 = ValueNormalizer.load(path)

        assert len(norm2) == len(norm)
        for code in norm._stats:
            for key in ("p05", "p95", "mean", "std"):
                assert norm2._stats[code][key] == pytest.approx(
                    norm._stats[code][key], rel=1e-9
                ), f"Mismatch after roundtrip: code={code}, key={key}"

        # Transform should give same result after reload
        v1, z1 = norm.transform("A", 5.0)
        v2, z2 = norm2.transform("A", 5.0)
        print(f"\n[test_save_load_roundtrip] before: (v={v1:.5f}, z={z1:.5f})")
        print(f"  after reload: (v={v2:.5f}, z={z2:.5f})")
        assert v1 == pytest.approx(v2, rel=1e-9)
        assert z1 == pytest.approx(z2, rel=1e-9)
    finally:
        os.unlink(path)


def test_transform_sequence():
    norm = _fit_known()
    codes = ["A", "B", "A", "UNSEEN"]
    values = [5.0, 3.0, None, 99.0]
    ws, zs = norm.transform_sequence(codes, values)
    print("\n[test_transform_sequence]")
    for c, v, w, z in zip(codes, values, ws, zs):
        print(f"  code={c}, value={v} → winsorized={w:.4f}, z={z:.4f}")
    assert len(ws) == 4
    assert ws[2] == 0.0   # missing
    assert ws[3] == 0.0   # unseen


if __name__ == "__main__":
    import traceback
    tests = [
        test_fit_stores_stats,
        test_winsorize,
        test_zscore,
        test_zscore_std_zero,
        test_missing_value,
        test_unseen_code,
        test_save_load_roundtrip,
        test_transform_sequence,
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
