"""Tests for data/ar_collator.py."""

from __future__ import annotations

import torch

from data.ar_collator import ARCollator


def _fake_sample(n: int, code_base: int = 0) -> dict:
    return {
        "codes": list(range(code_base, code_base + n)),
        "values": [1.0] * n,
        "z_scores": [0.0] * n,
        "delta_times": [0.1] * n,
        "value_mask": [1] * n,
        "times": [None] * n,
        "labels": 0,
        "subject_id": code_base,
    }


def test_ar_collator_single_segment_metadata():
    collator = ARCollator(pad_idx=0, max_len=8, pack_sequences=False, seed=0)
    batch = [_fake_sample(5, 10)]
    out = collator(batch)
    assert out["codes"].shape == (1, 8)
    assert out["attention_mask"].sum() == 5
    assert out["segment_starts"][0, 0].item() == 0
    assert out["segment_lengths"][0, 0].item() == 5


def test_ar_collator_packs_multiple_subjects():
    collator = ARCollator(pad_idx=0, max_len=12, pack_sequences=True, seed=42)
    batch = [_fake_sample(4, 0), _fake_sample(3, 100), _fake_sample(5, 200)]
    out = collator(batch)
    assert out["codes"].shape[0] >= 1
    real = int(out["attention_mask"].sum().item())
    assert real == 4 + 3 + 5
    n_segs = (out["segment_lengths"][0] > 0).sum().item()
    assert n_segs == 3


def test_ar_collator_preserves_tensor_fields():
    collator = ARCollator(pad_idx=0, max_len=6, pack_sequences=False, seed=1)
    batch = [_fake_sample(3)]
    out = collator(batch)
    assert "values" in out
    assert "delta_times" in out
    assert out["values"].shape == out["codes"].shape


if __name__ == "__main__":
    test_ar_collator_single_segment_metadata()
    test_ar_collator_packs_multiple_subjects()
    test_ar_collator_preserves_tensor_fields()
    print("test_ar_collator: all passed")
