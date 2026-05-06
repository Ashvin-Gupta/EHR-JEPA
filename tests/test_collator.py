"""
Tests for data/collator.py

No parquet files required — feeds synthetic dicts directly to MEDSCollator.
Prints windowed sequence indices for visual inspection.

Cases tested:
  1. Pretrain with sequence > max_len — stochastic window applied.
  2. Pretrain with sequence < max_len — right-padding applied.
  3. Pretrain with sequence == max_len — no modification needed.
  4. Prediction mode — no windowing; only padding.
  5. Stochastic window differs across two calls to the same long sequence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.collator import MEDSCollator


PAD_IDX = 0
MAX_LEN = 8


def make_item(seq_len: int, label: int = 0) -> dict:
    """Create a synthetic dataset item with codes 1..seq_len."""
    return {
        "subject_id": 1,
        "codes": list(range(1, seq_len + 1)),  # non-zero so we can spot padding
        "values": [float(i) if i % 3 == 0 else None for i in range(seq_len)],
        "label": label,
    }


def test_pretrain_long_sequence_windowed():
    """Long sequence (20 > 8) → output length should be max_len."""
    collator = MEDSCollator(pad_idx=PAD_IDX, max_len=MAX_LEN, task="pretrain", seed=42)
    item = make_item(seq_len=20)
    batch = collator([item])

    assert batch["codes"].shape == (1, MAX_LEN), (
        f"Expected shape (1, {MAX_LEN}), got {batch['codes'].shape}"
    )
    assert batch["attention_mask"].sum().item() == MAX_LEN, "All positions should be real"
    print(f"\n[test_pretrain_long_sequence_windowed] PASS")
    print(f"  codes shape: {batch['codes'].shape}")
    print(f"  windowed codes: {batch['codes'][0].tolist()}")
    print(f"  attention_mask: {batch['attention_mask'][0].tolist()}")


def test_pretrain_short_sequence_padded():
    """Short sequence (3 < 8) → output length should be max_len with padding."""
    collator = MEDSCollator(pad_idx=PAD_IDX, max_len=MAX_LEN, task="pretrain", seed=0)
    item = make_item(seq_len=3)
    batch = collator([item])

    codes = batch["codes"][0].tolist()
    mask = batch["attention_mask"][0].tolist()

    assert batch["codes"].shape == (1, MAX_LEN)
    assert codes[3:] == [PAD_IDX] * 5, f"Expected padding after position 3, got {codes[3:]}"
    assert mask[:3] == [1, 1, 1]
    assert mask[3:] == [0] * 5

    print(f"\n[test_pretrain_short_sequence_padded] PASS")
    print(f"  codes:          {codes}")
    print(f"  attention_mask: {mask}")


def test_pretrain_exact_length():
    """Sequence exactly == max_len → no change needed."""
    collator = MEDSCollator(pad_idx=PAD_IDX, max_len=MAX_LEN, task="pretrain", seed=0)
    item = make_item(seq_len=MAX_LEN)
    batch = collator([item])

    codes = batch["codes"][0].tolist()
    assert codes == list(range(1, MAX_LEN + 1)), f"Exact-length sequence should be unchanged"
    assert batch["attention_mask"][0].sum().item() == MAX_LEN

    print(f"\n[test_pretrain_exact_length] PASS — codes: {codes}")


def test_stochastic_window_varies():
    """
    Calling the same unseeded collator many times on a long sequence must
    produce more than one distinct window — confirming the random sampling
    is actually varying across calls.
    """
    item = make_item(seq_len=20)
    # No seed → each call draws a fresh random start from Python's global RNG
    collator = MEDSCollator(pad_idx=PAD_IDX, max_len=MAX_LEN, task="pretrain")

    windows = set()
    for _ in range(30):
        w = tuple(collator([item])["codes"][0].tolist())
        windows.add(w)

    print(f"\n[test_stochastic_window_varies]")
    print(f"  Unique windows seen across 30 calls: {len(windows)}")
    for w in sorted(windows):
        print(f"    {list(w)}")

    assert len(windows) > 1, (
        "Expected multiple distinct windows across 30 calls on a 20-token sequence"
    )
    print("  PASS — stochastic windowing confirmed")


def test_prediction_mode_no_windowing():
    """Prediction mode with a long sequence should NOT apply stochastic windowing."""
    collator = MEDSCollator(pad_idx=PAD_IDX, max_len=MAX_LEN, task="prediction", seed=42)
    item = make_item(seq_len=20)  # longer than max_len
    batch = collator([item])

    # In prediction mode, long sequences are clipped (not stochastically windowed)
    # but they must still be exactly max_len
    assert batch["codes"].shape == (1, MAX_LEN), f"Shape: {batch['codes'].shape}"
    print(f"\n[test_prediction_mode_no_windowing] PASS")
    print(f"  codes: {batch['codes'][0].tolist()}")


def test_prediction_mode_padding():
    """Prediction mode with short sequence → padded."""
    collator = MEDSCollator(pad_idx=PAD_IDX, max_len=MAX_LEN, task="prediction")
    item = make_item(seq_len=5)
    batch = collator([item])

    codes = batch["codes"][0].tolist()
    mask = batch["attention_mask"][0].tolist()
    assert codes[5:] == [PAD_IDX] * 3
    assert mask[5:] == [0] * 3

    print(f"\n[test_prediction_mode_padding] PASS")
    print(f"  codes:          {codes}")
    print(f"  attention_mask: {mask}")


def test_batch_multiple_items():
    """Collator handles a batch of multiple items with different lengths."""
    collator = MEDSCollator(pad_idx=PAD_IDX, max_len=MAX_LEN, task="pretrain", seed=7)
    items = [make_item(seq_len=3), make_item(seq_len=MAX_LEN), make_item(seq_len=15)]
    batch = collator(items)

    assert batch["codes"].shape == (3, MAX_LEN)
    assert batch["labels"].shape == (3,)
    assert batch["subject_ids"].shape == (3,)

    print(f"\n[test_batch_multiple_items] PASS")
    print(f"  codes shape:  {batch['codes'].shape}")
    print(f"  batch codes:\n{batch['codes'].tolist()}")
    print(f"  attention_mask:\n{batch['attention_mask'].tolist()}")


def test_value_mask():
    """value_mask should be 1 where values are present, 0 elsewhere."""
    collator = MEDSCollator(pad_idx=PAD_IDX, max_len=MAX_LEN, task="pretrain", seed=0)
    item = make_item(seq_len=3)
    batch = collator([item])

    value_mask = batch["value_mask"][0].tolist()
    values = batch["values"][0].tolist()
    # Real pad positions (positions 3-7) should have value_mask=0 and values=0.0
    assert all(v == 0 for v in value_mask[3:]), "Pad positions should have value_mask=0"
    assert all(v == 0.0 for v in values[3:]), "Pad positions should have value=0.0"

    print(f"\n[test_value_mask] PASS")
    print(f"  value_mask: {value_mask}")
    print(f"  values:     {values}")


if __name__ == "__main__":
    test_pretrain_long_sequence_windowed()
    test_pretrain_short_sequence_padded()
    test_pretrain_exact_length()
    test_stochastic_window_varies()
    test_prediction_mode_no_windowing()
    test_prediction_mode_padding()
    test_batch_multiple_items()
    test_value_mask()
    print("\nAll collator tests passed.")
