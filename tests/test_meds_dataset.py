"""
Tests for data/meds_dataset.py

Writes tiny synthetic parquet files to a temp directory so the full
load → parse → dataset pipeline runs end-to-end.

Cases tested:
  1. Pretrain mode — full sequence returned as-is.
  2. Prediction mode — time-cutoff filtering prevents leakage.
  3. Prediction mode with header-preserving truncation — sequence longer
     than max_seq_len; checks AGE value is updated correctly.
"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from data.meds_dataset import MEDSDataset
from data.vocab import build_vocab_from_codes


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_parquet(events: list, directory: str, filename: str = "part0.parquet"):
    """Write a list of (subject_id, time, code, numeric_value) rows as parquet."""
    df = pd.DataFrame(events, columns=["subject_id", "time", "code", "numeric_value"])
    df["time"] = pd.to_datetime(df["time"])
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce")
    os.makedirs(directory, exist_ok=True)
    df.to_parquet(os.path.join(directory, filename), index=False)
    return df


def _make_aces(rows: list, directory: str, filename: str = "labels.parquet"):
    """Write ACES label rows as parquet."""
    df = pd.DataFrame(rows, columns=["subject_id", "prediction_time", "label"])
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    df.to_parquet(path, index=False)
    return path


EVENTS = [
    # Subject 1 (65yo male) — 8 clinical events after header
    (1, "2019-12-31", "AGE",                    65.0),
    (1, "2019-12-31", "GENDER//M",              None),
    (1, "2019-12-31", "RACE//WHITE",            None),
    (1, "2019-12-31", "BMI",                    28.5),
    (1, "2020-01-01", "LAB//50882//mEq/L",      22.0),
    (1, "2020-01-02", "DIAGNOSIS//ICD//9//25000", None),
    (1, "2020-01-03", "LAB//51221//g/dL",       13.0),
    (1, "2020-01-04", "LAB//50912//mg/dL",      95.0),
    (1, "2020-01-05", "PROCEDURE//ICD//9//9904", None),
    (1, "2020-01-06", "LAB//50882//mEq/L",      21.5),
    (1, "2020-01-07", "LAB//51221//g/dL",       12.8),
    (1, "2020-01-08", "DIAGNOSIS//ICD//9//25000", None),
    # Subject 2 (45yo female) — 3 clinical events after header
    (2, "2021-03-08", "AGE",                    45.0),
    (2, "2021-03-08", "GENDER//F",              None),
    (2, "2021-03-08", "RACE//BLACK",            None),
    (2, "2021-03-09", "LAB//50912//mg/dL",      88.0),
    (2, "2021-03-10", "DIAGNOSIS//ICD//10//E119", None),
    (2, "2021-03-11", "LAB//51221//g/dL",       11.5),
]

ALL_CODES = list(set(row[2] for row in EVENTS))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def setup_data(tmp_dir):
    train_dir = os.path.join(tmp_dir, "train")
    _make_parquet(EVENTS, train_dir)
    vocab = build_vocab_from_codes(ALL_CODES, embedding_type="learned", top_k=20)
    return tmp_dir, vocab


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pretrain_mode():
    tmp = tempfile.mkdtemp()
    try:
        data_dir, vocab = setup_data(tmp)
        ds = MEDSDataset(data_dir=data_dir, vocab=vocab, split="train",
                         task="pretrain", max_seq_len=512)
        assert len(ds) == 2, f"Expected 2 subjects, got {len(ds)}"

        item = ds[0]
        assert "codes" in item and "values" in item and "times" in item
        assert item["label"] == 0
        print(f"\n[test_pretrain_mode] PASS — {len(ds)} subjects")
        print(f"  Sample item keys: {list(item.keys())}")
        print(f"  Subject {item['subject_id']} sequence length: {len(item['codes'])}")
        print(f"  First 4 codes (raw): {item['raw_codes'][:4]}")
        print(f"  First 4 encoded:     {item['codes'][:4]}")
    finally:
        shutil.rmtree(tmp)


def test_pretrain_full_sequence_returned():
    """Pretrain dataset returns the full sequence without truncation."""
    tmp = tempfile.mkdtemp()
    try:
        data_dir, vocab = setup_data(tmp)
        # Subject 1 has 12 events; even with max_seq_len=5 (small), pretrain returns all
        ds = MEDSDataset(data_dir=data_dir, vocab=vocab, split="train",
                         task="pretrain", max_seq_len=5)
        # Find subject 1
        item = next(i for i in (ds[0], ds[1]) if i["subject_id"] == 1)
        assert len(item["codes"]) == 12, (
            f"Expected 12 events for pretrain (no truncation), got {len(item['codes'])}"
        )
        print(f"[test_pretrain_full_sequence_returned] PASS — returned {len(item['codes'])} events")
    finally:
        shutil.rmtree(tmp)


def test_prediction_time_cutoff():
    """Events after prediction_time must be excluded."""
    tmp = tempfile.mkdtemp()
    try:
        data_dir, vocab = setup_data(tmp)
        aces_path = _make_aces(
            [(1, "2020-01-03 23:59:59", 1)],  # cutoff just before 2020-01-04
            tmp,
        )
        ds = MEDSDataset(data_dir=data_dir, vocab=vocab, split="train",
                         task="prediction", max_seq_len=512,
                         aces_label_path=aces_path)
        assert len(ds) == 1
        item = ds[0]
        assert item["label"] == 1

        # Events at 2020-01-04 and later must not be present
        for t in item["times"]:
            assert t <= pd.Timestamp("2020-01-03 23:59:59"), (
                f"Event after prediction_time found: {t}"
            )

        # Expected codes in order: header (4) + LAB(01-01) + DIAG(01-02) + LAB(01-03) = 7
        assert len(item["codes"]) == 7, (
            f"Expected 7 events up to cutoff, got {len(item['codes'])}"
        )
        print(f"\n[test_prediction_time_cutoff] PASS — {len(item['codes'])} events before cutoff")
        print(f"  Times: {[str(t.date()) for t in item['times']]}")
    finally:
        shutil.rmtree(tmp)


def test_header_preserving_truncation():
    """
    When the sequence after time-cutoff exceeds max_seq_len, the header is
    preserved and the AGE value is updated by the elapsed years.
    """
    tmp = tempfile.mkdtemp()
    try:
        data_dir, vocab = setup_data(tmp)
        # All 12 events of subject 1 are before the cutoff date
        aces_path = _make_aces(
            [(1, "2020-01-10", 0)],
            tmp,
        )
        # max_seq_len=6: header (4) + 2 tail events
        ds = MEDSDataset(data_dir=data_dir, vocab=vocab, split="train",
                         task="prediction", max_seq_len=6,
                         aces_label_path=aces_path)
        item = ds[0]

        assert len(item["codes"]) == 6, (
            f"Expected exactly 6 events after truncation, got {len(item['codes'])}"
        )

        # First 4 codes must be the header
        assert item["raw_codes"][0] == "AGE"
        assert item["raw_codes"][1] == "GENDER//M"
        assert item["raw_codes"][2] == "RACE//WHITE"
        assert item["raw_codes"][3] == "BMI"

        # The last 2 codes are the most recent clinical events
        assert item["raw_codes"][4] == "LAB//51221//g/dL",  item["raw_codes"]
        assert item["raw_codes"][5] == "DIAGNOSIS//ICD//9//25000", item["raw_codes"]

        # AGE value: original 65, tail starts at 2020-01-07 from header 2019-12-31
        # Δ ≈ 7 days ≈ 0 years rounded → new AGE = 65
        age_value = item["values"][0]
        assert age_value == 65.0, f"Expected AGE=65 (no full year elapsed), got {age_value}"

        print(f"\n[test_header_preserving_truncation] PASS")
        print(f"  Total events after truncation: {len(item['codes'])}")
        print(f"  Raw codes: {item['raw_codes']}")
        print(f"  AGE value (updated): {age_value}")
        print(f"  Tail times: {[str(t.date()) for t in item['times'][-2:]]}")
    finally:
        shutil.rmtree(tmp)


def test_header_age_updated_after_years():
    """
    Simulate a long time gap so that AGE should be incremented.
    We fabricate events spanning ~2 years to trigger the rounding.
    """
    tmp = tempfile.mkdtemp()
    try:
        # Build a custom event set with a 2-year gap
        events = [
            (3, "2015-01-01", "AGE",   50.0),
            (3, "2015-01-01", "GENDER//F", None),
            (3, "2015-01-01", "RACE//WHITE", None),
            (3, "2015-01-02", "LAB//50882//mEq/L", 20.0),
            (3, "2015-06-01", "LAB//51221//g/dL",  12.0),
            (3, "2016-01-01", "LAB//50912//mg/dL",  90.0),
            (3, "2016-06-01", "DIAGNOSIS//ICD//9//25000", None),
            (3, "2017-01-01", "LAB//50882//mEq/L",  21.0),  # ~2 years after header
        ]
        train_dir = os.path.join(tmp, "train")
        _make_parquet(events, train_dir)
        codes = list(set(e[2] for e in events))
        vocab = build_vocab_from_codes(codes, embedding_type="learned", top_k=20)

        aces_path = _make_aces([(3, "2017-06-01", 1)], tmp)
        # max_seq_len=5: header (3, no BMI) + 2 tail events
        ds = MEDSDataset(data_dir=tmp, vocab=vocab, split="train",
                         task="prediction", max_seq_len=5,
                         aces_label_path=aces_path)
        item = ds[0]

        age_value = item["values"][0]
        # Header time: 2015-01-01; tail starts at 2016-06-01 (approx 1.5 yr → round to 2)
        # New AGE = 50 + 2 = 52
        print(f"\n[test_header_age_updated_after_years] AGE value = {age_value}")
        assert age_value >= 51, f"Expected age >= 51 after ~2yr gap, got {age_value}"
        print(f"  PASS — AGE updated from 50 to {age_value}")
        print(f"  raw_codes: {item['raw_codes']}")
    finally:
        shutil.rmtree(tmp)


def test_aces_skips_missing_subjects():
    """ACES rows for unknown subjects are silently skipped."""
    tmp = tempfile.mkdtemp()
    try:
        data_dir, vocab = setup_data(tmp)
        aces_path = _make_aces(
            [
                (1, "2020-01-05", 1),
                (999, "2020-01-05", 0),   # subject 999 not in data
            ],
            tmp,
        )
        ds = MEDSDataset(data_dir=data_dir, vocab=vocab, split="train",
                         task="prediction", max_seq_len=512,
                         aces_label_path=aces_path)
        assert len(ds) == 1, f"Expected 1 sample (skipping missing subject), got {len(ds)}"
        print("[test_aces_skips_missing_subjects] PASS")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_pretrain_mode()
    test_pretrain_full_sequence_returned()
    test_prediction_time_cutoff()
    test_header_preserving_truncation()
    test_header_age_updated_after_years()
    test_aces_skips_missing_subjects()
    print("\nAll meds_dataset tests passed.")
