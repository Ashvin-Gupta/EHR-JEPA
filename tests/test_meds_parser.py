"""
Tests for data/meds_parser.py

Uses synthetic in-memory DataFrames — no real parquet files required.
Prints sample output so results can be inspected visually.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from data.meds_parser import (
    build_subject_sequences,
    extract_header,
    get_age_from_header,
    is_header_code,
)


def make_synthetic_df():
    """
    Two subjects, each with a demographic header followed by clinical events.
    Subject 1: 65-year-old male  (7 events total)
    Subject 2: 45-year-old female (5 events total)
    Events are deliberately out of time order to test sorting.
    """
    rows = [
        # subject_id, time, code, numeric_value
        # --- Subject 1 ---
        (1, "2020-01-01 08:00", "LAB//50882//mEq/L",  22.0),
        (1, "2019-12-31 00:00", "AGE",                65.0),
        (1, "2019-12-31 00:01", "GENDER//M",          None),
        (1, "2019-12-31 00:02", "RACE//WHITE",        None),
        (1, "2019-12-31 00:03", "BMI",                28.5),
        (1, "2020-01-01 06:00", "DIAGNOSIS//ICD//9//25000", None),
        (1, "2020-01-02 09:00", "LAB//51221//g/dL",   13.0),
        # --- Subject 2 ---
        (2, "2021-03-10 12:00", "LAB//50912//mg/dL",  95.0),
        (2, "2021-03-08 00:00", "AGE",                45.0),
        (2, "2021-03-08 00:01", "GENDER//F",          None),
        (2, "2021-03-08 00:02", "RACE//BLACK",        None),
        (2, "2021-03-09 10:00", "DIAGNOSIS//ICD//10//E119", None),
    ]
    df = pd.DataFrame(rows, columns=["subject_id", "time", "code", "numeric_value"])
    df["time"] = pd.to_datetime(df["time"])
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce")
    return df


def test_subject_count():
    df = make_synthetic_df()
    seqs = build_subject_sequences(df)
    assert len(seqs) == 2, f"Expected 2 subjects, got {len(seqs)}"
    print(f"\n[test_subject_count] PASS — found {len(seqs)} subjects")


def test_sequence_lengths():
    df = make_synthetic_df()
    seqs = build_subject_sequences(df)
    assert len(seqs[1]) == 7, f"Subject 1 expected 7 events, got {len(seqs[1])}"
    assert len(seqs[2]) == 5, f"Subject 2 expected 5 events, got {len(seqs[2])}"
    print(f"[test_sequence_lengths] PASS — subject 1: {len(seqs[1])} events, "
          f"subject 2: {len(seqs[2])} events")


def test_time_sorted():
    df = make_synthetic_df()
    seqs = build_subject_sequences(df)
    for sid, events in seqs.items():
        times = [e.time for e in events]
        assert times == sorted(times), f"Subject {sid} events not sorted by time"
    print("[test_time_sorted] PASS — all subjects have time-sorted events")


def test_first_event_is_age():
    df = make_synthetic_df()
    seqs = build_subject_sequences(df)
    for sid, events in seqs.items():
        assert events[0].code == "AGE", (
            f"Subject {sid} first event should be AGE, got {events[0].code}"
        )
    print("[test_first_event_is_age] PASS — first event is AGE for all subjects")


def test_is_header_code():
    assert is_header_code("AGE")
    assert is_header_code("GENDER//F")
    assert is_header_code("GENDER//M")
    assert is_header_code("RACE//WHITE")
    assert is_header_code("BMI")
    assert not is_header_code("LAB//50882//mEq/L")
    assert not is_header_code("DIAGNOSIS//ICD//9//25000")
    print("[test_is_header_code] PASS")


def test_extract_header():
    df = make_synthetic_df()
    seqs = build_subject_sequences(df)

    header1 = extract_header(seqs[1])
    codes1 = [e.code for e in header1]
    assert codes1 == ["AGE", "GENDER//M", "RACE//WHITE", "BMI"], f"Unexpected header: {codes1}"

    header2 = extract_header(seqs[2])
    codes2 = [e.code for e in header2]
    assert codes2 == ["AGE", "GENDER//F", "RACE//BLACK"], f"Unexpected header: {codes2}"

    print(f"[test_extract_header] PASS")
    print(f"  Subject 1 header: {codes1}")
    print(f"  Subject 2 header: {codes2}")


def test_get_age():
    df = make_synthetic_df()
    seqs = build_subject_sequences(df)
    age1 = get_age_from_header(seqs[1])
    age2 = get_age_from_header(seqs[2])
    assert age1 == 65.0, f"Expected 65.0, got {age1}"
    assert age2 == 45.0, f"Expected 45.0, got {age2}"
    print(f"[test_get_age] PASS — subject 1 age={age1}, subject 2 age={age2}")


def test_sample_output():
    """Print a full sample sequence for visual inspection."""
    df = make_synthetic_df()
    seqs = build_subject_sequences(df)
    print("\n--- Sample output: Subject 1 events ---")
    for i, e in enumerate(seqs[1]):
        print(f"  [{i}] time={e.time}  code={e.code!r:40s}  value={e.numeric_value}")


if __name__ == "__main__":
    test_subject_count()
    test_sequence_lengths()
    test_time_sorted()
    test_first_event_is_age()
    test_is_header_code()
    test_extract_header()
    test_get_age()
    test_sample_output()
    print("\nAll meds_parser tests passed.")
