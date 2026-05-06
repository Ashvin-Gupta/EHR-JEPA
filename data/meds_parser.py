"""
MEDS parquet loader and subject sequence builder.

Each parquet file has columns: subject_id, time, code, numeric_value.
Every subject's event block begins with an AGE token — this is how subject
boundaries are identified when iterating a flat DataFrame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Event:
    """A single clinical event."""
    time: pd.Timestamp
    code: str
    numeric_value: Optional[float]


# Header codes that appear at the start of every subject's sequence.
# These are always the first 2–4 events and must never be treated as
# clinical observations in downstream models.
HEADER_CODES = {"AGE", "BMI"}

def is_header_code(code: str) -> bool:
    """Return True if the code is one of the fixed demographic header codes."""
    if code in HEADER_CODES:
        return True
    if code.startswith("GENDER//") or code.startswith("RACE//"):
        return True
    return False


def load_split(data_dir: str, split: str) -> pd.DataFrame:
    """
    Load all parquet files from data_dir/split/ and return a single DataFrame.

    Expected columns: subject_id (int64), time (datetime64), code (object),
    numeric_value (object).
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    parquet_files = sorted(
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {split_dir}")

    frames = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # numeric_value may be stored as object; coerce to float where possible
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce")

    return df


def build_subject_sequences(df: pd.DataFrame) -> Dict[int, List[Event]]:
    """
    Group a flat MEDS DataFrame into per-subject chronological event lists.

    Subject identity comes from the subject_id column directly — we group
    by that field rather than trying to detect AGE-token boundaries in a
    flat stream.

    NaT timestamps (used for demographic header rows: AGE, GENDER, RACE, BMI)
    are sorted to the front with na_position="first" so the header always
    leads the sequence.

    Returns
    -------
    Dict mapping subject_id -> List[Event] sorted by time (NaT first).
    """
    sequences: Dict[int, List[Event]] = {}

    for subject_id, group in df.groupby("subject_id", sort=False):
        # na_position="first" keeps NaT header rows at the start of the sequence
        group_sorted = group.sort_values("time", na_position="first")
        events = [
            Event(
                time=row["time"],
                code=str(row["code"]),
                numeric_value=row["numeric_value"] if pd.notna(row["numeric_value"]) else None,
            )
            for _, row in group_sorted.iterrows()
        ]
        sequences[int(subject_id)] = events

    return sequences


def extract_header(events: List[Event]) -> List[Event]:
    """
    Return the leading demographic header events for a subject's sequence.

    The header consists of consecutive events from the start that match
    is_header_code().  Typically: [AGE, GENDER//{x}, RACE//{x}, BMI].
    At most 4 tokens are considered (safety bound).
    """
    header = []
    for event in events[:4]:
        if is_header_code(event.code):
            header.append(event)
        else:
            break
    return header


def get_age_from_header(events: List[Event]) -> Optional[float]:
    """Return the numeric age value from the AGE token, or None if absent."""
    for event in events[:4]:
        if event.code == "AGE":
            return event.numeric_value
    return None
