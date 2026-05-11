"""
PyTorch Dataset for MEDS-format EHR data.

Data storage
------------
The full split is stored as a single polars DataFrame (self._df) with a
compact subject_id → (start_row, end_row) index (self._row_index).

Event objects are created lazily in __getitem__ for ONE subject at a time,
not for the entire split at startup.  This avoids:
  • 20-minute upfront Python loops over 50 M rows
  • 10–20 GB pickle files that bust disk quotas

Two task modes:

  "pretrain"   — one sample per subject, full event sequence returned as-is.
                 Stochastic windowing to context length is handled later by
                 the collator so each training step sees a fresh window.

  "prediction" — one sample per ACES row (subject_id, prediction_time, label).
                 Events after prediction_time are filtered out to prevent leakage.
                 If the remaining sequence exceeds max_seq_len, header-preserving
                 truncation is applied (see _truncate_with_header).

Each __getitem__ returns a dict:
  {
    "subject_id":  int,
    "codes":       List[int],              # vocab-encoded code indices
    "raw_codes":   List[str],              # original code strings (for debugging)
    "times":       List[pd.Timestamp],
    "values":      List[Optional[float]],
    "z_scores":    List[float],            # z-scored numeric value (0.0 if missing)
    "delta_times": List[float],            # log(1 + hours_since_prev_event)
    "label":       int,
  }
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset

from data.meds_parser import (
    Event,
    extract_header,
    get_age_from_header,
    is_header_code,
    df_slice_to_events,
    load_or_build_polars_cache,
)
from data.normalizer import ValueNormalizer
from data.vocab import Vocab


class MEDSDataset(Dataset):
    """
    Parameters
    ----------
    data_dir:
        Root directory containing split subdirectories.
    vocab:
        Vocab instance used to encode code strings → integer indices.
    split:
        "train", "tuning", or "held_out".
    task:
        "pretrain" or "prediction".
    max_seq_len:
        Maximum sequence length for header-preserving truncation in
        prediction mode.  In pretrain mode the full sequence is returned
        and the collator handles windowing.
    aces_label_path:
        Path to an ACES-format parquet/CSV with columns:
        subject_id, prediction_time, label.
        Required when task == "prediction".
    normalizer:
        Optional ValueNormalizer.  When provided, z_scores are computed;
        otherwise z_scores are all 0.0.
    time_unit:
        Unit for delta_time computation.  "hours" (recommended) or "seconds".
    cache_dir:
        Directory for the parquet+index cache.  On first run the combined
        DataFrame and row index are written here.  On subsequent runs they
        are loaded directly.  Set to None to disable caching.
    """

    def __init__(
        self,
        data_dir: str,
        vocab: Vocab,
        split: str,
        task: str = "pretrain",
        max_seq_len: int = 512,
        aces_label_path: Optional[str] = None,
        normalizer: Optional[ValueNormalizer] = None,
        time_unit: str = "hours",
        cache_dir: Optional[str] = None,
        max_files: Optional[int] = None,
    ):
        if task not in ("pretrain", "prediction"):
            raise ValueError(f"task must be 'pretrain' or 'prediction', got '{task}'")
        if task == "prediction" and aces_label_path is None:
            raise ValueError("aces_label_path is required for task='prediction'")

        self.vocab = vocab
        self.task = task
        self.max_seq_len = max_seq_len
        self.normalizer = normalizer
        self.time_unit = time_unit

        # Load the full split as a polars DataFrame + a tiny row index dict.
        # No Python Event objects are created here — they are built lazily
        # in __getitem__ for one subject at a time.
        self._df, self._row_index = load_or_build_polars_cache(
            data_dir, split, cache_dir=cache_dir, max_files=max_files,
        )

        if task == "pretrain":
            self.samples: List[Dict[str, Any]] = [
                {"subject_id": sid, "prediction_time": None, "label": 0}
                for sid in sorted(self._row_index.keys())
            ]
        else:
            self.samples = self._load_aces_samples(aces_label_path)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # ACES loading
    # ------------------------------------------------------------------

    def _load_aces_samples(self, path: str) -> List[Dict[str, Any]]:
        if path.endswith(".parquet"):
            aces = pd.read_parquet(path)
        else:
            aces = pd.read_csv(path)

        if not pd.api.types.is_datetime64_any_dtype(aces["prediction_time"]):
            aces["prediction_time"] = pd.to_datetime(aces["prediction_time"])

        samples = []
        skipped = 0
        for _, row in aces.iterrows():
            sid = int(row["subject_id"])
            if sid not in self._row_index:
                skipped += 1
                continue
            samples.append({
                "subject_id": sid,
                "prediction_time": row["prediction_time"],
                "label": int(row["label"]) if not pd.isna(row["label"]) else 0,
            })

        if skipped:
            print(f"[MEDSDataset] Skipped {skipped} ACES rows with no matching subject.")
        return samples

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------

    def _get_events(self, subject_id: int) -> List[Event]:
        """Lazily convert one subject's rows from polars to Event objects."""
        start, end = self._row_index[subject_id]
        return df_slice_to_events(self._df[start:end])

    def _apply_time_cutoff(
        self,
        events: List[Event],
        prediction_time: pd.Timestamp,
    ) -> List[Event]:
        """Keep only events at or before prediction_time."""
        return [e for e in events if e.time <= prediction_time]

    def _truncate_with_header(self, events: List[Event]) -> List[Event]:
        """
        Header-preserving truncation for prediction sequences longer than
        max_seq_len.

        1. Separate the leading header tokens (AGE, GENDER, RACE, BMI).
        2. Identify the first clinical event in the tail window.
        3. Update AGE by adding round(years_elapsed) to keep it accurate.
        4. Return header + most-recent (max_seq_len - len(header)) events.
        """
        if len(events) <= self.max_seq_len:
            return events

        header = extract_header(events)
        n_header = len(header)
        tail_budget = self.max_seq_len - n_header

        clinical_events = events[n_header:]
        tail = clinical_events[-tail_budget:] if tail_budget > 0 else []

        updated_header = list(header)
        if tail and n_header > 0:
            original_age = get_age_from_header(header)
            header_time  = header[0].time
            tail_start   = tail[0].time

            if original_age is not None:
                delta_days  = (tail_start - header_time).total_seconds() / 86400.0
                new_age     = original_age + round(delta_days / 365.25)
                age_event   = header[0]
                updated_header[0] = Event(
                    time=age_event.time,
                    code=age_event.code,
                    numeric_value=float(new_age),
                )

        return updated_header + tail

    def _compute_delta_times(self, events: List[Event]) -> List[float]:
        """
        Compute log(1 + delta_time) for each event.
        Header events (NaT) → 0.0, first real event → 0.0.
        """
        seconds_per_unit = 3600.0 if self.time_unit == "hours" else 1.0
        delta_times: List[float] = []
        prev_time: Optional[pd.Timestamp] = None

        for e in events:
            if pd.isna(e.time):
                delta_times.append(0.0)
                continue
            if prev_time is None or pd.isna(prev_time):
                delta_times.append(0.0)
            else:
                diff_seconds = (e.time - prev_time).total_seconds()
                diff_units   = max(0.0, diff_seconds / seconds_per_unit)
                delta_times.append(math.log(1.0 + diff_units))
            prev_time = e.time

        return delta_times

    def _compute_z_scores(self, events: List[Event]) -> List[float]:
        """Compute z-scores using the normalizer; returns 0.0 if no normalizer."""
        if self.normalizer is None:
            return [0.0] * len(events)
        _, z_scores = self.normalizer.transform_sequence(
            [e.code for e in events],
            [e.numeric_value for e in events],
        )
        return z_scores

    def _encode_events(self, events: List[Event]) -> Dict[str, Any]:
        """Convert a list of Events into the dict returned by __getitem__."""
        return {
            "codes":       [self.vocab.encode(e.code) for e in events],
            "raw_codes":   [e.code for e in events],
            "times":       [e.time for e in events],
            "values":      [e.numeric_value for e in events],
            "z_scores":    self._compute_z_scores(events),
            "delta_times": self._compute_delta_times(events),
        }

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample     = self.samples[idx]
        subject_id = sample["subject_id"]
        label      = sample["label"]

        # Lazy: convert only this subject's ~200 rows to Events
        events = self._get_events(subject_id)

        if self.task == "prediction":
            events = self._apply_time_cutoff(events, sample["prediction_time"])
            events = self._truncate_with_header(events)

        encoded = self._encode_events(events)
        return {
            "subject_id": subject_id,
            "label":      label,
            **encoded,
        }
