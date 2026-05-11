"""
MEDS parquet loader and subject sequence builder.

Design
------
Instead of loading every parquet file and converting all rows to Python Event
objects upfront (which takes 20+ minutes and produces 10–20 GB of pickle data),
we load the entire split into a single compact **polars DataFrame** and build a
tiny subject-row index (subject_id → (start_row, end_row)).

Event objects are created lazily in MEDSDataset.__getitem__, one subject at a
time, via df_slice_to_events().  This means:

  • Startup time: pl.scan_parquet(292 files).collect()  → 1–3 min (parallel I/O)
  • Cache size  : one zstd-compressed parquet ≈ original data size (not 10 GB pickle)
  • Memory      : polars DataFrame for 50 M rows ≈ 1–2 GB  (vs 20+ GB Python objects)
  • Per-sample  : slice 200 rows + create 200 Events → microseconds

Cache
-----
load_or_build_polars_cache() writes two small files:
  {split}_{key}.parquet   — combined sorted polars DataFrame (zstd)
  {split}_{key}_index.pkl — Dict[subject_id → (start, end)]  (~4 MB)

The cache key is an MD5 of (filename, mtime, size) for all source parquet files,
so any data change forces a rebuild automatically.

Backwards-compatible helpers
-----------------------------
load_split() and build_subject_sequences() are kept for unit tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


@dataclass
class Event:
    """A single clinical event."""
    time: pd.Timestamp
    code: str
    numeric_value: Optional[float]


HEADER_CODES = {"AGE", "BMI"}


def is_header_code(code: str) -> bool:
    if code in HEADER_CODES:
        return True
    if code.startswith("GENDER//") or code.startswith("RACE//"):
        return True
    return False


# ---------------------------------------------------------------------------
# Polars-based primary loading path
# ---------------------------------------------------------------------------

def _sorted_parquet_files(data_dir: str, split: str) -> List[str]:
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    files = sorted(
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found in {split_dir}")
    return files


def _read_split_polars(parquet_files: List[str]):
    """
    Read all parquet files for a split into a single polars DataFrame.

    Uses scan_parquet (lazy, parallel I/O across files) then collect.
    Non-numeric values in numeric_value (e.g. "NEG") become null/NaN.
    """
    import polars as pl

    df = (
        pl.scan_parquet(parquet_files)
        .with_columns(pl.col("numeric_value").cast(pl.Float64, strict=False))
        .collect()
    )
    return df


def build_subject_row_index(df) -> Dict[int, Tuple[int, int]]:
    """
    Build a subject_id → (start_row, end_row) index from a pre-sorted DataFrame.

    Assumes rows are already contiguous per subject (no groupby / sort needed).
    Uses polars group_by for a fast vectorised scan.

    Returns a dict where sequences[sid] == df[start:end].
    """
    import polars as pl

    df_idx = df.with_row_index(name="_row")
    agg = (
        df_idx
        .group_by("subject_id", maintain_order=True)
        .agg([
            pl.col("_row").min().alias("start"),
            pl.col("_row").max().alias("end"),
        ])
    )
    index: Dict[int, Tuple[int, int]] = {
        int(row["subject_id"]): (int(row["start"]), int(row["end"]) + 1)
        for row in agg.iter_rows(named=True)
    }
    return index


def df_slice_to_events(df_slice) -> List[Event]:
    """
    Convert a polars DataFrame slice (one subject's rows) to a list of Events.

    Called per-subject in MEDSDataset.__getitem__, never for the full split.
    """
    import polars as pl

    times  = df_slice["time"].to_list()
    codes  = df_slice["code"].cast(pl.Utf8).to_list()
    values = df_slice["numeric_value"].to_list()
    return [
        Event(
            time=pd.Timestamp(t) if t is not None else pd.NaT,
            code=code,
            numeric_value=float(val) if val is not None else None,
        )
        for t, code, val in zip(times, codes, values)
    ]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _split_cache_key(data_dir: str, split: str) -> str:
    """MD5 over sorted (filename, mtime, size) tuples for all split parquet files."""
    split_dir = os.path.join(data_dir, split)
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".parquet"))
    file_info = [
        (f,
         os.path.getmtime(os.path.join(split_dir, f)),
         os.path.getsize(os.path.join(split_dir, f)))
        for f in files
    ]
    key_str = json.dumps({"dir": str(split_dir), "files": file_info}, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def load_or_build_polars_cache(
    data_dir: str,
    split: str,
    cache_dir: Optional[str] = None,
    max_files: Optional[int] = None,
):
    """
    Return (polars_df, row_index) for a split, using a parquet+index cache.

    Cache files:
      {split}_{key}.parquet      — combined zstd-compressed DataFrame
      {split}_{key}_index.pkl    — Dict[subject_id → (start, end)]  (~4 MB)

    On cache miss: reads all source parquet files via polars scan_parquet
    (parallel I/O), builds the row index, then tries to write the cache.
    Disk-quota errors during the cache write are caught and logged; the
    function still returns the in-memory result so training can continue.

    Parameters
    ----------
    max_files:
        If set, only the first N parquet files (alphabetically) are loaded.
        Caching is disabled when max_files is set to avoid polluting the
        full-split cache with a subset.
    """
    import polars as pl

    parquet_files = _sorted_parquet_files(data_dir, split)
    if max_files is not None and max_files < len(parquet_files):
        parquet_files = parquet_files[:max_files]
        print(f"  [max_files] Using {len(parquet_files)} of {len(_sorted_parquet_files(data_dir, split))} available files for split '{split}'")
        # Skip cache for subsets to avoid polluting the full-split cache.
        print(f"  scanning {len(parquet_files)} parquet files (no cache for subsets) …")
        df = _read_split_polars(parquet_files)
        print(f"  {len(df):,} rows loaded")
        row_index = build_subject_row_index(df)
        print(f"  {len(row_index):,} subjects indexed")
        return df, row_index

    # ---- Try cache ----
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key          = _split_cache_key(data_dir, split)
        cache_pq     = os.path.join(cache_dir, f"{split}_{key}.parquet")
        cache_idx    = os.path.join(cache_dir, f"{split}_{key}_index.pkl")

        if os.path.exists(cache_pq) and os.path.exists(cache_idx):
            print(f"  [cache] HIT  — loading {split} from {os.path.basename(cache_pq)}")
            df = pl.read_parquet(cache_pq)
            with open(cache_idx, "rb") as fh:
                row_index = pickle.load(fh)
            print(f"  [cache]       {len(row_index):,} subjects, {len(df):,} rows")
            return df, row_index

        print(f"  [cache] MISS — scanning {len(parquet_files)} parquet files …")
    else:
        print(f"  scanning {len(parquet_files)} parquet files …")

    # ---- Build ----
    df = _read_split_polars(parquet_files)
    print(f"  {len(df):,} rows loaded")

    print(f"  building subject row index …")
    row_index = build_subject_row_index(df)
    print(f"  {len(row_index):,} subjects indexed")

    # ---- Save cache (graceful on disk-quota error) ----
    if cache_dir is not None:
        try:
            print(f"  saving cache to {cache_dir} …")
            df.write_parquet(cache_pq, compression="zstd")
            with open(cache_idx, "wb") as fh:
                pickle.dump(row_index, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [cache]       saved ({split}_{key[:8]}…)")
            # Remove stale cache files for this split
            for fname in os.listdir(cache_dir):
                old_pq  = fname.startswith(f"{split}_") and fname.endswith(".parquet") and fname != os.path.basename(cache_pq)
                old_idx = fname.startswith(f"{split}_") and fname.endswith("_index.pkl") and fname != os.path.basename(cache_idx)
                if old_pq or old_idx:
                    os.remove(os.path.join(cache_dir, fname))
                    print(f"  [cache]       removed stale: {fname}")
        except OSError as e:
            print(f"  [cache] WARNING: could not save cache ({e})")
            print(f"  [cache] Continuing without cache — data will be reloaded next run.")
            print(f"  [cache] Tip: set data.cache_dir to a path with more disk space,")
            print(f"  [cache]      or set data.cache_dir: null to disable caching.")

    return df, row_index


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def extract_header(events: List[Event]) -> List[Event]:
    """Return the leading demographic header events (AGE, GENDER, RACE, BMI)."""
    header = []
    for event in events[:4]:
        if is_header_code(event.code):
            header.append(event)
        else:
            break
    return header


def get_age_from_header(events: List[Event]) -> Optional[float]:
    """Return the numeric age from the AGE token, or None if absent."""
    for event in events[:4]:
        if event.code == "AGE":
            return event.numeric_value
    return None


# ---------------------------------------------------------------------------
# Pandas fallback — kept for unit tests only
# ---------------------------------------------------------------------------

def load_split(data_dir: str, split: str) -> pd.DataFrame:
    """Load all parquet files for a split into a pandas DataFrame (used by tests)."""
    parquet_files = _sorted_parquet_files(data_dir, split)
    frames = []
    for fpath in tqdm(parquet_files, desc=f"  reading {split}", unit="file", leave=False):
        frames.append(pd.read_parquet(fpath))
    df = pd.concat(frames, ignore_index=True)
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce")
    return df


def build_subject_sequences(df: pd.DataFrame) -> Dict[int, List[Event]]:
    """Build subject sequences from a pandas DataFrame (used by tests)."""
    sequences: Dict[int, List[Event]] = {}
    groups = list(df.groupby("subject_id", sort=False))
    for subject_id, group in tqdm(groups, desc="  building sequences", unit="subject", leave=False):
        group_sorted = group.sort_values("time", na_position="first")
        events = [
            Event(
                time=row.time,
                code=str(row.code),
                numeric_value=row.numeric_value if pd.notna(row.numeric_value) else None,
            )
            for row in group_sorted.itertuples(index=False)
        ]
        sequences[int(subject_id)] = events
    return sequences
