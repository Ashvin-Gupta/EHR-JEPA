"""
Value normalizer for EHR-JEPA.

Computes per-code normalization statistics from training data only.

Steps:
  1. Winsorize values at 5th / 95th percentile per code.
  2. Compute mean and standard deviation on the winsorized values.
  3. At inference time, transform (value, code) → (winsorized_value, z_score).

Edge cases:
  - std == 0        → z_score = 0.0
  - missing value   → returns (0.0, 0.0)
  - unseen code     → returns (0.0, 0.0)

Usage
-----
  normalizer = ValueNormalizer()
  normalizer.fit(data_dir, split="train")
  normalizer.save("normalizer_stats.json")

  normalizer = ValueNormalizer.load("normalizer_stats.json")
  val, z = normalizer.transform("LAB//50882//mEq/L", 4.2)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ValueNormalizer:
    """Per-code value normalizer using winsorization + z-scoring."""

    def __init__(self) -> None:
        # Maps code string → {"p05", "p95", "mean", "std"}
        self._stats: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data_dir: str, split: str = "train") -> "ValueNormalizer":
        """
        Iterate all parquet files in data_dir/split, collecting
        (code, numeric_value) pairs, then compute per-code statistics.

        Only real (non-None, non-NaN) numeric values are used.
        """
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Accumulate values per code
        code_values: Dict[str, List[float]] = {}

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".parquet"):
                continue
            fpath = os.path.join(split_dir, fname)
            df = pd.read_parquet(fpath, columns=["code", "numeric_value"])
            df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce")
            df = df.dropna(subset=["numeric_value"])

            for code, val in zip(df["code"], df["numeric_value"]):
                if code not in code_values:
                    code_values[code] = []
                code_values[code].append(float(val))

        self._stats = {}
        for code, vals in code_values.items():
            arr = np.array(vals, dtype=np.float64)
            p05 = float(np.percentile(arr, 5))
            p95 = float(np.percentile(arr, 95))
            arr_clipped = np.clip(arr, p05, p95)
            mean = float(arr_clipped.mean())
            std = float(arr_clipped.std())
            self._stats[code] = {"p05": p05, "p95": p95, "mean": mean, "std": std}

        return self

    def fit_from_records(
        self, records: List[Tuple[str, float]]
    ) -> "ValueNormalizer":
        """
        Fit from an in-memory list of (code, value) tuples.
        Useful for unit testing without parquet files.
        """
        code_values: Dict[str, List[float]] = {}
        for code, val in records:
            if code not in code_values:
                code_values[code] = []
            code_values[code].append(float(val))

        self._stats = {}
        for code, vals in code_values.items():
            arr = np.array(vals, dtype=np.float64)
            p05 = float(np.percentile(arr, 5))
            p95 = float(np.percentile(arr, 95))
            arr_clipped = np.clip(arr, p05, p95)
            mean = float(arr_clipped.mean())
            std = float(arr_clipped.std())
            self._stats[code] = {"p05": p05, "p95": p95, "mean": mean, "std": std}

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(
        self, code: str, value: Optional[float]
    ) -> Tuple[float, float]:
        """
        Returns (winsorized_value, z_score).

        - Missing value (None / NaN)  → (0.0, 0.0)
        - Unseen code                 → (0.0, 0.0)
        - std == 0                    → (winsorized_value, 0.0)
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0, 0.0

        stats = self._stats.get(code)
        if stats is None:
            return 0.0, 0.0

        v_clipped = float(np.clip(value, stats["p05"], stats["p95"]))
        std = stats["std"]
        if std == 0.0:
            return v_clipped, 0.0

        z = (v_clipped - stats["mean"]) / std
        return v_clipped, float(z)

    def transform_sequence(
        self, codes: List[str], values: List[Optional[float]]
    ) -> Tuple[List[float], List[float]]:
        """Vectorised transform for a full event sequence."""
        winsorized, z_scores = [], []
        for code, val in zip(codes, values):
            v, z = self.transform(code, val)
            winsorized.append(v)
            z_scores.append(z)
        return winsorized, z_scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ValueNormalizer":
        inst = cls()
        with open(path) as f:
            inst._stats = json.load(f)
        return inst

    def __len__(self) -> int:
        return len(self._stats)
