"""
Vocabulary builder for MEDS event codes.

Two modes, selected by embedding_type:

  "learned"    — top-K most frequent codes from the training split + UNK.
                 Codes outside top-K are mapped to the UNK index at encode time.

  "text_based" — all unique codes observed in the training split + UNK.
                 No frequency cutoff; every code gets its own text-initialised
                 embedding, so rare but clinically important codes are preserved.

The UNK token is always the last index: vocab_size (i.e. len(code_to_idx) - 1
after building, but accessed via Vocab.unk_idx).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd


UNK_TOKEN = "<UNK>"


class Vocab:
    """Bidirectional code ↔ integer index mapping."""

    def __init__(self, code_to_idx: Dict[str, int], unk_idx: int):
        self.code_to_idx = code_to_idx
        self.idx_to_code: Dict[int, str] = {v: k for k, v in code_to_idx.items()}
        self.unk_idx = unk_idx

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, code: str) -> int:
        """Return integer index for code; returns unk_idx for unknowns."""
        return self.code_to_idx.get(code, self.unk_idx)

    def decode(self, idx: int) -> str:
        """Return code string for an integer index."""
        return self.idx_to_code.get(idx, UNK_TOKEN)

    def __len__(self) -> int:
        return len(self.code_to_idx)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "code_to_idx": self.code_to_idx,
            "unk_idx": self.unk_idx,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """Load vocabulary from a JSON file."""
        with open(path, "r") as f:
            payload = json.load(f)
        return cls(
            code_to_idx=payload["code_to_idx"],
            unk_idx=payload["unk_idx"],
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of entries including UNK."""
        return len(self.code_to_idx)

    @property
    def num_codes(self) -> int:
        """Number of real codes (excluding UNK)."""
        return len(self.code_to_idx) - 1


# ------------------------------------------------------------------
# Builder functions
# ------------------------------------------------------------------

def _collect_codes_from_parquets(data_dir: str, split: str = "train") -> Counter:
    """Read all parquet files in data_dir/split/ and return a code Counter."""
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    parquet_files = [
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {split_dir}")

    counter: Counter = Counter()
    for p in sorted(parquet_files):
        df = pd.read_parquet(p, columns=["code"])
        counter.update(df["code"].dropna().astype(str).tolist())
    return counter


def build_vocab(
    data_dir: str,
    embedding_type: str = "learned",
    top_k: int = 5000,
    split: str = "train",
) -> Vocab:
    """
    Build a Vocab from the training split.

    Parameters
    ----------
    data_dir:
        Root data directory containing split sub-folders.
    embedding_type:
        "learned"    → keep only top_k most frequent codes.
        "text_based" → keep ALL unique codes (top_k is ignored).
    top_k:
        Number of codes to retain when embedding_type == "learned".
    split:
        Which split to scan (default: "train").

    Returns
    -------
    Vocab with UNK as the final index.
    """
    counter = _collect_codes_from_parquets(data_dir, split)

    if embedding_type == "learned":
        selected_codes: List[str] = [code for code, _ in counter.most_common(top_k)]
    elif embedding_type == "text_based":
        # All unique codes, sorted for determinism
        selected_codes = sorted(counter.keys())
    else:
        raise ValueError(
            f"embedding_type must be 'learned' or 'text_based', got '{embedding_type}'"
        )

    # Build index mapping; UNK is appended as the last entry
    code_to_idx: Dict[str, int] = {code: idx for idx, code in enumerate(selected_codes)}
    unk_idx = len(code_to_idx)
    code_to_idx[UNK_TOKEN] = unk_idx

    return Vocab(code_to_idx=code_to_idx, unk_idx=unk_idx)


def build_vocab_from_codes(
    codes: List[str],
    embedding_type: str = "learned",
    top_k: int = 5000,
) -> Vocab:
    """
    Build a Vocab directly from a list of code strings (useful for testing).

    Parameters mirror build_vocab; the list is treated as if it were the full
    training corpus (duplicates contribute to frequency counts).
    """
    counter: Counter = Counter(codes)

    if embedding_type == "learned":
        selected_codes = [code for code, _ in counter.most_common(top_k)]
    elif embedding_type == "text_based":
        selected_codes = sorted(counter.keys())
    else:
        raise ValueError(
            f"embedding_type must be 'learned' or 'text_based', got '{embedding_type}'"
        )

    code_to_idx: Dict[str, int] = {code: idx for idx, code in enumerate(selected_codes)}
    unk_idx = len(code_to_idx)
    code_to_idx[UNK_TOKEN] = unk_idx

    return Vocab(code_to_idx=code_to_idx, unk_idx=unk_idx)
