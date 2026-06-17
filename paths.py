"""
Portable path resolution for the EHR-JEPA workspace layout.

Expected sibling directories under the workspace root (parent of this repo):
  EHR-JEPA/          — this project
  EHR-JEPA-Data/     — vocab, labels, cache, checkpoints, etc.
  MIMIC_data/        — raw MIMIC lookup tables
  clean_meds/        — processed parquet splits (train/, tuning/, held_out/)

Override any root with environment variables:
  EHR_JEPA_ROOT, EHR_JEPA_DATA, MIMIC_DATA, CLEAN_MEDS
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(
    os.environ.get("EHR_JEPA_ROOT", Path(__file__).resolve().parent)
).resolve()
WORKSPACE_ROOT = PROJECT_ROOT.parent

EHR_JEPA_DATA = Path(
    os.environ.get("EHR_JEPA_DATA", WORKSPACE_ROOT / "EHR-JEPA-Data")
).resolve()
MIMIC_DATA = Path(
    os.environ.get("MIMIC_DATA", WORKSPACE_ROOT / "MIMIC_data")
).resolve()
CLEAN_MEDS = Path(
    os.environ.get("CLEAN_MEDS", WORKSPACE_ROOT / "clean_meds")
).resolve()

_CONFIG_PATH_KEYS: dict[str, list[str]] = {
    "data": [
        "data_dir",
        "lookup_dir",
        "vocab_path",
        "labels_base_dir",
        "cache_dir",
        "aces_label_path",
    ],
    "normalizer": ["stats_path"],
    "model": ["code_embeddings_path"],
    "training": ["checkpoint_dir", "resume_from"],
    "downstream_eval": ["checkpoint_path"],
}


def resolve_path(path: str | None, *, base: Path | None = None) -> str | None:
    """Expand a config path: absolute paths unchanged; relative paths from project root."""
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return text
    candidate = Path(text)
    if candidate.is_absolute():
        return str(candidate)
    root = base or PROJECT_ROOT
    return str((root / candidate).resolve())


def resolve_config_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *cfg* with known filesystem paths resolved."""
    out = dict(cfg)
    for section, keys in _CONFIG_PATH_KEYS.items():
        block = out.get(section)
        if not isinstance(block, dict):
            continue
        block = dict(block)
        for key in keys:
            if key in block and block[key] is not None:
                block[key] = resolve_path(block[key])
        out[section] = block
    return out
