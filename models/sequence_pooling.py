"""Sequence-level pooling for downstream evaluation (CLS vs masked mean)."""

from __future__ import annotations

from typing import Literal, Union

import torch

SequencePoolingMode = Literal["cls", "mean_pool"]


def parse_pooling_mode(value: Union[str, None]) -> SequencePoolingMode:
    """Normalize config/CLI pooling string."""
    if value is None:
        return "cls"
    mode = str(value).strip().lower().replace("-", "_")
    if mode in ("cls", "mean_pool", "meanpool", "mean"):
        return "mean_pool" if mode != "cls" else "cls"
    raise ValueError(
        f"pooling must be 'cls' or 'mean_pool', got {value!r}"
    )


def get_config_pooling(cfg: dict, section: str = "downstream_eval") -> SequencePoolingMode:
    """Read pooling from a config block (downstream_eval or downstream)."""
    block = cfg.get(section) or {}
    return parse_pooling_mode(block.get("pooling", "cls"))


def get_eval_pooling(cfg: dict) -> SequencePoolingMode:
    """Prefer downstream_eval.pooling, else downstream.pooling."""
    de = cfg.get("downstream_eval") or {}
    if de.get("pooling") is not None:
        return parse_pooling_mode(de["pooling"])
    return get_config_pooling(cfg, "downstream")


def mean_pool_sequence(
    h: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked mean over sequence positions.

    Parameters
    ----------
    h: (B, L, d)
    attention_mask: (B, L), 1=real, 0=pad
    """
    real = attention_mask.unsqueeze(-1).float()
    return (h * real).sum(1) / real.sum(1).clamp(min=1)
