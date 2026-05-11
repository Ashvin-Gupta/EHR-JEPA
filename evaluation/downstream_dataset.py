"""
Shared downstream dataset wrapper for prediction-time filtering.

This module intentionally contains no imports from `main.py` or
`evaluation/run_linear_probe.py` so it can be safely imported by both
without introducing circular dependencies.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
from data.meds_dataset import MEDSDataset


class DownstreamDataset(torch.utils.data.Dataset):
    """
    Wraps MEDSDataset, adds label lookup and prediction_time filtering.

    Only subjects present in `labels` are included.
    Events after prediction_time are removed to prevent data leakage.
    """

    def __init__(
        self,
        base_dataset: MEDSDataset,
        labels: Dict[int, tuple],
    ) -> None:
        self.base = base_dataset
        self.labels = labels
        self._indices = [
            i for i in range(len(base_dataset))
            if base_dataset.samples[i]["subject_id"] in labels
        ]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[self._indices[idx]]
        subject_id = item["subject_id"]
        pred_time, label = self.labels[subject_id]

        if pred_time is not None:
            cutoff = pd.Timestamp(pred_time)
            keep = [i for i, t in enumerate(item["times"]) if t <= cutoff]
            item["codes"]       = [item["codes"][i]       for i in keep]
            item["values"]      = [item["values"][i]      for i in keep]
            item["times"]       = [item["times"][i]       for i in keep]
            # z_scores and delta_times must be filtered to the same positions
            # so the collator receives matching-length lists for every feature.
            if "z_scores" in item:
                item["z_scores"]    = [item["z_scores"][i]    for i in keep]
            if "delta_times" in item:
                item["delta_times"] = [item["delta_times"][i] for i in keep]

        item["label"] = label
        return item
