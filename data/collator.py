"""
Batch collator for MEDS event sequences.

Handles two sequence-length cases for pretrain and prediction modes:

  Pretrain — longer than max_len:
    Sample a random start index i ~ Uniform(0, seq_len - max_len) and take
    events[i : i + max_len].  This stochastic sliding window means each
    training step sees a different slice of a long sequence.

  Pretrain — shorter than max_len:
    Right-pad with pad_idx to max_len; attention_mask = 0 on pad positions.

  Prediction:
    Header-preserving truncation has already been applied at the dataset
    level.  The collator only pads short sequences here.

Output batch dict (all LongTensor or FloatTensor, shape [B, L]):
  {
    "codes":          LongTensor  [B, max_len]   — vocab indices
    "attention_mask": LongTensor  [B, max_len]   — 1 real, 0 pad
    "values":         FloatTensor [B, max_len]   — numeric_value (0.0 for None)
    "value_mask":     LongTensor  [B, max_len]   — 1 if value present, 0 if not
    "z_scores":       FloatTensor [B, max_len]   — z-scored values (0.0 if missing)
    "delta_times":    FloatTensor [B, max_len]   — log(1 + hours_since_prev)
    "labels":         LongTensor  [B]
    "subject_ids":    LongTensor  [B]

  When a masker is supplied (pretrain), three extra fields are added that
  carry the pre-computed span-masking results.  The trainer reads these
  directly instead of re-running the masker on the main thread, so all
  masking CPU work executes inside the DataLoader worker processes while
  the GPU is busy with the previous batch:

    "mask_context_indices": List[List[int]]            — one per sample
    "mask_target_spans":    List[List[List[int]]]       — one per sample
    "mask_span_times":      List[List[Tuple[float,float]]] — one per sample
  }
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from masking.span_masking import SpanMasker


class MEDSCollator:
    """
    Parameters
    ----------
    pad_idx:
        Vocabulary index used for padding code sequences.
    max_len:
        Context window length.  Sequences longer than this are windowed
        (pretrain) or assumed pre-truncated (prediction).
    task:
        "pretrain" — stochastic windowing for long sequences.
        "prediction" — no windowing; only pad.
    masker:
        Optional SpanMasker.  When provided, span masking is applied here
        inside the DataLoader worker process (overlaps with GPU computation).
        When None, the trainer performs masking on the main thread.
    seed:
        Optional random seed for reproducible windowing (testing only).
    """

    def __init__(
        self,
        pad_idx: int,
        max_len: int,
        task: str = "pretrain",
        masker: "Optional[SpanMasker]" = None,
        seed: Optional[int] = None,
    ):
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.task = task
        self.masker = masker
        self._rng = random.Random(seed)

    def _window_or_pad(
        self,
        codes: List[int],
        values: List[Optional[float]],
        z_scores: List[float],
        delta_times: List[float],
    ):
        """
        Apply windowing (pretrain long) or padding to a single sequence.

        Returns (codes, values, z_scores, delta_times, attention_mask) all as
        plain Python lists of length max_len.
        """
        seq_len = len(codes)

        if self.task == "pretrain" and seq_len > self.max_len:
            start = self._rng.randint(0, seq_len - self.max_len)
            codes = codes[start : start + self.max_len]
            values = values[start : start + self.max_len]
            z_scores = z_scores[start : start + self.max_len]
            delta_times = delta_times[start : start + self.max_len]
            attention_mask = [1] * self.max_len

        elif seq_len >= self.max_len:
            codes = codes[: self.max_len]
            values = values[: self.max_len]
            z_scores = z_scores[: self.max_len]
            delta_times = delta_times[: self.max_len]
            attention_mask = [1] * self.max_len

        else:
            pad_len = self.max_len - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            codes = codes + [self.pad_idx] * pad_len
            values = values + [None] * pad_len
            z_scores = z_scores + [0.0] * pad_len
            delta_times = delta_times + [0.0] * pad_len

        return codes, values, z_scores, delta_times, attention_mask

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_codes: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        all_values: List[torch.Tensor] = []
        all_value_masks: List[torch.Tensor] = []
        all_z_scores: List[torch.Tensor] = []
        all_delta_times: List[torch.Tensor] = []
        labels: List[int] = []
        subject_ids: List[int] = []
        orig_seq_lengths: List[int] = []  # length before any windowing/padding

        # Pre-computed masking (populated when self.masker is not None)
        all_ctx_indices: List[List[int]] = []
        all_tgt_spans: List[List[List[int]]] = []
        all_span_times: List[List[tuple]] = []

        for item in batch:
            z_scores_in = item.get("z_scores", [0.0] * len(item["codes"]))
            delta_times_in = item.get("delta_times", [0.0] * len(item["codes"]))

            orig_seq_lengths.append(len(item["codes"]))

            codes, values, z_scores, delta_times, attention_mask = self._window_or_pad(
                item["codes"], item["values"], z_scores_in, delta_times_in
            )

            # Run span masking here (inside the worker process) so the main
            # thread never has to wait for it between GPU steps.
            if self.masker is not None:
                mask_result = self.masker(
                    seq_len=self.max_len,
                    attention_mask=attention_mask,
                )
                all_ctx_indices.append(mask_result.context_indices)
                all_tgt_spans.append(mask_result.target_spans)
                all_span_times.append(mask_result.span_times)

            float_values = [v if v is not None else 0.0 for v in values]
            value_present = [1 if v is not None else 0 for v in values]

            all_codes.append(torch.tensor(codes, dtype=torch.long))
            all_masks.append(torch.tensor(attention_mask, dtype=torch.long))
            all_values.append(torch.tensor(float_values, dtype=torch.float))
            all_value_masks.append(torch.tensor(value_present, dtype=torch.long))
            all_z_scores.append(torch.tensor(z_scores, dtype=torch.float))
            all_delta_times.append(torch.tensor(delta_times, dtype=torch.float))
            labels.append(item.get("label", 0))
            subject_ids.append(item.get("subject_id", -1))

        out: Dict[str, Any] = {
            "codes": torch.stack(all_codes),
            "attention_mask": torch.stack(all_masks),
            "values": torch.stack(all_values),
            "value_mask": torch.stack(all_value_masks),
            "z_scores": torch.stack(all_z_scores),
            "delta_times": torch.stack(all_delta_times),
            "labels": torch.tensor(labels, dtype=torch.long),
            "subject_ids": torch.tensor(subject_ids, dtype=torch.long),
        }

        # Original trajectory length before any windowing — this is the true
        # patient sequence length and is independent of max_seq_len or masking.
        out["orig_seq_lengths"] = torch.tensor(orig_seq_lengths, dtype=torch.long)

        if self.masker is not None:
            # These are plain Python lists (variable-length per sample).
            # DataLoader leaves non-Tensor values as-is, so pin_memory is a no-op
            # for these fields — that's fine, they never touch the GPU.
            out["mask_context_indices"] = all_ctx_indices
            out["mask_target_spans"]    = all_tgt_spans
            out["mask_span_times"]      = all_span_times

        return out
