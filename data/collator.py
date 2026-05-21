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

  When a span masker is supplied (pretrain span_budget), three extra fields:
    "mask_context_indices", "mask_target_spans", "mask_span_times"

  When a CausalFutureMasker is supplied (pretrain causal_future):
    "mask_causal_contexts", "mask_causal_targets", "mask_causal_span_times"

  When a CausalSingleCutMasker is supplied (pretrain causal_single; also
  mask_cutpoint_indices, mask_context_start_indices):
    same span keys as SpanMasker plus optional "mask_target_delta_minutes"
    (minutes from cut t to each target token, for exponential time-decay loss).
  }
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from masking.causal_future_masking import CausalFutureMasker
    from masking.causal_single_cut_masking import CausalSingleCutMasker
    from masking.span_masking import SpanMasker


def _hours_since_first_window(times_seq: List[Any]) -> List[float]:
    """Hours since first timestamp in the (already windowed/padded) list."""
    if not times_seq:
        return []
    t0 = times_seq[0]
    out: List[float] = []
    for t in times_seq:
        if t is None:
            out.append(0.0)
            continue
        try:
            delta = t - t0
            if hasattr(delta, "total_seconds"):
                sec = float(delta.total_seconds())
            else:
                sec = float(delta)
            out.append(sec / 3600.0)
        except Exception:
            out.append(0.0)
    return out


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
        Optional SpanMasker or CausalFutureMasker.  When provided, masking is
        applied here inside the DataLoader worker process.
    seed:
        Optional random seed for reproducible windowing (testing only).
    """

    def __init__(
        self,
        pad_idx: int,
        max_len: int,
        task: str = "pretrain",
        masker: "Optional[Union[SpanMasker, CausalFutureMasker, CausalSingleCutMasker]]" = None,
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
        times: Optional[List[Any]] = None,
    ):
        """
        Apply windowing (pretrain long) or padding to a single sequence.

        Returns (codes, values, z_scores, delta_times, attention_mask, times_out)
        where times_out is aligned with the returned codes (same length as codes),
        or None if times was None or length-mismatched.
        """
        seq_len = len(codes)
        times_ok = times is not None and len(times) == seq_len

        if self.task == "pretrain" and seq_len > self.max_len:
            start = self._rng.randint(0, seq_len - self.max_len)
            codes = codes[start : start + self.max_len]
            values = values[start : start + self.max_len]
            z_scores = z_scores[start : start + self.max_len]
            delta_times = delta_times[start : start + self.max_len]
            attention_mask = [1] * self.max_len
            times_out = (
                times[start : start + self.max_len] if times_ok else None
            )

        elif seq_len >= self.max_len:
            codes = codes[: self.max_len]
            values = values[: self.max_len]
            z_scores = z_scores[: self.max_len]
            delta_times = delta_times[: self.max_len]
            attention_mask = [1] * self.max_len
            times_out = times[: self.max_len] if times_ok else None

        else:
            pad_len = self.max_len - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            codes = codes + [self.pad_idx] * pad_len
            values = values + [None] * pad_len
            z_scores = z_scores + [0.0] * pad_len
            delta_times = delta_times + [0.0] * pad_len
            if times_ok:
                pad_ts = times[seq_len - 1] if seq_len > 0 else None
                times_out = list(times) + [pad_ts] * pad_len
            else:
                times_out = None

        return codes, values, z_scores, delta_times, attention_mask, times_out

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from masking.causal_future_masking import CausalFutureMasker
        from masking.causal_single_cut_masking import CausalSingleCutMasker

        all_codes: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        all_values: List[torch.Tensor] = []
        all_value_masks: List[torch.Tensor] = []
        all_z_scores: List[torch.Tensor] = []
        all_delta_times: List[torch.Tensor] = []
        labels: List[int] = []
        subject_ids: List[int] = []
        orig_seq_lengths: List[int] = []

        all_ctx_indices: List[List[int]] = []
        all_tgt_spans: List[List[List[int]]] = []
        all_span_times: List[List[tuple]] = []
        all_target_delta_minutes: List[List[List[float]]] = []
        all_cutpoint_indices: List[int] = []
        all_context_start_indices: List[int] = []

        all_causal_ctx: List[List[List[int]]] = []
        all_causal_tgt: List[List[List[int]]] = []
        all_causal_st: List[List[tuple]] = []

        for item in batch:
            z_scores_in = item.get("z_scores", [0.0] * len(item["codes"]))
            delta_times_in = item.get("delta_times", [0.0] * len(item["codes"]))
            times_in = item.get("times")

            orig_seq_lengths.append(len(item["codes"]))

            codes, values, z_scores, delta_times, attention_mask, times_win = (
                self._window_or_pad(
                    item["codes"],
                    item["values"],
                    z_scores_in,
                    delta_times_in,
                    times=times_in,
                )
            )

            if self.masker is not None:
                if isinstance(self.masker, CausalFutureMasker):
                    th = (
                        _hours_since_first_window(times_win)
                        if times_win is not None
                        else None
                    )
                    cr = self.masker(
                        seq_len=self.max_len,
                        attention_mask=attention_mask,
                        times_hours=th,
                    )
                    all_causal_ctx.append(cr.contexts)
                    all_causal_tgt.append(cr.target_spans)
                    all_causal_st.append(cr.span_times)
                elif isinstance(self.masker, CausalSingleCutMasker):
                    th = (
                        _hours_since_first_window(times_win)
                        if times_win is not None
                        else None
                    )
                    mask_result = self.masker(
                        seq_len=self.max_len,
                        attention_mask=attention_mask,
                        times_hours=th,
                    )
                    all_ctx_indices.append(mask_result.context_indices)
                    all_tgt_spans.append(mask_result.target_spans)
                    all_span_times.append(mask_result.span_times)
                    dm = mask_result.target_token_delta_minutes
                    all_target_delta_minutes.append(dm if dm is not None else [])
                    cp = mask_result.cutpoint_index
                    all_cutpoint_indices.append(int(cp) if cp is not None else -1)
                    cs = mask_result.context_start_index
                    all_context_start_indices.append(int(cs) if cs is not None else -1)
                else:
                    mask_result = self.masker(
                        seq_len=self.max_len,
                        attention_mask=attention_mask,
                    )
                    all_ctx_indices.append(mask_result.context_indices)
                    all_tgt_spans.append(mask_result.target_spans)
                    all_span_times.append(mask_result.span_times)
                    dm = mask_result.target_token_delta_minutes
                    all_target_delta_minutes.append(dm if dm is not None else [])

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

        out["orig_seq_lengths"] = torch.tensor(orig_seq_lengths, dtype=torch.long)

        if self.masker is not None:
            if isinstance(self.masker, CausalFutureMasker):
                out["mask_causal_contexts"] = all_causal_ctx
                out["mask_causal_targets"] = all_causal_tgt
                out["mask_causal_span_times"] = all_causal_st
            else:
                out["mask_context_indices"] = all_ctx_indices
                out["mask_target_spans"] = all_tgt_spans
                out["mask_span_times"] = all_span_times
                if any(all_target_delta_minutes):
                    out["mask_target_delta_minutes"] = all_target_delta_minutes
                if any(i >= 0 for i in all_cutpoint_indices):
                    out["mask_cutpoint_indices"] = all_cutpoint_indices
                if any(i >= 0 for i in all_context_start_indices):
                    out["mask_context_start_indices"] = all_context_start_indices

        return out
