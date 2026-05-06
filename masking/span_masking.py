"""
JEPA Span Masking.

For a sequence of N real (non-padding) tokens:

  B           = floor(N * mask_ratio)
  num_spans   = default_num_spans   if B >= default_num_spans * min_span_length
              = max(1, floor(B / min_span_length))  otherwise
  T_span      = floor(B / num_spans)       ← base span length
  last_span   = T_span + (B - T_span * num_spans)  ← receives remainder

Span selection:
  Sample non-overlapping, non-padding spans by rejection sampling.
  Each span is a contiguous range of positions from the set of real token
  positions.

Outputs (SpanMaskResult):
  context_indices: List[int]             — positions NOT in any span
  target_spans:    List[List[int]]       — per-span position lists
  span_times:      List[Tuple[float,float]] — (midpoint_hours, duration_hours)
                                            computed from a times array if provided;
                                            (float(mid_pos), float(span_len)) otherwise
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class SpanMaskResult:
    """Output of SpanMasker.__call__."""
    context_indices: List[int]
    target_spans: List[List[int]]
    span_times: List[Tuple[float, float]]  # (midpoint_hours, duration_hours)


class SpanMasker:
    """
    Parameters
    ----------
    mask_ratio:
        Fraction of real tokens to mask.  Default 0.30.
    default_num_spans:
        Target number of spans when the sequence is long enough.
    min_span_length:
        Minimum events per span.  Used to dynamically reduce num_spans
        for short sequences.
    seed:
        Optional random seed for reproducibility (testing only).
    """

    def __init__(
        self,
        mask_ratio: float = 0.30,
        default_num_spans: int = 4,
        min_span_length: int = 15,
        seed: Optional[int] = None,
    ) -> None:
        self.mask_ratio = mask_ratio
        self.default_num_spans = default_num_spans
        self.min_span_length = min_span_length
        self._rng = random.Random(seed)

    def __call__(
        self,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        times: Optional[List[float]] = None,
    ) -> SpanMaskResult:
        """
        Parameters
        ----------
        seq_len:
            Total sequence length (including padding).
        attention_mask:
            BoolTensor or LongTensor of shape (seq_len,) — 1/True = real token.
            If None, all positions are treated as real.
        times:
            Optional list of timestamps in hours (length seq_len).
            Used to compute (midpoint_hours, duration_hours) for span_times.
            If None, positions (integers) are used as proxy times.

        Returns
        -------
        SpanMaskResult
        """
        # Real (non-padding) positions
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                mask_np = attention_mask.bool().tolist()
            else:
                mask_np = [bool(v) for v in attention_mask]
            real_positions = [i for i, m in enumerate(mask_np) if m]
        else:
            real_positions = list(range(seq_len))

        N = len(real_positions)
        B = int(N * self.mask_ratio)

        # Dynamic num_spans
        if B >= self.default_num_spans * self.min_span_length:
            num_spans = self.default_num_spans
        else:
            num_spans = max(1, B // self.min_span_length)

        if B == 0 or num_spans == 0:
            return SpanMaskResult(
                context_indices=real_positions,
                target_spans=[],
                span_times=[],
            )

        T_span_base = B // num_spans
        remainder = B - T_span_base * num_spans

        # Span lengths: all T_span_base except last gets remainder
        span_lengths = [T_span_base] * num_spans
        span_lengths[-1] += remainder

        # Remove zero-length spans (can happen for very short sequences)
        span_lengths = [s for s in span_lengths if s > 0]
        num_spans = len(span_lengths)

        # Sample non-overlapping spans (positions in real_positions space)
        selected_spans: List[List[int]] = self._sample_spans(
            real_positions, span_lengths
        )

        # Flatten to a set of masked positions
        masked_set: set = set()
        for span in selected_spans:
            masked_set.update(span)

        context_indices = [p for p in real_positions if p not in masked_set]

        # Compute span_times
        span_times: List[Tuple[float, float]] = []
        for span in selected_spans:
            if times is not None:
                span_t = [times[p] for p in span]
                mid = (span_t[0] + span_t[-1]) / 2.0
                dur = span_t[-1] - span_t[0]
            else:
                mid = (span[0] + span[-1]) / 2.0
                dur = float(len(span))
            span_times.append((mid, dur))

        return SpanMaskResult(
            context_indices=context_indices,
            target_spans=selected_spans,
            span_times=span_times,
        )

    # ------------------------------------------------------------------
    # Span sampling
    # ------------------------------------------------------------------

    def _sample_spans(
        self,
        real_positions: List[int],
        span_lengths: List[int],
    ) -> List[List[int]]:
        """
        Sample len(span_lengths) non-overlapping contiguous spans from
        real_positions (a contiguous range by index, not necessarily by
        original position values).

        Uses rejection sampling with a maximum attempt budget.
        """
        N = len(real_positions)
        selected: List[List[int]] = []
        occupied: set = set()

        for span_len in span_lengths:
            max_start_idx = N - span_len
            if max_start_idx < 0:
                # Sequence too short; skip this span
                continue

            chosen = None
            for _ in range(1000):
                start_idx = self._rng.randint(0, max_start_idx)
                candidate_idx = list(range(start_idx, start_idx + span_len))
                candidate_pos = [real_positions[i] for i in candidate_idx]

                if not occupied.intersection(candidate_pos):
                    chosen = candidate_pos
                    break

            if chosen is not None:
                selected.append(chosen)
                occupied.update(chosen)

        return selected
