"""
JEPA Span Masking.

For a sequence of N real (non-padding) tokens:

  B           = floor(N * mask_ratio)
  num_spans   = default_num_spans   if B >= default_num_spans * min_span_length
              = max(1, floor(B / min_span_length))  otherwise
  T_span      = floor(B / num_spans)       ← base span length
  last_span   = T_span + (B - T_span * num_spans)  ← receives remainder

Span selection:
  - allow_overlap=False (default): sample non-overlapping spans.
  - allow_overlap=True: sample spans independently; overlaps are allowed
    (closer to i-JEPA style target block sampling).
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
    min_gap_events:
        Minimum number of events between consecutive spans when
        allow_overlap=False.  Ignored when allow_overlap=True.
    allow_overlap:
        If True, target spans may overlap. If False, spans are forced to be
        disjoint whenever the budget allows.
    seed:
        Optional random seed for reproducibility (testing only).
    """

    def __init__(
        self,
        mask_ratio: float = 0.30,
        default_num_spans: int = 4,
        min_span_length: int = 15,
        min_gap_events: int = 0,
        allow_overlap: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.mask_ratio = mask_ratio
        self.default_num_spans = default_num_spans
        self.min_span_length = min_span_length
        self.min_gap_events = max(0, int(min_gap_events))
        self.allow_overlap = allow_overlap
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
        # Clamp budget to [0, N] so impossible configurations (e.g. mask_ratio>1)
        # never request more target tokens than available.
        B = max(0, min(int(N * self.mask_ratio), N))

        # Dynamic num_spans
        if B >= self.default_num_spans * self.min_span_length:
            num_spans = self.default_num_spans
        else:
            num_spans = max(1, B // self.min_span_length)

        # Enforce minimum inter-span gap for non-overlap mode:
        # total_span_tokens + gap*(num_spans-1) must fit into N.
        if not self.allow_overlap and self.min_gap_events > 0:
            while num_spans > 1 and (B + self.min_gap_events * (num_spans - 1) > N):
                num_spans -= 1

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

        # Sample spans (overlap behavior controlled by allow_overlap)
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
        Sample len(span_lengths) contiguous spans from
        real_positions (a contiguous range by index, not necessarily by
        original position values).

        If allow_overlap=True, spans are sampled independently and may overlap.
        If allow_overlap=False, uses a gap-sampling construction ("stars and
        bars") in index space so all spans are always placed when
        sum(span_lengths) <= N.
        """
        N = len(real_positions)
        if not span_lengths:
            return []

        if self.allow_overlap:
            selected: List[List[int]] = []
            for span_len in span_lengths:
                max_start_idx = N - span_len
                if max_start_idx < 0:
                    continue
                start_idx = self._rng.randint(0, max_start_idx)
                end_idx = start_idx + span_len
                selected.append([real_positions[j] for j in range(start_idx, end_idx)])
            return selected

        total_len = sum(span_lengths)
        k = len(span_lengths)
        mandatory_gap = self.min_gap_events * max(0, k - 1)
        if total_len + mandatory_gap > N:
            # Should be prevented by budgeting above; fail-safe fallback that
            # still returns a valid mask instead of returning [].
            max_len = max(1, N - mandatory_gap)
            if max_len <= 0:
                return []
            span_lengths = span_lengths[:1]
            span_lengths[0] = min(span_lengths[0], max_len)
            total_len = span_lengths[0]
            k = 1
            mandatory_gap = 0

        # Number of extra index positions not covered by spans and mandatory gaps.
        spare = N - total_len - mandatory_gap

        # Sample k+1 non-negative gaps that sum to `spare`.
        # We sample separators in a stars-and-bars representation to get a
        # random valid layout without overlap.
        cuts = sorted(self._rng.sample(range(spare + k), k))
        gaps: List[int] = []
        prev = -1
        for c in cuts + [spare + k]:
            gaps.append(c - prev - 1)
            prev = c

        selected: List[List[int]] = []
        idx_ptr = gaps[0]
        for i, span_len in enumerate(span_lengths):
            start_idx = idx_ptr
            end_idx = start_idx + span_len
            candidate_pos = [real_positions[j] for j in range(start_idx, end_idx)]
            selected.append(candidate_pos)
            # Internal gaps include configured minimum gap + sampled slack.
            internal_gap = gaps[i + 1] + (self.min_gap_events if i < k - 1 else 0)
            idx_ptr = end_idx + internal_gap

        return selected
