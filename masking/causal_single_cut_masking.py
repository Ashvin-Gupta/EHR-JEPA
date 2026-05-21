"""
Causal single-cut masking for JEPA pretraining.

Two random indices on the windowed sequence (no context-window hour/event cap):

  1. context_start = s  — random start of the active region (in bounds).
  2. cutpoint t in (s, e] — random split inside [s, e], e = last real token.

  - context: real indices s <= p <= t
  - target:  real indices t < p <= e

Tokens before s are excluded from JEPA for that sample.

Returns SpanMaskResult (one target span) for the standard span JEPA forward path.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

from masking.causal_future_masking import _real_positions, _span_midpoint_duration
from masking.span_masking import SpanMaskResult


class CausalSingleCutMasker:
    """
    Parameters
    ----------
    future_max_events / future_max_hours:
        Kept for config/API compatibility with causal_future; **not used** here.
        Target length is (t, last_real] only.
    min_context_events:
        At least this many tokens in [s, t].
    min_target_events:
        At least this many tokens in (t, e].
    max_cutpoint_resamples:
        Random (s, t) draws before falling back to a shuffled valid pair.
    seed:
        Optional RNG seed (tests).
    """

    def __init__(
        self,
        future_max_events: int = 128,
        future_max_hours: float = 6.0,
        min_context_events: int = 15,
        min_target_events: int = 15,
        max_cutpoint_resamples: int = 12,
        seed: Optional[int] = None,
    ) -> None:
        _ = future_max_events, future_max_hours
        self.min_context_events = max(1, int(min_context_events))
        self.min_target_events = max(1, int(min_target_events))
        self.max_cutpoint_resamples = max(1, int(max_cutpoint_resamples))
        self._rng = random.Random(seed)

    def _valid_pairs(self, reals: List[int]) -> List[Tuple[int, int]]:
        """
        All (context_start, cutpoint) pairs with context [s..t] and target (t..e].
        """
        if len(reals) < 2:
            return []
        e = reals[-1]
        pairs: List[Tuple[int, int]] = []
        for si, s in enumerate(reals):
            if s == e:
                continue
            segment = reals[si:]
            for j in range(len(segment) - 1):
                t = segment[j]
                n_ctx = j + 1
                n_tgt = len(segment) - j - 1
                if n_ctx >= self.min_context_events and n_tgt >= self.min_target_events:
                    pairs.append((s, t))
        return pairs

    def __call__(
        self,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        times_hours: Optional[List[float]] = None,
        times: Optional[List[float]] = None,
    ) -> SpanMaskResult:
        """Compatible with SpanMasker / collator (times_hours or legacy times=)."""
        if times_hours is None and times is not None:
            times_hours = times

        reals = _real_positions(attention_mask, seq_len)
        if times_hours is None or len(times_hours) != seq_len:
            times_hours = [float(i) for i in range(seq_len)]

        empty = SpanMaskResult(
            context_indices=[],
            target_spans=[[]],
            span_times=[(0.0, 0.0)],
            target_token_delta_minutes=[[]],
        )

        if len(reals) < 2:
            return empty

        pairs = self._valid_pairs(reals)
        if not pairs:
            return empty

        s_chosen, t_chosen = pairs[0]
        for _ in range(self.max_cutpoint_resamples):
            s_chosen, t_chosen = self._rng.choice(pairs)
            break

        e = reals[-1]
        context_indices = [p for p in reals if s_chosen <= p <= t_chosen]
        tgt = [p for p in reals if t_chosen < p <= e]
        if len(context_indices) < self.min_context_events or len(tgt) < self.min_target_events:
            return empty

        st = _span_midpoint_duration(tgt, times_hours)
        t_cut_h = times_hours[t_chosen]
        delta_min = [(times_hours[p] - t_cut_h) * 60.0 for p in tgt]

        return SpanMaskResult(
            context_indices=context_indices,
            target_spans=[tgt],
            span_times=[st],
            target_token_delta_minutes=[delta_min],
            cutpoint_index=t_chosen,
            context_start_index=s_chosen,
        )
