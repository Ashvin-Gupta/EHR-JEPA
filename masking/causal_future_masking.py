"""
Causal future-window masking for JEPA pretraining.

Samples S cutpoints t_s on the (windowed) timeline. For each s:
  - context_s: all real indices with position <= t_s (full prefix).
  - target_s: all real indices strictly after t_s through the end of the window.

Distant future tokens are down-weighted at training time via
``future_time_decay_lambda`` (see loss config), not by hard event/hour caps.

Outputs CausalSCutsMaskResult — consumed by MEDSCollator / JEPATrainer causal branch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class CausalSCutsMaskResult:
    """S independent (context, target) pairs aligned by index s."""

    contexts: List[List[int]]  # length S — each is sorted token indices (context)
    target_spans: List[List[int]]  # length S — each is contiguous target indices
    span_times: List[Tuple[float, float]]  # length S — (midpoint_h, duration_h)


def _real_positions(attention_mask: Optional[torch.Tensor], seq_len: int) -> List[int]:
    if attention_mask is None:
        return list(range(seq_len))
    if isinstance(attention_mask, torch.Tensor):
        mask_np = attention_mask.bool().tolist()
    else:
        mask_np = [bool(v) for v in attention_mask]
    return [i for i, m in enumerate(mask_np) if m]


def _span_midpoint_duration(span_idx: List[int], times_hours: List[float]) -> Tuple[float, float]:
    """Match SpanMasker: (midpoint, duration) in the same time units as times_hours."""
    if not span_idx:
        return (0.0, 0.0)
    span_t = [times_hours[p] for p in span_idx]
    mid = (span_t[0] + span_t[-1]) / 2.0
    dur = span_t[-1] - span_t[0]
    return (mid, dur)


def _build_target_after_t(
    real_positions: List[int],
    t_idx: int,
) -> List[int]:
    """All real indices strictly after t_idx through the end of the window."""
    return [p for p in real_positions if p > t_idx]


class CausalFutureMasker:
    """
    Parameters
    ----------
    num_cutpoints_S:
        Number of independent cutpoints (and context/target pairs) per call.
    min_target_events:
        Minimum number of real events in the future target window (t, end].
        Cutpoint t is resampled until this is met or attempts are exhausted.
    max_cutpoint_resamples:
        Maximum random t draws per slot before trying a shuffled pass over all
        valid cutpoints, then giving up on that slot.
    seed:
        Optional RNG seed (tests).
    """

    def __init__(
        self,
        num_cutpoints_S: int = 4,
        min_target_events: int = 1,
        max_cutpoint_resamples: int = 64,
        seed: Optional[int] = None,
    ) -> None:
        self.num_cutpoints_S = max(1, int(num_cutpoints_S))
        self.min_target_events = max(1, int(min_target_events))
        self.max_cutpoint_resamples = max(1, int(max_cutpoint_resamples))
        self._rng = random.Random(seed)

    def __call__(
        self,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        times_hours: Optional[List[float]] = None,
    ) -> CausalSCutsMaskResult:
        reals = _real_positions(attention_mask, seq_len)
        N = len(reals)
        if times_hours is None or len(times_hours) != seq_len:
            times_hours = [float(i) for i in range(seq_len)]

        if N < 2:
            return CausalSCutsMaskResult(
                contexts=[[]] * self.num_cutpoints_S,
                target_spans=[[]] * self.num_cutpoints_S,
                span_times=[(0.0, 0.0)] * self.num_cutpoints_S,
            )

        def sample_t() -> int:
            candidates = reals[:-1]
            if not candidates:
                return reals[0]
            return self._rng.choice(candidates)

        def valid_target_for_t(t_idx: int) -> bool:
            tgt = _build_target_after_t(reals, t_idx)
            return len(tgt) >= self.min_target_events

        def build_pair(t_idx: int) -> Tuple[List[int], List[int], Tuple[float, float]]:
            tgt = _build_target_after_t(reals, t_idx)
            ctx = [p for p in reals if p <= t_idx]
            mid_dur = _span_midpoint_duration(tgt, times_hours)
            return ctx, tgt, mid_dur

        contexts: List[List[int]] = []
        target_spans: List[List[int]] = []
        span_times: List[Tuple[float, float]] = []

        valid_ts = [t for t in reals[:-1] if valid_target_for_t(t)]
        used_t: set[int] = set()

        def pick_cutpoint() -> Optional[int]:
            if not valid_ts:
                return None
            t_chosen: Optional[int] = None
            for _ in range(self.max_cutpoint_resamples):
                t_cand = sample_t()
                if t_cand in used_t and len(used_t) < len(valid_ts):
                    continue
                if valid_target_for_t(t_cand):
                    t_chosen = t_cand
                    break
            if t_chosen is not None:
                return t_chosen
            order = valid_ts[:]
            self._rng.shuffle(order)
            for t_cand in order:
                if t_cand in used_t and len(used_t) < len(valid_ts):
                    continue
                return t_cand
            return self._rng.choice(valid_ts)

        for _ in range(self.num_cutpoints_S):
            t_chosen = pick_cutpoint()
            if t_chosen is None:
                contexts.append([])
                target_spans.append([])
                span_times.append((0.0, 0.0))
                continue

            ctx, tgt, st = build_pair(t_chosen)
            if len(tgt) < self.min_target_events:
                tgt = []
                st = (0.0, 0.0)
            else:
                used_t.add(t_chosen)
            contexts.append(ctx)
            target_spans.append(tgt)
            span_times.append(st)

        return CausalSCutsMaskResult(
            contexts=contexts,
            target_spans=target_spans,
            span_times=span_times,
        )
