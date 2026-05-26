"""
Additive self-attention masks for compact causal_single predictor sequences.

Without CLS: [context₀ … context_{nc-1} | MASK@target₀ …]

With CLS (include_cls=True): [CLS | context₀ … | MASK@target₀ …]

Structured modes (0 = attend, -inf = block) share the same off-diagonal layout:
  - context × target: blocked
  - target × context: all allowed
  - target × target: diagonal only

Top-left (CLS + context) × (CLS + context) block:
  - quadrant (ctx_causal=False): full bidirectional
  - partial_causal (ctx_causal=True): lower-triangular along compact order
    (CLS, then context in sequence order — each position sees itself and earlier)

With CLS additionally (quadrant only for non-causal top-left):
  - CLS → CLS, CLS → context: allowed; CLS → target: blocked
  - context → CLS: allowed
  - target → CLS: allowed
"""

from __future__ import annotations

from typing import List, Sequence

import torch


def build_causal_single_structured_mask(
    n_ctx: int,
    n_tgt: int,
    *,
    include_cls: bool = False,
    ctx_causal: bool = False,
    max_len: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build (L, L) additive mask for one row.

    L = (1 + n_ctx + n_tgt) if include_cls else (n_ctx + n_tgt), or max_len if set.
    Padding beyond the compact region stays blocked; combine with key_padding_mask.
    """
    if n_ctx < 0 or n_tgt < 0:
        raise ValueError(f"n_ctx and n_tgt must be non-negative, got {n_ctx}, {n_tgt}")
    n_cls = 1 if include_cls else 0
    compact = n_cls + n_ctx + n_tgt
    L = compact if max_len is None else max_len
    if L < compact:
        raise ValueError(f"max_len={L} < compact={compact}")

    mask = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
    if compact == 0:
        return mask

    off = n_cls
    ctx_end = off + n_ctx
    tgt_end = ctx_end + n_tgt

    if ctx_causal:
        top = ctx_end
        if top > 0:
            row_idx = torch.arange(top, device=device)
            causal = row_idx.unsqueeze(1) >= row_idx.unsqueeze(0)
            mask[:top, :top].masked_fill_(causal, 0.0)
    else:
        if include_cls:
            mask[0, 0] = 0.0
            if n_ctx > 0:
                mask[0, off:ctx_end] = 0.0
        if n_ctx > 0:
            if include_cls:
                mask[off:ctx_end, 0] = 0.0
            mask[off:ctx_end, off:ctx_end] = 0.0

    if n_tgt > 0 and n_ctx > 0:
        mask[ctx_end:tgt_end, off:ctx_end] = 0.0

    if n_tgt > 0:
        if include_cls:
            mask[ctx_end:tgt_end, 0] = 0.0
        idx = torch.arange(ctx_end, tgt_end, device=mask.device)
        mask[idx, idx] = 0.0

    return mask


def build_causal_single_quadrant_mask(
    n_ctx: int,
    n_tgt: int,
    *,
    include_cls: bool = False,
    max_len: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Quadrant mask: bidirectional top-left (CLS + context)."""
    return build_causal_single_structured_mask(
        n_ctx,
        n_tgt,
        include_cls=include_cls,
        ctx_causal=False,
        max_len=max_len,
        device=device,
        dtype=dtype,
    )


def build_causal_single_partial_causal_mask(
    n_ctx: int,
    n_tgt: int,
    *,
    include_cls: bool = False,
    max_len: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Partial-causal mask: causal (lower-triangular) top-left; other blocks as quadrant."""
    return build_causal_single_structured_mask(
        n_ctx,
        n_tgt,
        include_cls=include_cls,
        ctx_causal=True,
        max_len=max_len,
        device=device,
        dtype=dtype,
    )


def build_causal_single_structured_mask_batch(
    lengths_c: Sequence[int],
    lengths_t: Sequence[int],
    max_len: int,
    *,
    include_cls: bool = False,
    ctx_causal: bool = False,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Batched (B, max_len, max_len) mask — fully vectorized."""
    if len(lengths_c) != len(lengths_t):
        raise ValueError("lengths_c and lengths_t must have the same length")
    B = len(lengths_c)
    if B == 0:
        return torch.zeros(0, max_len, max_len, device=device, dtype=dtype)

    n_cls = 1 if include_cls else 0
    lc = torch.tensor(lengths_c, device=device, dtype=torch.long)
    lt = torch.tensor(lengths_t, device=device, dtype=torch.long)
    ctx_end = n_cls + lc
    tgt_end = ctx_end + lt

    out = torch.full((B, max_len, max_len), float("-inf"), device=device, dtype=dtype)
    idx = torch.arange(max_len, device=device)

    max_ce = int(ctx_end.max().item()) if B > 0 else 0
    if ctx_causal and max_ce > 0:
        r = torch.arange(max_ce, device=device)
        causal_template = r.unsqueeze(1) >= r.unsqueeze(0)

    for b in range(B):
        ce = int(ctx_end[b])
        te = int(tgt_end[b])
        nc = int(lc[b])
        off = n_cls

        if ctx_causal:
            if ce > 0:
                out[b, :ce, :ce].masked_fill_(causal_template[:ce, :ce], 0.0)
        else:
            if include_cls:
                out[b, 0, 0] = 0.0
                if nc > 0:
                    out[b, 0, off:ce] = 0.0
            if nc > 0:
                if include_cls:
                    out[b, off:ce, 0] = 0.0
                out[b, off:ce, off:ce] = 0.0

        if nc > 0 and te > ce:
            out[b, ce:te, off:ce] = 0.0
        if te > ce:
            if include_cls:
                out[b, ce:te, 0] = 0.0
            tgt_idx = idx[ce:te]
            out[b, tgt_idx, tgt_idx] = 0.0

    return out


def build_causal_single_quadrant_mask_batch(
    lengths_c: Sequence[int],
    lengths_t: Sequence[int],
    max_len: int,
    *,
    include_cls: bool = False,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return build_causal_single_structured_mask_batch(
        lengths_c,
        lengths_t,
        max_len,
        include_cls=include_cls,
        ctx_causal=False,
        device=device,
        dtype=dtype,
    )


def build_causal_single_partial_causal_mask_batch(
    lengths_c: Sequence[int],
    lengths_t: Sequence[int],
    max_len: int,
    *,
    include_cls: bool = False,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return build_causal_single_structured_mask_batch(
        lengths_c,
        lengths_t,
        max_len,
        include_cls=include_cls,
        ctx_causal=True,
        device=device,
        dtype=dtype,
    )


def structured_mask_allows(
    mask: torch.Tensor, query: int, key: int, *, atol: float = 1e-6
) -> bool:
    """True if additive mask permits attention from query to key."""
    return bool(mask[query, key].item() > -1e4)


# Backward-compatible alias
quadrant_mask_allows = structured_mask_allows


__all__ = [
    "build_causal_single_structured_mask",
    "build_causal_single_structured_mask_batch",
    "build_causal_single_quadrant_mask",
    "build_causal_single_quadrant_mask_batch",
    "build_causal_single_partial_causal_mask",
    "build_causal_single_partial_causal_mask_batch",
    "structured_mask_allows",
    "quadrant_mask_allows",
]
