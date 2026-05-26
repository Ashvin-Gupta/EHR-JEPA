"""
JEPA Prediction Loss.

L_pred = distance(Z_pred, stop_gradient(Z_target))

The stop-gradient is applied here so the target encoder is NOT trained
by this loss.  It receives gradient only through L_cov (covariance_reg.py).

Both tensors are expected to have shape [B, num_spans, n_latents, d_model].
The loss is averaged over all dimensions.

All prediction losses accept a ``loss_type`` parameter:
  "mse"       — L2 squared error (default, preserves original behaviour)
  "smooth_l1" — Huber / Smooth L1 loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_VALID_LOSS_TYPES = ("mse", "smooth_l1")


def _per_token_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Per-token distance averaged over the embedding dimension.

    Parameters
    ----------
    pred, target:
        FloatTensor (..., d).  Same shape.
    loss_type:
        ``"mse"`` or ``"smooth_l1"``.

    Returns
    -------
    FloatTensor (...) — one scalar per token position.
    """
    if loss_type == "mse":
        return ((pred - target) ** 2).mean(dim=-1)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    raise ValueError(
        f"loss_type must be one of {_VALID_LOSS_TYPES}, got {loss_type!r}"
    )


def jepa_prediction_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Parameters
    ----------
    z_pred:
        FloatTensor (B, num_spans, n_latents, d_model) — predictor output.
    z_target:
        FloatTensor (B, num_spans, n_latents, d_model) — target encoder output.
        Stop-gradient is applied inside this function.
    loss_type:
        ``"mse"`` or ``"smooth_l1"``.

    Returns
    -------
    Scalar loss tensor (gradient flows only through z_pred).
    """
    tgt = z_target.detach()
    if loss_type == "mse":
        return F.mse_loss(z_pred, tgt)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(z_pred, tgt)
    raise ValueError(
        f"loss_type must be one of {_VALID_LOSS_TYPES}, got {loss_type!r}"
    )


def jepa_prediction_loss_weighted(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    weights: torch.Tensor,
    loss_type: str = "mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Token-level weighted prediction loss for Branch B (and similar rank-3 tensors).

    Parameters
    ----------
    z_pred, z_target:
        FloatTensor (B, N, d).  Stop-gradient applied to z_target.
    weights:
        FloatTensor (B, N) — non-negative; zero masks padded span positions.
        Typical: W_j = exp(-lambda * delta_minutes_j) for causal_single targets.
    loss_type:
        ``"mse"`` or ``"smooth_l1"``.

    Returns
    -------
    Scalar: sum(weights * per-token loss) / sum(weights).
    """
    tgt = z_target.detach()
    per_token = _per_token_distance(z_pred, tgt, loss_type)  # (B, N)
    w = weights.to(device=z_pred.device, dtype=z_pred.dtype)
    if w.shape != per_token.shape:
        raise ValueError(
            f"weights shape {w.shape} must match per-token loss {per_token.shape}"
        )
    num = (w * per_token).sum()
    den = w.sum().clamp(min=eps)
    return num / den


def jepa_prediction_loss_token_masked(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    token_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    loss_type: str = "mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Token-level prediction loss averaged only over positions where
    ``token_mask == 1``.

    Parameters
    ----------
    z_pred, z_target:
        (B, N, d).  Stop-gradient on target.
    token_mask:
        (B, N) — 1 for real target tokens, 0 for batch padding within N.
    weights:
        Optional (B, N) non-negative weights (e.g. time decay); padded positions
        should be zero.
    loss_type:
        ``"mse"`` or ``"smooth_l1"``.
    """
    tgt = z_target.detach()
    per_token = _per_token_distance(z_pred, tgt, loss_type)
    m = token_mask.to(device=z_pred.device, dtype=per_token.dtype)
    if weights is not None:
        w = weights.to(device=z_pred.device, dtype=per_token.dtype)
        m = m * w
    return (per_token * m).sum() / m.sum().clamp(min=eps)


def causal_ar_prediction_loss(
    y_hat: torch.Tensor,
    y_tgt: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_type: str = "mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Shifted-by-1 prediction loss for causal autoregressive JEPA.

    Prediction at position i is compared to target at position i+1.

    Parameters
    ----------
    y_hat:
        (B, N, d) — predictor output at each position.
    y_tgt:
        (B, N, d) — target encoder output.  Stop-gradient applied here.
    attention_mask:
        (B, N) — 1 for real tokens, 0 for padding.
    loss_type:
        ``"mse"`` or ``"smooth_l1"``.

    Returns
    -------
    Scalar loss averaged over valid shifted pairs.
    """
    pred = y_hat[:, :-1, :]
    target = y_tgt[:, 1:, :].detach()
    mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).float()
    per_token = _per_token_distance(pred, target, loss_type)
    return (per_token * mask).sum() / mask.sum().clamp(min=eps)


def future_time_decay_weights(
    delta_minutes: torch.Tensor,
    lam: float,
    floor: float = 0.0,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    W = exp(-lam * delta_minutes), optionally clamped below at ``floor``.

    Parameters
    ----------
    delta_minutes:
        (B, N) minutes since the causal cut (or equivalent).
    lam:
        Decay rate in 1/minutes; use 0 to return ones (caller should skip).
    floor:
        Minimum weight when > 0.
    mask:
        Optional (B, N) multiplier (e.g. token_mask); zeros stay zero.
    """
    w = torch.exp(-lam * delta_minutes.clamp(min=0).float())
    if floor > 0:
        w = w.clamp(min=floor)
    if mask is not None:
        w = w * mask.to(device=w.device, dtype=w.dtype)
    return w
