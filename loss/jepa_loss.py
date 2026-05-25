"""
JEPA Prediction Loss.

L_pred = || Z_pred - stop_gradient(Z_target) ||²

The stop-gradient is applied here so the target encoder is NOT trained
by this loss.  It receives gradient only through L_cov (covariance_reg.py).

Both tensors are expected to have shape [B, num_spans, n_latents, d_model].
The loss is averaged over all dimensions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_prediction_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    z_pred:
        FloatTensor (B, num_spans, n_latents, d_model) — predictor output.
    z_target:
        FloatTensor (B, num_spans, n_latents, d_model) — target encoder output.
        Stop-gradient is applied inside this function.

    Returns
    -------
    Scalar loss tensor (gradient flows only through z_pred).
    """
    return F.mse_loss(z_pred, z_target.detach())


def jepa_prediction_loss_weighted(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Token-level weighted MSE for Branch B (and similar rank-3 tensors).

    Parameters
    ----------
    z_pred, z_target:
        FloatTensor (B, N, d).  Stop-gradient applied to z_target.
    weights:
        FloatTensor (B, N) — non-negative; zero masks padded span positions.
        Typical: W_j = exp(-lambda * delta_minutes_j) for causal_single targets.

    Returns
    -------
    Scalar: sum(weights * per-token MSE) / sum(weights).
    """
    tgt = z_target.detach()
    per_token = ((z_pred - tgt) ** 2).mean(dim=-1)  # (B, N)
    w = weights.to(device=z_pred.device, dtype=z_pred.dtype)
    if w.shape != per_token.shape:
        raise ValueError(
            f"weights shape {w.shape} must match per-token MSE {per_token.shape}"
        )
    num = (w * per_token).sum()
    den = w.sum().clamp(min=eps)
    return num / den


def jepa_prediction_loss_token_masked(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    token_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Token-level MSE averaged only over positions where ``token_mask == 1``.

    Parameters
    ----------
    z_pred, z_target:
        (B, N, d).  Stop-gradient on target.
    token_mask:
        (B, N) — 1 for real target tokens, 0 for batch padding within N.
    weights:
        Optional (B, N) non-negative weights (e.g. time decay); padded positions
        should be zero.
    """
    tgt = z_target.detach()
    per_token = ((z_pred - tgt) ** 2).mean(dim=-1)
    m = token_mask.to(device=z_pred.device, dtype=per_token.dtype)
    if weights is not None:
        w = weights.to(device=z_pred.device, dtype=per_token.dtype)
        m = m * w
    return (per_token * m).sum() / m.sum().clamp(min=eps)


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
