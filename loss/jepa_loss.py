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
