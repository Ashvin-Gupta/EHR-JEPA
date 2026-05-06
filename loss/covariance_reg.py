"""
Covariance Regularization Loss (VICReg-style).

Prevents representational collapse by penalising off-diagonal entries in
the covariance matrix of the projected target embeddings.

Algorithm:
  1. Flatten z_target to [N, d_model] — accepts any leading dimensions
     (e.g. [B, n_latents, d] for Branch A or [B, N_span, d] for Branch B).
  2. Project: Z_proj = Z_flat @ W   →  [N, proj_dim].
  3. Center Z_proj along N.
  4. Compute covariance C = (Z_proj^T @ Z_proj) / (N - 1).
  5. Loss = || C - I ||_F  (Frobenius norm of deviation from identity).

No stop_gradient is applied here — gradients flow back through Z_target
to the target encoder.  This is the sole training signal for that encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CovarianceRegularizationLoss(nn.Module):
    """
    Parameters
    ----------
    d_model:
        Hidden dimension of the target embeddings.
    proj_dim:
        Projection dimension.  Default 64.
    """

    def __init__(self, d_model: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, proj_dim, bias=False)
        self.proj_dim = proj_dim

    def forward(self, z_target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_target:
            FloatTensor of shape (..., d_model) — any number of leading dims.
            All leading dimensions are flattened into the sample dimension N.
            Gradients flow through this tensor — no detach here.

        Returns
        -------
        Scalar covariance regularization loss.
        """
        d_model = z_target.shape[-1]

        # Flatten all leading dims into N: (..., d_model) → (N, d_model)
        z_flat = z_target.reshape(-1, d_model)

        # Project: (N, proj_dim)
        z_proj = self.proj(z_flat)

        N = z_proj.shape[0]
        if N <= 1:
            return z_proj.new_zeros(())

        # Center along N
        z_proj = z_proj - z_proj.mean(dim=0, keepdim=True)

        # Covariance: (proj_dim, proj_dim)
        cov = (z_proj.T @ z_proj) / (N - 1)

        # Loss: || C - I ||_F
        identity = torch.eye(self.proj_dim, device=cov.device, dtype=cov.dtype)
        loss = (cov - identity).norm(p="fro")
        return loss
