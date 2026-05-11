"""
Projection head for JEPA representation spaces.

Applies a 1-layer Linear → BatchNorm1d projection to break the unit-sphere
constraint imposed by the final LayerNorm of the encoder / perceiver.

Background
----------
Standard Transformer blocks end with LayerNorm, which keeps each token
embedding on a (approximately) unit hypersphere.  Covariance-based anti-
collapse objectives (VICReg, CovReg) need a free representation space to
push dimensions apart — they fight against LayerNorm's constraint.

Adding Linear + BN1d after the perceiver output:
  1. Maps out of the LN-constrained space into a free one.
  2. BN running statistics independently normalise per-feature variance.
  3. Provides a clean gradient highway that bypasses the LN saturation.

This mirrors the design in LeWorldModel (Meo et al., 2025) applied to a
Vision Transformer.

Usage
-----
Applied symmetrically to BOTH pathways:

  Target pathway:
    target_pooler (→ LayerNorm) → ProjectionHead → Z_tgt_proj

  Predictor pathway:
    predictor    (→ LayerNorm) → ProjectionHead → Z_hat_proj

  Loss:
    MSE(Z_hat_proj, stop_grad(Z_tgt_proj))  +  λ · CovReg(Z_tgt_proj)

The downstream linear probe uses the *raw* perceiver output (before this
head), following standard SSL practice where the projection head is treated
as a training artefact rather than part of the learned representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    1-layer Linear + BatchNorm1d projection.

    Works with any leading batch dimensions: the tensor is flattened to
    (N, d_model) before the linear and BN operations, then reshaped back.
    This correctly handles both 2-D inputs (N, d) and 3-D inputs
    (B, n_latents, d).

    Parameters
    ----------
    d_model:
        Input and output dimension.  The projection is square (d → d).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # bias=False because BN has its own affine shift
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.bn     = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: FloatTensor (..., d_model)
            Any number of leading dimensions are supported.

        Returns
        -------
        FloatTensor (..., d_model) — same shape as input.
        """
        shape = x.shape
        flat  = x.reshape(-1, shape[-1])   # (N, d_model)
        out   = self.bn(self.linear(flat))  # (N, d_model)
        return out.reshape(shape)
