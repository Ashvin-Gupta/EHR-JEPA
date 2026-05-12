"""
SIGReg Loss (Sketched Isotropic Gaussian Regularization).

Replaces covariance Frobenius regularization. Encourages embeddings to match an
isotropic Gaussian via the Epps–Pulley statistic on random 1D projections.

Random directions A must be identical on every GPU when using DDP: we sample A
on CPU with ``torch.Generator.manual_seed(global_step)`` then move to device,
so all ranks get the same matrix without relying on CUDA RNG parity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist


class SIGRegLoss(nn.Module):
    """
    Parameters
    ----------
    num_slices:
        Number of random 1D projection directions (sketches) per forward pass.
    """

    def __init__(self, num_slices: int = 32) -> None:
        super().__init__()
        self.num_slices = num_slices

    def forward(
        self,
        z_target: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_target:
            FloatTensor of shape (..., d_model). Gradients flow through this tensor.
        global_step:
            Training step index. When set, seeds the CPU Generator so every rank
            draws the same projection matrix A (required for correct DDP SIGReg).
        """
        d_model = z_target.shape[-1]
        x = z_target.reshape(-1, d_model)
        N_local = x.shape[0]

        if N_local <= 1:
            return x.new_zeros(())

        dev = x.device
        dtype = x.dtype

        # Sample A on CPU with a deterministic seed so all ranks share identical A.
        g = torch.Generator(device="cpu")
        if global_step is not None:
            g.manual_seed(int(global_step))
        else:
            g.seed()

        A = torch.randn(
            (d_model, self.num_slices),
            generator=g,
            dtype=torch.float32,
        )
        A = A.to(device=dev, dtype=dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Integration grid for Epps–Pulley / characteristic function distance
        t = torch.linspace(-5.0, 5.0, 17, device=dev, dtype=dtype)
        exp_f = torch.exp(-0.5 * t**2)

        # Projections: (N_local, num_slices), then (N_local, num_slices, T)
        proj = x @ A
        x_t = proj.unsqueeze(-1) * t.view(1, 1, -1)
        ecf = torch.exp(1j * x_t).mean(dim=0)

        if dist.is_available() and dist.is_initialized():
            # Average empirical CF across ranks (equal batch sizes → pooled CF).
            ecf_c = torch.view_as_real(ecf)
            dist.all_reduce(ecf_c, op=dist.ReduceOp.AVG)
            ecf = torch.view_as_complex(ecf_c.contiguous())

        err = (ecf - exp_f).abs().square() * exp_f

        _trapz = getattr(torch, "trapezoid", torch.trapz)
        stat = _trapz(err, t, dim=-1)

        return stat.mean()


# Backward-compatible alias (older checkpoints / docs may reference this name).
CovarianceRegularizationLoss = SIGRegLoss
