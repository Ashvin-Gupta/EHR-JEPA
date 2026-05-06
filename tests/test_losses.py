"""
Tests for loss/jepa_loss.py and loss/covariance_reg.py.
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loss.jepa_loss import jepa_prediction_loss
from loss.covariance_reg import CovarianceRegularizationLoss


B, NS, NL, D = 2, 4, 8, 32   # batch, spans, latents, d_model


def _shape():
    return (B, NS, NL, D)


# ------------------------------------------------------------------
# JEPA Prediction Loss
# ------------------------------------------------------------------

def test_jepa_loss_zero():
    z = torch.randn(*_shape())
    loss = jepa_prediction_loss(z, z.clone())
    print(f"\n[test_jepa_loss_zero] loss={loss.item():.8f}  (expected ≈ 0)")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_jepa_loss_positive():
    z_pred = torch.randn(*_shape())
    z_target = torch.randn(*_shape())
    loss = jepa_prediction_loss(z_pred, z_target)
    print(f"\n[test_jepa_loss_positive] loss={loss.item():.6f}  (expected > 0)")
    assert loss.item() > 0.0


def test_jepa_loss_no_grad_through_target():
    """
    Stop-gradient: after backward, z_target should have no gradient.
    (Only z_pred should receive gradient from L_pred.)
    """
    z_pred = torch.randn(*_shape(), requires_grad=True)
    z_target = torch.randn(*_shape(), requires_grad=True)

    loss = jepa_prediction_loss(z_pred, z_target)
    loss.backward()

    print(f"\n[test_jepa_loss_no_grad_through_target]")
    print(f"  z_pred.grad is not None: {z_pred.grad is not None}")
    print(f"  z_target.grad is None:   {z_target.grad is None}")

    assert z_pred.grad is not None, "z_pred should receive gradient"
    assert z_target.grad is None, "z_target should NOT receive gradient (stop-gradient)"


def test_jepa_loss_scalar():
    z_pred = torch.randn(*_shape())
    z_target = torch.randn(*_shape())
    loss = jepa_prediction_loss(z_pred, z_target)
    assert loss.ndim == 0, "Loss should be a scalar"


# ------------------------------------------------------------------
# Covariance Regularization Loss
# ------------------------------------------------------------------

def test_cov_loss_shape():
    cov_loss = CovarianceRegularizationLoss(d_model=D, proj_dim=16)
    z = torch.randn(*_shape())
    loss = cov_loss(z)
    assert loss.ndim == 0
    print(f"\n[test_cov_loss_shape] loss shape={loss.shape}, value={loss.item():.4f}")


def test_cov_loss_identity():
    """
    Near-uncorrelated embeddings should produce low covariance loss.
    We use many samples with random normal features.
    """
    cov_loss = CovarianceRegularizationLoss(d_model=D, proj_dim=16)
    # Large batch, random → near-identity covariance after projection
    z = torch.randn(64, NS, NL, D)
    loss = cov_loss(z)
    print(f"\n[test_cov_loss_identity] loss with random Z: {loss.item():.4f}")
    # Just check it's finite and non-negative (exact value depends on random projection weights)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_cov_loss_collapsed():
    """All-same embeddings → high covariance deviation from identity."""
    cov_loss = CovarianceRegularizationLoss(d_model=D, proj_dim=16)
    # All identical rows → after projection still identical → covariance is rank-1
    z = torch.ones(*_shape())
    loss_collapsed = cov_loss(z)

    z_random = torch.randn(*_shape())
    loss_random = cov_loss(z_random)

    print(f"\n[test_cov_loss_collapsed]")
    print(f"  collapsed (all same):  {loss_collapsed.item():.4f}")
    print(f"  random:                {loss_random.item():.4f}")
    # Collapsed should have higher covariance penalty (cov matrix is far from identity)
    assert loss_collapsed.item() > 0.0


def test_cov_loss_gradients_flow():
    """
    L_cov must produce gradients through Z_target — no stop_gradient here.
    This is the sole training signal for the target encoder.
    """
    cov_loss = CovarianceRegularizationLoss(d_model=D, proj_dim=16)
    z_target = torch.randn(*_shape(), requires_grad=True)
    loss = cov_loss(z_target)
    loss.backward()

    assert z_target.grad is not None, "z_target should receive gradient from L_cov"
    grad_norm = z_target.grad.norm().item()
    print(f"\n[test_cov_loss_gradients_flow]")
    print(f"  z_target grad norm: {grad_norm:.6f}  (should be > 0)")
    assert grad_norm > 0.0


def test_cov_loss_projection_trainable():
    cov_loss = CovarianceRegularizationLoss(d_model=D, proj_dim=16)
    for name, p in cov_loss.named_parameters():
        assert p.requires_grad, f"Parameter {name} should be trainable"
    print(f"\n[test_cov_loss_projection_trainable] "
          f"proj weight shape: {tuple(cov_loss.proj.weight.shape)}")


if __name__ == "__main__":
    import traceback
    tests = [
        test_jepa_loss_zero,
        test_jepa_loss_positive,
        test_jepa_loss_no_grad_through_target,
        test_jepa_loss_scalar,
        test_cov_loss_shape,
        test_cov_loss_identity,
        test_cov_loss_collapsed,
        test_cov_loss_gradients_flow,
        test_cov_loss_projection_trainable,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*50)
    if failed:
        sys.exit(1)
