"""
Tests for loss/jepa_loss.py and loss/covariance_reg.py (SIGReg).
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loss.jepa_loss import jepa_prediction_loss, jepa_prediction_loss_weighted
from loss.covariance_reg import SIGRegLoss


B, NS, NL, D = 2, 4, 8, 32


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


def test_jepa_loss_weighted_near_term_dominates():
    """High weight on token 0 makes mismatch at 0 cost more than mismatch at 1."""
    B, N, D = 1, 3, 8
    z_pred = torch.zeros(B, N, D)
    z_tgt = torch.zeros(B, N, D)
    z_tgt[0, 1, 0] = 1.0
    w = torch.tensor([[10.0, 1.0, 1.0]])
    l_baseline = jepa_prediction_loss_weighted(z_pred, z_tgt, w)
    z_wrong_near = z_pred.clone()
    z_wrong_near[0, 0, 0] = 1.0
    l_near = jepa_prediction_loss_weighted(z_wrong_near, z_tgt, w)
    z_wrong_far = z_pred.clone()
    z_wrong_far[0, 1, 0] = 1.0
    l_far = jepa_prediction_loss_weighted(z_wrong_far, z_tgt, w)
    assert l_baseline.item() < l_near.item()
    assert l_near.item() > l_far.item()


def test_time_decay_weight_floor():
    import math

    lam, floor = 0.05, 0.05
    w_far = max(floor, math.exp(-lam * 10_000.0))
    w_near = max(floor, math.exp(-lam * 0.0))
    assert w_far == floor
    assert w_near == pytest.approx(1.0)


def test_jepa_loss_weighted_no_grad_through_target():
    z_pred = torch.randn(2, 5, 16, requires_grad=True)
    z_target = torch.randn(2, 5, 16, requires_grad=True)
    w = torch.ones(2, 5)
    loss = jepa_prediction_loss_weighted(z_pred, z_target, w)
    loss.backward()
    assert z_pred.grad is not None
    assert z_target.grad is None


# ------------------------------------------------------------------
# SIGReg Loss
# ------------------------------------------------------------------

def test_sigreg_loss_shape():
    sig = SIGRegLoss(num_slices=16)
    z = torch.randn(*_shape())
    loss = sig(z, global_step=1)
    assert loss.ndim == 0
    print(f"\n[test_sigreg_loss_shape] loss shape={loss.shape}, value={loss.item():.4f}")


def test_sigreg_deterministic_given_global_step():
    """Same global_step → same loss on same input (CPU Generator sync)."""
    sig = SIGRegLoss(num_slices=16)
    z = torch.randn(*_shape())
    l1 = sig(z, global_step=12345)
    l2 = sig(z, global_step=12345)
    assert l1.item() == pytest.approx(l2.item(), rel=0.0, abs=1e-6)


def test_sigreg_gradients_flow():
    sig = SIGRegLoss(num_slices=16)
    z_target = torch.randn(*_shape(), requires_grad=True)
    loss = sig(z_target, global_step=7)
    loss.backward()

    assert z_target.grad is not None
    assert z_target.grad.norm().item() > 0.0


def test_sigreg_no_trainable_submodules():
    sig = SIGRegLoss(num_slices=32)
    assert list(sig.parameters()) == []


def test_covariance_alias_import():
    """Backward-compat alias still imports."""
    from loss.covariance_reg import CovarianceRegularizationLoss
    assert CovarianceRegularizationLoss is SIGRegLoss


if __name__ == "__main__":
    import traceback
    tests = [
        test_jepa_loss_zero,
        test_jepa_loss_positive,
        test_jepa_loss_no_grad_through_target,
        test_jepa_loss_scalar,
        test_jepa_loss_weighted_near_term_dominates,
        test_time_decay_weight_floor,
        test_jepa_loss_weighted_no_grad_through_target,
        test_sigreg_loss_shape,
        test_sigreg_deterministic_given_global_step,
        test_sigreg_gradients_flow,
        test_sigreg_no_trainable_submodules,
        test_covariance_alias_import,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception:
            print(f"\n  FAILED: {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 50)
    if failed:
        sys.exit(1)
