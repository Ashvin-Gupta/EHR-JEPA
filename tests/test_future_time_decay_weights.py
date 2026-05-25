"""Vectorized future time-decay loss weights."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loss.jepa_loss import future_time_decay_weights


def test_decay_weights_vectorized():
    delta = torch.tensor([[0.0, 60.0, 120.0], [30.0, 0.0, 0.0]])
    lam, floor = 0.01, 0.05
    w = future_time_decay_weights(delta, lam, floor)
    expected = torch.exp(-lam * delta.clamp(min=0)).clamp(min=floor)
    assert torch.allclose(w, expected)


def test_decay_weights_respects_mask():
    delta = torch.tensor([[10.0, 20.0]])
    mask = torch.tensor([[1.0, 0.0]])
    w = future_time_decay_weights(delta, 0.1, 0.0, mask=mask)
    assert w[0, 1].item() == 0.0
    assert w[0, 0].item() > 0.0


if __name__ == "__main__":
    test_decay_weights_vectorized()
    test_decay_weights_respects_mask()
    print("ok")
