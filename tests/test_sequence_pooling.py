"""Tests for models/sequence_pooling.py."""

from __future__ import annotations

import pytest
import torch

from models.sequence_pooling import (
    get_config_pooling,
    get_eval_pooling,
    mean_pool_sequence,
    parse_pooling_mode,
)


def test_parse_pooling_mode_aliases():
    assert parse_pooling_mode("cls") == "cls"
    assert parse_pooling_mode("mean_pool") == "mean_pool"
    assert parse_pooling_mode("mean") == "mean_pool"
    assert parse_pooling_mode("mean-pool") == "mean_pool"
    with pytest.raises(ValueError):
        parse_pooling_mode("max")


def test_mean_pool_ignores_padding():
    h = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [99.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0]])
    out = mean_pool_sequence(h, mask)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[1.5, 0.0]]))


def test_get_config_pooling_sections():
    cfg = {"downstream_eval": {"pooling": "mean_pool"}, "downstream": {"pooling": "cls"}}
    assert get_config_pooling(cfg, "downstream_eval") == "mean_pool"
    assert get_config_pooling(cfg, "downstream") == "cls"
    assert get_eval_pooling(cfg) == "mean_pool"
