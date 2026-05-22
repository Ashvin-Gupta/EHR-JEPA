"""Tests for training/checkpoint_utils.py JEPA/BERT split helpers."""

from __future__ import annotations

import os
import tempfile

import torch

from training.checkpoint_utils import (
    ar_backbone_state_dict_for_arehrmodel,
    assert_jepa_split_covers_full,
    bert_backbone_state_dict_for_bertehrmodel,
    load_jepa_backbone_state_dict,
    merge_jepa_split_state_dict,
    save_jepa_split_checkpoints,
    split_ar_trainer_state_dict,
    split_bert_trainer_state_dict,
    split_jepa_state_dict,
)


def test_split_jepa_partitions_all_keys():
    fake_sd = {
        "embedding.weight": torch.zeros(3),
        "encoder.layer.foo": torch.ones(2),
        "cls_token": torch.randn(8),
        "context_pooler.latent_tokens": torch.randn(4),
        "mask_token": torch.randn(5),
    }
    assert_jepa_split_covers_full(fake_sd)
    parts = split_jepa_state_dict(fake_sd)
    merged = merge_jepa_split_state_dict(parts["backbone"], parts["jepa_aux"])
    assert merged.keys() == fake_sd.keys()
    for k in fake_sd:
        assert torch.equal(merged[k], fake_sd[k])


def test_save_jepa_split_roundtrip_files():
    fake_sd = {
        "embedding.w": torch.tensor(1.0),
        "encoder.w": torch.tensor(2.0),
        "predictor.w": torch.tensor(3.0),
    }
    with tempfile.TemporaryDirectory() as d:
        save_jepa_split_checkpoints(fake_sd, d)
        bb = torch.load(os.path.join(d, "backbone.pt"))
        aux = torch.load(os.path.join(d, "jepa_aux.pt"))
        assert set(bb.keys()) == {"embedding.w", "encoder.w"}
        assert set(aux.keys()) == {"predictor.w"}

    fake_with_cls = {
        "embedding.w": torch.tensor(1.0),
        "encoder.w": torch.tensor(2.0),
        "cls_token": torch.tensor(3.0),
        "predictor.w": torch.tensor(4.0),
    }
    parts = split_jepa_state_dict(fake_with_cls)
    assert "cls_token" in parts["backbone"]
    assert "predictor.w" in parts["jepa_aux"]


def test_load_jepa_backbone_from_nested_checkpoint():
    full = {
        "embedding.a": torch.tensor(7.0),
        "encoder.b": torch.tensor(8.0),
        "other": torch.tensor(9.0),
    }
    ckpt = {"model_state": full, "epoch": 1}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
        torch.save(ckpt, path)
    try:
        bb = load_jepa_backbone_state_dict(path)
        assert set(bb.keys()) == {"embedding.a", "encoder.b"}
    finally:
        os.unlink(path)

    full_cls = {
        "embedding.a": torch.tensor(7.0),
        "encoder.b": torch.tensor(8.0),
        "cls_token": torch.tensor(9.0),
        "other": torch.tensor(10.0),
    }
    ckpt2 = {"model_state": full_cls, "epoch": 1}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f2:
        path2 = f2.name
        torch.save(ckpt2, path2)
    try:
        bb2 = load_jepa_backbone_state_dict(path2)
        assert set(bb2.keys()) == {"embedding.a", "encoder.b", "cls_token"}
    finally:
        os.unlink(path2)


def test_bert_split_and_strip_prefix():
    sd = {
        "model.embedding.weight": torch.randn(2, 2),
        "model.encoder.layer": torch.randn(3),
        "model.cls_token": torch.randn(1),
    }
    parts = split_bert_trainer_state_dict(sd)
    inner = bert_backbone_state_dict_for_bertehrmodel(parts["backbone"])
    assert set(inner.keys()) == {"embedding.weight", "encoder.layer"}


def test_ar_split_and_strip_prefix():
    sd = {
        "model.embedding.weight": torch.randn(2, 2),
        "model.encoder.layer": torch.randn(3),
        "model.cls_token": torch.randn(1),
        "model.eos_token": torch.randn(1),
        "model.lm_head.0.weight": torch.randn(4, 4),
    }
    parts = split_ar_trainer_state_dict(sd)
    inner = ar_backbone_state_dict_for_arehrmodel(parts["backbone"])
    assert set(inner.keys()) == {"embedding.weight", "encoder.layer"}
    assert "model.cls_token" in parts["ar_aux"]
    assert "model.eos_token" in parts["ar_aux"]
