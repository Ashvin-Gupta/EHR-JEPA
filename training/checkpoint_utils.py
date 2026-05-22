"""
Split full-model checkpoints into resume-friendly bundles.

JEPA (JEPATrainer.state_dict):
  - backbone.pt   — EventEmbedding + shared encoder + cls_token
  - jepa_aux.pt   — poolers, predictor, SIGReg-free aux modules, etc.

BERT (BERTTrainer.state_dict, keys under ``model.``):
  - backbone.pt   — model.embedding + model.encoder
  - bert_aux.pt   — CLS, MLM head, mask embedding (continue MLM pretrain)

AR (ARTrainer.state_dict, keys under ``model.``):
  - backbone.pt   — model.embedding + model.encoder
  - ar_aux.pt     — CLS, EOS, LM head (continue AR pretrain)
"""

from __future__ import annotations

import os
from typing import Dict

import torch


def split_jepa_state_dict(full_sd: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Partition JEPATrainer state_dict into backbone vs JEPA-auxiliary tensors."""
    backbone: Dict[str, torch.Tensor] = {}
    aux: Dict[str, torch.Tensor] = {}
    for k, v in full_sd.items():
        if (
            k.startswith("embedding.")
            or k.startswith("encoder.")
            or k == "cls_token"
        ):
            backbone[k] = v
        else:
            aux[k] = v
    return {"backbone": backbone, "jepa_aux": aux}


def split_bert_trainer_state_dict(
    full_sd: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Partition BERTTrainer state_dict (DDP may add ``module.`` prefix)."""
    backbone: Dict[str, torch.Tensor] = {}
    aux: Dict[str, torch.Tensor] = {}

    for k, v in full_sd.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        if nk.startswith("model.embedding.") or nk.startswith("model.encoder."):
            backbone[k] = v
        else:
            aux[k] = v
    return {"backbone": backbone, "bert_aux": aux}


def merge_jepa_split_state_dict(
    backbone: Dict[str, torch.Tensor],
    jepa_aux: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Merge backbone + jepa_aux into one state_dict (order: backbone first)."""
    out: Dict[str, torch.Tensor] = {}
    out.update(backbone)
    out.update(jepa_aux)
    return out


def assert_jepa_split_covers_full(full_sd: Dict[str, torch.Tensor]) -> None:
    """Raise if split does not partition all keys."""
    parts = split_jepa_state_dict(full_sd)
    merged = merge_jepa_split_state_dict(parts["backbone"], parts["jepa_aux"])
    if set(merged.keys()) != set(full_sd.keys()):
        missing = set(full_sd.keys()) - set(merged.keys())
        extra = set(merged.keys()) - set(full_sd.keys())
        raise AssertionError(f"JEPA split key mismatch missing={missing} extra={extra}")


def load_jepa_backbone_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load or extract JEPA embedding+encoder weights from a checkpoint file."""
    obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        obj = obj["model_state"]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a state dict or checkpoint at {checkpoint_path!r}")
    if obj and all(
        k.startswith("embedding.") or k.startswith("encoder.") or k == "cls_token"
        for k in obj
    ):
        return obj
    return split_jepa_state_dict(obj)["backbone"]


def load_bert_backbone_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Embedding+encoder tensors from BERT backbone or full trainer checkpoint."""
    obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        obj = obj["model_state"]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected checkpoint dict at {checkpoint_path!r}")
    return split_bert_trainer_state_dict(obj)["backbone"]


def bert_backbone_state_dict_for_bertehrmodel(
    backbone_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Map ``model.embedding.*`` / DDP ``module.`` keys → ``embedding.*`` for BERTEHRModel."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in backbone_sd.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        nk = nk[len("model.") :] if nk.startswith("model.") else nk
        out[nk] = v
    return out


def save_jepa_split_checkpoints(model_state: Dict[str, torch.Tensor], directory: str) -> None:
    """Write backbone.pt and jepa_aux.pt next to best/last checkpoints."""
    parts = split_jepa_state_dict(model_state)
    os.makedirs(directory, exist_ok=True)
    torch.save(parts["backbone"], os.path.join(directory, "backbone.pt"))
    torch.save(parts["jepa_aux"], os.path.join(directory, "jepa_aux.pt"))


def save_bert_split_checkpoints(model_state: Dict[str, torch.Tensor], directory: str) -> None:
    parts = split_bert_trainer_state_dict(model_state)
    os.makedirs(directory, exist_ok=True)
    torch.save(parts["backbone"], os.path.join(directory, "backbone.pt"))
    torch.save(parts["bert_aux"], os.path.join(directory, "bert_aux.pt"))


def split_ar_trainer_state_dict(
    full_sd: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Partition ARTrainer state_dict (DDP may add ``module.`` prefix)."""
    backbone: Dict[str, torch.Tensor] = {}
    aux: Dict[str, torch.Tensor] = {}

    for k, v in full_sd.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        if nk.startswith("model.embedding.") or nk.startswith("model.encoder."):
            backbone[k] = v
        else:
            aux[k] = v
    return {"backbone": backbone, "ar_aux": aux}


def load_ar_backbone_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Embedding+encoder tensors from AR backbone or full trainer checkpoint."""
    obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        obj = obj["model_state"]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected checkpoint dict at {checkpoint_path!r}")
    return split_ar_trainer_state_dict(obj)["backbone"]


def ar_backbone_state_dict_for_arehrmodel(
    backbone_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Map ``model.embedding.*`` / DDP ``module.`` keys → ``embedding.*`` for AREHRModel."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in backbone_sd.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        nk = nk[len("model.") :] if nk.startswith("model.") else nk
        out[nk] = v
    return out


def save_ar_split_checkpoints(model_state: Dict[str, torch.Tensor], directory: str) -> None:
    parts = split_ar_trainer_state_dict(model_state)
    os.makedirs(directory, exist_ok=True)
    torch.save(parts["backbone"], os.path.join(directory, "backbone.pt"))
    torch.save(parts["ar_aux"], os.path.join(directory, "ar_aux.pt"))


def load_backbone_into_module(
    module: torch.nn.Module,
    backbone_path: str,
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load only keys present in ``backbone.pt`` (embedding + encoder weights)
    into a module that owns ``embedding`` and ``encoder`` submodules.
    """
    bb = torch.load(backbone_path, map_location="cpu")
    if not isinstance(bb, dict):
        raise ValueError(f"Expected state dict in {backbone_path}")
    missing, unexpected = module.load_state_dict(bb, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"load_backbone_into_module strict=True failed missing={missing} unexpected={unexpected}")
    return module
