"""
Downstream binary classification via a frozen JEPA encoder + linear probe.

Architecture
------------
Pretrained (frozen):
    EventEmbedding  →  EHRTransformerEncoder  →  LatentCrossAttentionPool
    Produces a fixed-size representation:  [B, n_latents * d_model]

Trainable:
    Linear(n_latents * d_model  →  1)  →  Sigmoid
    Trained with BCEWithLogitsLoss.

Usage
-----
Use run_linear_probe.py to train from a checkpoint end-to-end, or import
FrozenEHREncoder / LinearProbe directly for custom experiments.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.event_embedding import EventEmbedding
from models.transformer_encoder import EHRTransformerEncoder
from models.latent_pooling import LatentCrossAttentionPool


# ---------------------------------------------------------------------------
# Frozen encoder
# ---------------------------------------------------------------------------

class FrozenEHREncoder(nn.Module):
    """
    Wraps the pretrained embedding, encoder, and latent pooler from a JEPA
    checkpoint.  All parameters are frozen; no gradients flow through this
    module.

    Output shape: [B, n_latents * d_model]  (Perceiver mode)
              or  [B, d_model]              (mean-pooling fallback when pooler=None)

    Parameters
    ----------
    embedding:
        Pretrained EventEmbedding.
    encoder:
        Pretrained EHRTransformerEncoder.
    pooler:
        Pretrained LatentCrossAttentionPool (context pooler from JEPA).
        Pass None to fall back to mean pooling over real tokens — useful
        when use_perceiver=False (Branch B) or for quick experiments.
    """

    def __init__(
        self,
        embedding: EventEmbedding,
        encoder: EHRTransformerEncoder,
        pooler: Optional[LatentCrossAttentionPool],
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder   = encoder
        self.pooler    = pooler

        for p in self.parameters():
            p.requires_grad_(False)

        # Compute once at init as a plain int stored in __dict__.
        # Avoids the silent-failure pattern where an AttributeError inside a
        # @property on nn.Module gets re-routed to __getattr__ and looks like
        # the property itself doesn't exist.
        d_model = encoder.config.d_model
        if pooler is not None:
            # latent_tokens is the nn.Parameter holding the learned queries
            n_latents = pooler.latent_tokens.shape[0]
            self.output_dim: int = n_latents * d_model
        else:
            self.output_dim = d_model   # mean-pool fallback

    @torch.no_grad()
    def forward(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        codes:          LongTensor  (B, L)
        attention_mask: LongTensor  (B, L)
        values / z_scores / delta_times / value_mask: optional FloatTensors (B, L)

        Returns
        -------
        FloatTensor (B, n_latents * d_model)
        """
        x = self.embedding(
            codes,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask.float() if value_mask is not None else None,
        )  # (B, L, d_model)

        # EHRTransformerEncoder takes attention_mask (1=real, 0=pad) directly
        h = self.encoder(x, attention_mask=attention_mask)  # (B, L, d_model)

        if self.pooler is not None:
            # Perceiver latent pooling → fixed-size representation
            # LatentCrossAttentionPool expects key_padding_mask (True=ignore)
            pad_mask = attention_mask == 0
            z = self.pooler(h, key_padding_mask=pad_mask)  # (B, n_latents, d_model)
            return z.flatten(1)                             # (B, n_latents * d_model)
        else:
            # Mean pooling fallback over real tokens
            real = attention_mask.unsqueeze(-1).float()     # (B, L, 1)
            return (h * real).sum(1) / real.sum(1).clamp(min=1)  # (B, d_model)


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """
    Single linear layer for binary classification on top of a frozen encoder.

    Parameters
    ----------
    input_dim:
        Dimensionality of the encoder output (n_latents * d_model).
    dropout:
        Dropout probability applied before the linear layer (default 0.0).
    """

    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z: FloatTensor (B, input_dim)

        Returns
        -------
        logits: FloatTensor (B,)  — raw (un-sigmoidised) log-odds
        """
        return self.net(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_linear_probe(
    encoder: FrozenEHREncoder,
    probe: LinearProbe,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    n_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
    on_epoch_end: Optional[callable] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Train the linear probe with the encoder frozen.

    Returns
    -------
    (history, final_val_metrics)
        history            — per-epoch lists for train/val loss and all metrics
        final_val_metrics  — full metric dict from the last val evaluation
                             (empty dict if no val_loader)
    """
    import time as _t

    _device = torch.device(device)
    encoder.to(_device).eval()
    probe.to(_device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history: Dict[str, List[float]] = {k: [] for k in (
        "train_loss", "val_loss",
        "train_auroc", "val_auroc",
        "train_aupr",  "val_aupr",
        "train_recall","val_recall",
        "train_precision","val_precision",
        "train_accuracy", "val_accuracy",
        "epoch_runtime_s",
    )}

    final_val: Dict[str, float] = {}

    for epoch in range(n_epochs):
        t0 = _t.perf_counter()
        probe.train()
        epoch_loss = 0.0
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for batch in train_loader:
            codes       = batch["codes"].to(_device, non_blocking=True)
            attn_mask   = batch["attention_mask"].to(_device, non_blocking=True)
            labels      = batch["labels"].float().to(_device, non_blocking=True)
            values      = batch["values"].to(_device, non_blocking=True)      if "values"      in batch else None
            z_scores    = batch["z_scores"].to(_device, non_blocking=True)    if "z_scores"    in batch else None
            delta_times = batch["delta_times"].to(_device, non_blocking=True) if "delta_times" in batch else None
            value_mask  = batch["value_mask"].to(_device, non_blocking=True)  if "value_mask"  in batch else None

            z      = encoder(codes, attn_mask, values, z_scores, delta_times, value_mask)
            logits = probe(z)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        train_m = _compute_all_metrics(
            torch.cat(all_labels), torch.cat(all_logits)
        )
        epoch_rt = _t.perf_counter() - t0

        history["train_loss"].append(avg_train_loss)
        history["epoch_runtime_s"].append(epoch_rt)
        for k, v in train_m.items():
            history[f"train_{k}"].append(v)

        line = (f"  probe epoch {epoch+1}/{n_epochs}  "
                f"loss={avg_train_loss:.4f}  auroc={train_m['auroc']:.4f}")

        if val_loader is not None:
            val_loss, val_m = _eval_probe(encoder, probe, val_loader, criterion, _device)
            history["val_loss"].append(val_loss)
            for k, v in val_m.items():
                history[f"val_{k}"].append(v)
            final_val = {"loss": val_loss, **val_m}
            line += (f"  val_loss={val_loss:.4f}  val_auroc={val_m['auroc']:.4f}"
                     f"  val_aupr={val_m['aupr']:.4f}")

        if verbose:
            print(line)

        if on_epoch_end is not None:
            on_epoch_end(epoch, {"train_loss": avg_train_loss, **{f"train_{k}": v
                                  for k, v in train_m.items()}, **{f"val_{k}": v
                                  for k, v in final_val.items()}})

    return history, final_val


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_probe(
    encoder: FrozenEHREncoder,
    probe: LinearProbe,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Returns (avg_loss, metrics_dict) with auroc/aupr/recall/precision/accuracy."""
    probe.eval()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        codes       = batch["codes"].to(device, non_blocking=True)
        attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
        labels      = batch["labels"].float().to(device, non_blocking=True)
        values      = batch["values"].to(device, non_blocking=True)      if "values"      in batch else None
        z_scores    = batch["z_scores"].to(device, non_blocking=True)    if "z_scores"    in batch else None
        delta_times = batch["delta_times"].to(device, non_blocking=True) if "delta_times" in batch else None
        value_mask  = batch["value_mask"].to(device, non_blocking=True)  if "value_mask"  in batch else None

        z      = encoder(codes, attn_mask, values, z_scores, delta_times, value_mask)
        logits = probe(z)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / max(len(loader), 1)
    metrics  = _compute_all_metrics(torch.cat(all_labels), torch.cat(all_logits))
    return avg_loss, metrics


def _compute_all_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute AUROC, AUPR, recall, precision and accuracy from raw logits.

    Parameters
    ----------
    labels:  FloatTensor (N,) — ground-truth binary labels (0 / 1)
    logits:  FloatTensor (N,) — raw model output (before sigmoid)
    threshold: decision threshold for recall/precision/accuracy
    """
    probs = torch.sigmoid(logits).float()
    labels = labels.float()
    preds  = (probs >= threshold).float()

    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    tn = ((1 - preds) * (1 - labels)).sum().item()

    precision = tp / max(tp + fp, 1e-8)
    recall    = tp / max(tp + fn, 1e-8)
    accuracy  = (tp + tn) / max(len(labels), 1)

    return {
        "auroc":     _roc_auc(labels, probs),
        "aupr":      _au_pr(labels, probs),
        "recall":    recall,
        "precision": precision,
        "accuracy":  accuracy,
    }


def _roc_auc(labels: torch.Tensor, probs: torch.Tensor) -> float:
    """AUROC via O(n log n) sort-based trapezoidal integration."""
    labels = labels.float()
    n_pos  = labels.sum().item()
    n_neg  = (1 - labels).sum().item()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order  = torch.argsort(probs, descending=True)
    labels = labels[order]
    tp = torch.cumsum(labels, dim=0)
    fp = torch.cumsum(1 - labels, dim=0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    d_fpr = torch.diff(fpr, prepend=fpr.new_zeros(1))
    return float((tpr * d_fpr).sum().item())


def _au_pr(labels: torch.Tensor, probs: torch.Tensor) -> float:
    """AUPR (area under precision-recall curve) via sort-based trapezoidal integration."""
    labels = labels.float()
    n_pos  = labels.sum().item()
    if n_pos == 0:
        return 0.0

    order  = torch.argsort(probs, descending=True)
    labels = labels[order]
    tp = torch.cumsum(labels, dim=0)
    fp = torch.cumsum(1 - labels, dim=0)

    precision = tp / (tp + fp).clamp(min=1e-8)
    recall    = tp / n_pos

    # Prepend (recall=0, precision=1) sentinel for a complete curve
    precision = torch.cat([precision.new_ones(1), precision])
    recall    = torch.cat([recall.new_zeros(1), recall])

    d_recall = torch.diff(recall)
    return float((precision[1:] * d_recall).sum().item())


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_frozen_encoder_from_checkpoint(
    checkpoint_path: str,
    model: "JEPATrainer",  # noqa: F821
) -> FrozenEHREncoder:
    """
    Load best.pt / last.pt into a JEPATrainer instance and return a
    FrozenEHREncoder wrapping its pretrained components.

    Parameters
    ----------
    checkpoint_path:
        Path to a .pt file saved by JEPATrainer.train_loop.
    model:
        A JEPATrainer instance whose architecture matches the checkpoint.
        Build it with build_model() in main.py first.

    Returns
    -------
    FrozenEHREncoder ready for downstream probing.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    if model.context_pooler is None:
        raise ValueError(
            "load_frozen_encoder_from_checkpoint requires a Perceiver model "
            "(use_perceiver=True) — context_pooler must not be None."
        )

    return FrozenEHREncoder(
        embedding=model.embedding,
        encoder=model.encoder,
        pooler=model.context_pooler,
    )
