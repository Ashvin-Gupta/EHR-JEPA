"""
Supervised downstream evaluation on labelled task data.

Configuration lives under ``downstream_eval`` in the YAML (``mode``,
``run_fraction_sweep``, ``head_type``, etc.). CLI flags override when given.

Typical jobs (three separate Slurm / shell runs):
  1. Pretraining: ``python main.py`` or ``torchrun ... main.py`` (not this script).
  2. Supervised finetuning on **full** train data: set ``run_fraction_sweep: false``,
     ``mode: jepa``, ``checkpoint_path: /path/to/best.pt`` (or pass ``--checkpoint``),
     then ``python main_supervised_downstream.py --config configs/ehr_config.yaml``.
  3. Low-data study: same YAML but ``run_fraction_sweep: true`` (or one-off
     ``--low-data``) to loop ``train_fractions``; val/test protocol unchanged.

Modes (``downstream_eval.mode`` or ``--mode``):
  jepa    — Perceiver classifier; optional checkpoint loads backbone weights only.
  scratch — Same as jepa without loading pretrained weights.
  bert    — BERT + CLS head; checkpoint required.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time as _time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.normalizer import ValueNormalizer
from data.vocab import Vocab
from evaluation.bert_supervised import BERTSupervisedClassifier
from evaluation.linear_probe import _compute_all_metrics
from evaluation.supervised_cls import SupervisedCLSClassifier
from evaluation.supervised_perceiver import SupervisedPerceiverClassifier
from main import (
    _ensure_normalizer,
    _ensure_vocab,
    _needs_normalizer,
    build_probe_loaders,
    init_wandb,
    load_config,
    set_seed,
)
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.bert_trainer import BERTEHRModel
from training.checkpoint_utils import (
    bert_backbone_state_dict_for_bertehrmodel,
    load_bert_backbone_state_dict,
    load_jepa_backbone_state_dict,
)


def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_supervised_run_name(cfg: dict, mode: str) -> str:
    task = cfg["data"].get("labels_task", "task")
    de = cfg.get("downstream_eval", {})
    parts = [
        "supervised",
        str(mode),
        str(task),
        str(de.get("head_type", "linear")),
    ]
    m = cfg["model"]
    if m.get("use_value"):
        parts.append("value")
    if m.get("use_time"):
        parts.append("time")
    return "__".join(parts)


def _wandb_log_epoch(
    run,
    prefix: str,
    metrics: Dict[str, float],
    step: int,
) -> None:
    """Log flat keys like probe/train_loss, probe/val_auroc (linear probe style)."""
    if run is None:
        return
    run.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=step)


def _val_wandb_payload(val_m: Dict[str, float]) -> Dict[str, float]:
    payload = {"val_loss": val_m["loss"]}
    for k, v in val_m.items():
        if k != "loss":
            payload[f"val_{k}"] = v
    return payload


def _bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if label_smoothing <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, labels)
    targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets)


def _count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _configure_supervised_freeze(model: nn.Module, freeze_backbone: bool) -> None:
    """Freeze pretrained trunk; keep pooler+head (JEPA) or classification head (BERT) trainable."""
    if not freeze_backbone:
        return
    if isinstance(model, SupervisedPerceiverClassifier):
        for mod in (model.embedding, model.encoder):
            for p in mod.parameters():
                p.requires_grad = False
    elif isinstance(model, SupervisedCLSClassifier):
        for mod in (model.embedding, model.encoder):
            for p in mod.parameters():
                p.requires_grad = False
    elif isinstance(model, BERTSupervisedClassifier):
        for p in model.bert.parameters():
            p.requires_grad = False
    else:
        raise TypeError(f"Unknown supervised model type: {type(model)}")


def _build_supervised_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_scale: float,
    freeze_backbone: bool,
) -> optim.Optimizer:
    if freeze_backbone or backbone_lr_scale >= 1.0:
        return optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=weight_decay,
        )

    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    if isinstance(model, SupervisedPerceiverClassifier):
        for mod in (model.embedding, model.encoder):
            backbone_params.extend(p for p in mod.parameters() if p.requires_grad)
        for mod in (model.pooler, model.head):
            head_params.extend(p for p in mod.parameters() if p.requires_grad)
    elif isinstance(model, SupervisedCLSClassifier):
        for mod in (model.embedding, model.encoder):
            backbone_params.extend(p for p in mod.parameters() if p.requires_grad)
        head_params.extend(p for p in model.head.parameters() if p.requires_grad)
        if model.cls_token.requires_grad:
            head_params.append(model.cls_token)
    elif isinstance(model, BERTSupervisedClassifier):
        backbone_params.extend(p for p in model.bert.parameters() if p.requires_grad)
        head_params.extend(p for p in model.head.parameters() if p.requires_grad)
    else:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * backbone_lr_scale})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    return optim.AdamW(groups, weight_decay=weight_decay)


def _build_jepa_supervised(
    cfg: dict, vocab: Vocab
) -> SupervisedPerceiverClassifier | SupervisedCLSClassifier:
    m = cfg["model"]
    t = cfg.get("transformer", {})
    lp = cfg.get("latent_pooling", {})
    p = cfg.get("predictor", {})
    use_perceiver = bool(p.get("use_perceiver", True))

    d_model = int(m["d_model"])
    use_value = bool(m.get("use_value", False))
    use_time = bool(m.get("use_time", False))
    emb_type = m["embedding_type"]
    n_heads = int(t.get("n_heads", 8))
    n_latents = int(lp.get("n_latents", 16))

    embedding = EventEmbedding(
        EmbeddingConfig(
            embedding_type=emb_type,
            vocab_size=vocab.vocab_size,
            d_model=d_model,
            code_embeddings_path=m.get("code_embeddings_path"),
            encoder_hidden_dim=m.get("encoder_hidden_dim", 768),
            unk_idx=vocab.unk_idx,
            use_value=use_value,
            use_time=use_time,
        )
    )
    encoder = EHRTransformerEncoder(
        TransformerEncoderConfig(
            n_layers=int(t.get("n_layers", 6)),
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=int(t.get("ffn_dim", 1024)),
            dropout=float(t.get("dropout", 0.1)),
        )
    )
    de = cfg.get("downstream_eval", {})
    head_type = str(de.get("head_type", "linear"))
    head_dropout = float(de.get("head_dropout", 0.1))
    if head_type not in ("linear", "mlp"):
        raise ValueError("downstream_eval.head_type must be 'linear' or 'mlp'")

    if use_perceiver:
        pooler = LatentCrossAttentionPool(d_model, n_latents=n_latents, n_heads=n_heads)
        return SupervisedPerceiverClassifier(
            embedding=embedding,
            encoder=encoder,
            pooler=pooler,
            head_type=head_type,  # type: ignore[arg-type]
            head_dropout=head_dropout,
        )

    cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
    return SupervisedCLSClassifier(
        embedding=embedding,
        encoder=encoder,
        cls_token=cls_token,
        head_type=head_type,  # type: ignore[arg-type]
        head_dropout=head_dropout,
    )


def _build_bert_supervised(cfg: dict, vocab: Vocab) -> BERTSupervisedClassifier:
    m = cfg["model"]
    t = cfg.get("transformer", {})
    d_model = int(m["d_model"])
    use_value = bool(m.get("use_value", False))
    use_time = bool(m.get("use_time", False))
    emb_type = m["embedding_type"]
    n_heads = int(t.get("n_heads", 8))

    embedding = EventEmbedding(
        EmbeddingConfig(
            embedding_type=emb_type,
            vocab_size=vocab.vocab_size,
            d_model=d_model,
            code_embeddings_path=m.get("code_embeddings_path"),
            encoder_hidden_dim=m.get("encoder_hidden_dim", 768),
            unk_idx=vocab.unk_idx,
            use_value=use_value,
            use_time=use_time,
        )
    )
    encoder = EHRTransformerEncoder(
        TransformerEncoderConfig(
            n_layers=int(t.get("n_layers", 6)),
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=int(t.get("ffn_dim", 1024)),
            dropout=float(t.get("dropout", 0.1)),
        )
    )
    bert = BERTEHRModel(
        embedding=embedding,
        encoder=encoder,
        vocab_size=vocab.vocab_size,
    )
    de = cfg.get("downstream_eval", {})
    head_type = str(de.get("head_type", "linear"))
    head_dropout = float(de.get("head_dropout", 0.1))
    if head_type not in ("linear", "mlp"):
        raise ValueError("downstream_eval.head_type must be 'linear' or 'mlp'")
    return BERTSupervisedClassifier(
        bert=bert,
        head_type=head_type,  # type: ignore[arg-type]
        head_dropout=head_dropout,
    )


def _load_jepa_backbone(
    model: SupervisedPerceiverClassifier | SupervisedCLSClassifier,
    path: Optional[str],
) -> None:
    if not path:
        return
    sd = load_jepa_backbone_state_dict(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _ = missing, unexpected


def _load_bert_backbone(model: BERTSupervisedClassifier, path: str) -> None:
    bb = load_bert_backbone_state_dict(path)
    inner = bert_backbone_state_dict_for_bertehrmodel(bb)
    model.bert.load_state_dict(inner, strict=False)


def _train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    device: torch.device,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    run=None,
    log_prefix: str = "supervised",
    step_offset: int = 0,
    log_train_every: int = 1,
    val_eval_every: int = 0,
    grad_clip: float = 0.0,
    label_smoothing: float = 0.0,
    early_stopping_patience: int = 0,
    freeze_backbone: bool = False,
    backbone_lr_scale: float = 1.0,
    use_lr_scheduler: bool = False,
    lr_scheduler_patience: int = 2,
    lr_scheduler_factor: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Train and log to W&B.

    log_train_every: log batch train_loss every N optimizer steps (1 = every step).
    val_eval_every:  run val every N steps; 0 = only at end of each epoch (recommended).
    early_stopping_patience: stop if val AUROC does not improve for this many epochs (0=off).

    Test eval loads weights from the epoch with highest val AUROC (when val_loader is set).
    """
    model.to(device)
    _configure_supervised_freeze(model, freeze_backbone)
    n_trainable = _count_trainable_params(model)
    print(f"  [train] Trainable parameters: {n_trainable:,}"
          f"{' (backbone frozen)' if freeze_backbone else ''}")

    opt = _build_supervised_optimizer(
        model, lr, weight_decay, backbone_lr_scale, freeze_backbone
    )
    scheduler = None
    if use_lr_scheduler and val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
        )

    val_metrics: Dict[str, float] = {}
    best_val_auroc = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    global_step = step_offset
    patience_left = early_stopping_patience if early_stopping_patience > 0 else 0
    stopped_early = False

    for epoch in range(n_epochs):
        t0 = _time.perf_counter()
        model.train()
        epoch_loss = 0.0
        train_logits: List[torch.Tensor] = []
        train_labels: List[torch.Tensor] = []

        for batch in train_loader:
            codes = batch["codes"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].float().to(device, non_blocking=True)
            values = batch["values"].to(device) if "values" in batch else None
            z_scores = batch["z_scores"].to(device) if "z_scores" in batch else None
            delta_times = batch["delta_times"].to(device) if "delta_times" in batch else None
            value_mask = batch["value_mask"].to(device) if "value_mask" in batch else None

            opt.zero_grad()
            logits = model(codes, attn, values, z_scores, delta_times, value_mask)
            loss = _bce_with_logits(logits, labels, label_smoothing)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            opt.step()

            epoch_loss += loss.item()
            train_logits.append(logits.detach().cpu())
            train_labels.append(labels.cpu())
            global_step += 1

            if run is not None and log_train_every > 0 and global_step % log_train_every == 0:
                _wandb_log_epoch(
                    run, log_prefix, {"train_loss_step": loss.item()}, global_step
                )

            if (
                val_loader is not None
                and val_eval_every > 0
                and global_step % val_eval_every == 0
            ):
                val_metrics = _eval_loader(
                    model, val_loader, device, label_smoothing=label_smoothing
                )
                model.train()
                _wandb_log_epoch(
                    run, log_prefix, _val_wandb_payload(val_metrics), global_step
                )

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        train_m = _compute_all_metrics(
            torch.cat(train_labels), torch.cat(train_logits)
        )
        epoch_metrics: Dict[str, float] = {
            "train_loss": avg_train_loss,
            "epoch": float(epoch + 1),
            "epoch_runtime_s": _time.perf_counter() - t0,
            **{f"train_{k}": v for k, v in train_m.items()},
        }

        line = (
            f"  epoch {epoch + 1}/{n_epochs}  "
            f"loss={avg_train_loss:.4f}  auroc={train_m['auroc']:.4f}"
        )

        if val_loader is not None:
            val_metrics = _eval_loader(
                model, val_loader, device, label_smoothing=label_smoothing
            )
            epoch_metrics.update(_val_wandb_payload(val_metrics))
            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if patience_left > 0:
                    patience_left = early_stopping_patience
            elif patience_left > 0:
                patience_left -= 1
            if scheduler is not None:
                scheduler.step(val_metrics["auroc"])
                epoch_metrics["lr"] = opt.param_groups[0]["lr"]
            line += (
                f"  val_loss={val_metrics['loss']:.4f}"
                f"  val_auroc={val_metrics['auroc']:.4f}"
                f"  val_aupr={val_metrics['aupr']:.4f}"
            )

        print(line)
        _wandb_log_epoch(run, log_prefix, epoch_metrics, global_step)

        if patience_left == 0 and early_stopping_patience > 0:
            print(
                f"  [early stop] no val AUROC improvement for "
                f"{early_stopping_patience} epoch(s) — stopping at epoch {epoch + 1}"
            )
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        val_metrics = _eval_loader(
            model, val_loader, device, label_smoothing=label_smoothing  # type: ignore[arg-type]
        )
        stop_note = " (early stop)" if stopped_early else ""
        print(
            f"  [best] epoch {best_epoch}/{n_epochs}{stop_note}  "
            f"val_auroc={best_val_auroc:.4f}  (weights restored for test)"
        )

    test_metrics: Dict[str, float] = {}
    if test_loader is not None:
        test_metrics = _eval_loader(
            model, test_loader, device, label_smoothing=label_smoothing
        )
        test_payload = {f"test_{k}": v for k, v in test_metrics.items()}
        _wandb_log_epoch(run, log_prefix, test_payload, global_step + 1)
        if run is not None and best_epoch > 0:
            run.summary[f"{log_prefix}/best_val_epoch"] = best_epoch
            run.summary[f"{log_prefix}/best_val_auroc"] = best_val_auroc
    return val_metrics, test_metrics


@torch.no_grad()
def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for batch in loader:
        codes = batch["codes"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].float().to(device, non_blocking=True)
        values = batch["values"].to(device) if "values" in batch else None
        z_scores = batch["z_scores"].to(device) if "z_scores" in batch else None
        delta_times = batch["delta_times"].to(device) if "delta_times" in batch else None
        value_mask = batch["value_mask"].to(device) if "value_mask" in batch else None

        logits = model(codes, attn, values, z_scores, delta_times, value_mask)
        loss = _bce_with_logits(logits, labels, label_smoothing)
        total += loss.item()
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())

    avg_loss = total / max(len(loader), 1)
    y = torch.cat(labels_list)
    z = torch.cat(logits_list)
    m = _compute_all_metrics(y, z)
    return {"loss": avg_loss, **m}


def _subset_train_loader(
    base_train: DataLoader,
    fraction: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = base_train.dataset
    n = len(ds)
    k = min(n, max(1, int(math.ceil(fraction * n))))
    rng = random.Random(seed + int(fraction * 100000))
    idx = sorted(rng.sample(range(n), k))
    sub = Subset(ds, idx)
    return DataLoader(
        sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=base_train.collate_fn,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised downstream evaluation")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "ehr_config.yaml"))
    parser.add_argument(
        "--mode",
        choices=("jepa", "scratch", "bert"),
        default=None,
        help="Override downstream_eval.mode (jepa | scratch | bert)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override downstream_eval.checkpoint_path",
    )
    parser.add_argument("--task", default=None, help="Override data.labels_task")
    parser.add_argument(
        "--low-data",
        action="store_true",
        help="Force train_fractions sweep (overrides config; same as run_fraction_sweep: true)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging for this run",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.task:
        cfg = copy.deepcopy(cfg)
        cfg["data"]["labels_task"] = args.task

    de = cfg.get("downstream_eval", {})

    mode = args.mode or de.get("mode")
    if mode not in ("jepa", "scratch", "bert"):
        raise SystemExit(
            "Set downstream_eval.mode to jepa, scratch, or bert (or pass --mode)."
        )
    mode = str(mode)

    ckpt = args.checkpoint
    if ckpt is None and de.get("checkpoint_path") not in (None, "", "null"):
        ckpt = str(de["checkpoint_path"])

    do_fraction_sweep = bool(args.low_data) or bool(de.get("run_fraction_sweep", False))

    seed = args.seed if args.seed is not None else cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    vocab = _ensure_vocab(cfg)
    normalizer: Optional[ValueNormalizer] = (
        _ensure_normalizer(cfg) if _needs_normalizer(cfg) else None
    )

    val_max = int(de.get("val_max_files", 5))
    probe_train, probe_val, probe_test = build_probe_loaders(
        cfg,
        vocab,
        normalizer,
        force=True,
        val_max_files_override=val_max,
    )
    if probe_train is None:
        raise SystemExit("Could not build train loader — check labels paths and downstream config.")

    tr = cfg.get("training", {})
    batch_size = int(de.get("batch_size", tr.get("batch_size", 32)))
    num_workers = int(tr.get("num_workers", 4))
    pin_memory = bool(tr.get("pin_memory", True)) and torch.cuda.is_available()

    train_full = DataLoader(
        probe_train.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=probe_train.collate_fn,
    )

    device = torch.device(_device_str())
    n_epochs = int(de.get("n_epochs", 10))
    lr = float(de.get("lr", 1e-4))
    wd = float(de.get("weight_decay", 0.01))
    fractions: List[float] = [float(x) for x in de.get("train_fractions", [1.0])]
    log_train_every = int(de.get("wandb_log_train_every", 1))
    val_eval_every = int(de.get("val_eval_every", 0))
    train_kw = dict(
        grad_clip=float(de.get("grad_clip", 1.0)),
        label_smoothing=float(de.get("label_smoothing", 0.0)),
        early_stopping_patience=int(de.get("early_stopping_patience", 3)),
        freeze_backbone=bool(de.get("freeze_backbone", False)),
        backbone_lr_scale=float(de.get("backbone_lr_scale", 0.1)),
        use_lr_scheduler=bool(de.get("lr_scheduler", True)),
        lr_scheduler_patience=int(de.get("lr_scheduler_patience", 2)),
        lr_scheduler_factor=float(de.get("lr_scheduler_factor", 0.5)),
        log_train_every=log_train_every,
        val_eval_every=val_eval_every,
    )
    if mode == "scratch" and train_kw["freeze_backbone"]:
        print("[train] freeze_backbone ignored in scratch mode (no pretrained trunk).")
        train_kw["freeze_backbone"] = False

    if mode in ("jepa", "scratch"):
        model = _build_jepa_supervised(cfg, vocab)
        if mode == "jepa" and ckpt:
            _load_jepa_backbone(model, ckpt)
    else:
        model = _build_bert_supervised(cfg, vocab)
        if not ckpt:
            raise SystemExit("bert mode requires a checkpoint (downstream_eval.checkpoint_path or --checkpoint)")
        _load_bert_backbone(model, ckpt)

    seed_i = int(seed) if seed is not None else 0

    run = init_wandb(
        cfg,
        args.config,
        disabled=args.no_wandb,
        run_name=_make_supervised_run_name(cfg, mode),
    )
    if run is not None:
        run.config.update({
            "supervised_mode": mode,
            "supervised_checkpoint": ckpt or "",
            "run_fraction_sweep": do_fraction_sweep,
        })

    if do_fraction_sweep:
        for frac_i, frac in enumerate(fractions):
            print(f"\n=== fraction={frac} ===")
            train_loader = _subset_train_loader(
                train_full, frac, seed_i, batch_size, num_workers, pin_memory
            )
            if mode in ("jepa", "scratch"):
                m = _build_jepa_supervised(cfg, vocab)
                if mode == "jepa" and ckpt:
                    _load_jepa_backbone(m, ckpt)
            else:
                m = _build_bert_supervised(cfg, vocab)
                _load_bert_backbone(m, ckpt)
            log_prefix = f"supervised/frac_{frac:g}"
            frac_steps = n_epochs * max(len(train_loader), 1)
            step_offset = frac_i * (frac_steps + 2)
            val_m, test_m = _train_one_run(
                m,
                train_loader,
                probe_val,
                probe_test,
                device,
                n_epochs,
                lr,
                wd,
                run=run,
                log_prefix=log_prefix,
                step_offset=step_offset,
                **train_kw,
            )
            print(f"  val:  {val_m}")
            print(f"  test: {test_m}")
            if run is not None:
                run.summary[f"{log_prefix}/test_auroc"] = test_m.get("auroc")
                run.summary[f"{log_prefix}/test_aupr"] = test_m.get("aupr")
                # best_val_* set inside _train_one_run
    else:
        val_m, test_m = _train_one_run(
            model,
            train_full,
            probe_val,
            probe_test,
            device,
            n_epochs,
            lr,
            wd,
            run=run,
            **train_kw,
        )
        print(f"\nval:  {val_m}")
        print(f"test: {test_m}")
        if run is not None:
            run.summary["supervised/test_auroc"] = test_m.get("auroc")
            run.summary["supervised/test_aupr"] = test_m.get("aupr")
            # supervised/best_val_epoch and best_val_auroc set in _train_one_run

    if run is not None:
        print(f"[wandb] Run finished: {run.url}")
        run.finish()


if __name__ == "__main__":
    main()
