"""
EHR-JEPA main training entry point.

Usage:
    python main.py                        # uses configs/ehr_config.yaml
    python main.py --config path/to.yaml  # custom config
    python main.py --no-wandb             # disable W&B logging for this run

What this script does:
  1. Load the YAML config and print it in full.
  2. Decide what pre-processing is needed:
       - Vocab:      only when embedding_type == "learned"
       - Normalizer: only when model.use_value == True
  3. For each artefact, check if it already exists on disk and skip if so.
  4. Initialise a W&B run (unless --no-wandb or wandb.enabled: false).
  5. Build all model components and run the training loop.
  6. Log per-batch and per-epoch metrics to W&B.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys

import numpy as np
import torch
import yaml
import torch.optim as optim
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.collator import MEDSCollator
from data.meds_dataset import MEDSDataset
from data.normalizer import ValueNormalizer
from data.vocab import build_vocab, Vocab
from loss.covariance_reg import CovarianceRegularizationLoss
from masking.span_masking import SpanMasker
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.predictor import Predictor, TemporalSpanPrompt
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.trainer import JEPATrainer, TrainerConfig


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int | None) -> None:
    """Fix all random seeds for reproducibility, or do nothing if seed is None."""
    if seed is None:
        print("[seed] No seed set — run is non-deterministic.")
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops fully deterministic at a small speed cost
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] All random seeds fixed to {seed}.")


def _print_config(cfg: dict, config_path: str) -> None:
    """Pretty-print the full config to stdout."""
    width = 62
    print("=" * width)
    print(f"  CONFIG  ({config_path})")
    print("=" * width)
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False).rstrip())
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Artefact decision logic
# ---------------------------------------------------------------------------

def _needs_vocab(cfg: dict) -> bool:
    """Vocab (nn.Embedding table) is only needed for learned embeddings."""
    return cfg["model"]["embedding_type"] == "learned"


def _needs_normalizer(cfg: dict) -> bool:
    """Normalizer is only needed when numeric values feed the MLP."""
    return bool(cfg["model"].get("use_value", False))


def _ensure_vocab(cfg: dict) -> Vocab:
    vocab_path = cfg["data"]["vocab_path"]
    if os.path.exists(vocab_path):
        print(f"[vocab] Loading from '{vocab_path}'")
        return Vocab.load(vocab_path)
    print(f"[vocab] Not found — building from training data …")
    top_k = cfg["model"].get("vocab_size", 5000)
    vocab = build_vocab(cfg["data"]["data_dir"], embedding_type="learned", top_k=top_k)
    vocab.save(vocab_path)
    print(f"[vocab] Built {len(vocab)} entries  →  saved to '{vocab_path}'")
    return vocab


def _ensure_normalizer(cfg: dict) -> ValueNormalizer:
    stats_path = cfg.get("normalizer", {}).get("stats_path", "normalizer_stats.json")
    if os.path.exists(stats_path):
        print(f"[normalizer] Loading from '{stats_path}'")
        return ValueNormalizer.load(stats_path)
    print(f"[normalizer] Not found — fitting on training data …")
    norm = ValueNormalizer()
    norm.fit(cfg["data"]["data_dir"], split="train")
    norm.save(stats_path)
    print(f"[normalizer] Fitted {len(norm)} codes  →  saved to '{stats_path}'")
    return norm


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict, vocab: Vocab | None) -> JEPATrainer:
    m   = cfg["model"]
    t   = cfg.get("transformer", {})
    lp  = cfg.get("latent_pooling", {})
    p   = cfg.get("predictor", {})
    lc  = cfg.get("loss", {})
    mk  = cfg.get("masking", {})
    tr  = cfg.get("training", {})

    d_model:       int  = m["d_model"]
    use_value:     bool = bool(m.get("use_value", False))
    use_time:      bool = bool(m.get("use_time", False))
    emb_type:      str  = m["embedding_type"]
    n_heads:       int  = t.get("n_heads", 8)
    n_latents:     int  = lp.get("n_latents", 16)
    use_perceiver: bool = bool(p.get("use_perceiver", True))

    vocab_size = vocab.vocab_size if vocab is not None else m.get("vocab_size", 5001)
    unk_idx    = vocab.unk_idx    if vocab is not None else vocab_size - 1

    embedding = EventEmbedding(EmbeddingConfig(
        embedding_type=emb_type,
        vocab_size=vocab_size,
        d_model=d_model,
        code_embeddings_path=m.get("code_embeddings_path"),
        encoder_hidden_dim=m.get("encoder_hidden_dim", 768),
        unk_idx=unk_idx,
        use_value=use_value,
        use_time=use_time,
    ))

    # Single shared encoder — used for both target and context pathways
    encoder = EHRTransformerEncoder(TransformerEncoderConfig(
        n_layers=t.get("n_layers", 6),
        d_model=d_model,
        n_heads=n_heads,
        ffn_dim=t.get("ffn_dim", 1024),
        dropout=t.get("dropout", 0.1),
    ))

    prompt   = TemporalSpanPrompt(d_model)
    cov_loss = CovarianceRegularizationLoss(d_model, proj_dim=lc.get("cov_proj_dim", 64))
    masker   = SpanMasker(
        mask_ratio=mk.get("mask_ratio", 0.30),
        default_num_spans=mk.get("default_num_spans", 4),
        min_span_length=mk.get("min_span_length", 15),
    )

    # Branch A: Perceiver poolers + latent predictor
    context_pooler: LatentCrossAttentionPool | None = None
    target_pooler:  LatentCrossAttentionPool | None = None
    if use_perceiver:
        context_pooler = LatentCrossAttentionPool(
            d_model, n_latents=n_latents, n_heads=n_heads
        )
        target_pooler = LatentCrossAttentionPool(
            d_model, n_latents=n_latents, n_heads=n_heads
        )

    # Latent predictor (Branch A) — shallow transformer on latent tokens
    predictor = Predictor(
        d_model, n_heads=p.get("n_heads", 8), n_layers=p.get("n_layers", 2)
    )

    # Token predictor (Branch B) — shallow transformer on raw tokens
    token_predictor = EHRTransformerEncoder(TransformerEncoderConfig(
        n_layers=p.get("n_layers", 2),
        d_model=d_model,
        n_heads=p.get("n_heads", 8),
        ffn_dim=d_model * 4,
        dropout=t.get("dropout", 0.1),
    ))

    trainer_cfg = TrainerConfig(
        use_perceiver=use_perceiver,
        min_span_for_perceiver=p.get("min_span_for_perceiver", 15),
        lambda_cov=lc.get("lambda_cov", 0.1),
        lr=tr.get("lr", 1e-4),
        weight_decay=tr.get("weight_decay", 1e-2),
        grad_clip=tr.get("grad_clip", 1.0),
        scheduler=tr.get("scheduler", "cosine_warmup"),
        warmup_ratio=tr.get("warmup_ratio", 0.05),
        min_lr_ratio=tr.get("min_lr_ratio", 0.1),
        early_stopping_patience=tr.get("early_stopping_patience", 5),
        early_stopping_metric=tr.get("early_stopping_metric", "val_loss"),
        n_epochs=tr.get("n_epochs", 10),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    return JEPATrainer(
        embedding=embedding,
        encoder=encoder,
        prompt=prompt,
        predictor=predictor,
        token_predictor=token_predictor,
        context_pooler=context_pooler,
        target_pooler=target_pooler,
        cov_loss=cov_loss,
        masker=masker,
        config=trainer_cfg,
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_loaders(
    cfg: dict,
    vocab: Vocab | None,
    normalizer: ValueNormalizer | None,
) -> tuple[DataLoader, DataLoader | None]:
    tr        = cfg.get("training", {})
    data_cfg  = cfg["data"]

    max_seq_len: int        = tr.get("max_seq_len", 512)
    batch_size: int         = tr.get("batch_size", 32)
    time_unit: str          = tr.get("time_unit", "hours")
    task: str               = tr.get("task", "pretrain")
    cache_dir: str | None   = data_cfg.get("cache_dir", None)
    num_workers: int        = tr.get("num_workers", 4)
    pin_memory: bool        = bool(tr.get("pin_memory", True)) and torch.cuda.is_available()

    if vocab is None:
        raise RuntimeError("vocab is required to encode sequences.")

    if cache_dir:
        print(f"[data] Sequence cache dir: {cache_dir}")

    def _ds(split: str) -> MEDSDataset:
        return MEDSDataset(
            data_dir=data_cfg["data_dir"],
            vocab=vocab,
            split=split,
            task=task,
            max_seq_len=max_seq_len,
            aces_label_path=data_cfg.get("aces_label_path"),
            normalizer=normalizer,
            time_unit=time_unit,
            cache_dir=cache_dir,
        )

    # Build the span masker and hand it to the collator so masking runs inside
    # the DataLoader worker processes (parallel to GPU computation).
    # Only used for pretrain; prediction tasks don't need masking in the loader.
    collator_masker: SpanMasker | None = None
    if task == "pretrain":
        mask_cfg = cfg.get("masking", {})
        collator_masker = SpanMasker(
            mask_ratio=mask_cfg.get("mask_ratio", 0.30),
            default_num_spans=mask_cfg.get("default_num_spans", 4),
            min_span_length=mask_cfg.get("min_span_length", 15),
        )

    collator = MEDSCollator(
        pad_idx=vocab.unk_idx,
        max_len=max_seq_len,
        task=task,
        masker=collator_masker,
    )

    print("[data] Loading train split …")
    train_ds = _ds("train")
    print("[data] Loading tuning split …")
    val_ds   = _ds("tuning")

    # num_workers > 0: worker processes call __getitem__ in parallel while the
    # GPU is computing the previous batch, eliminating the CPU↔GPU idle gap.
    # pin_memory: tensors are allocated in page-locked RAM for faster DMA transfer.
    # persistent_workers: keep workers alive between epochs (avoids fork overhead).
    # prefetch_factor: each worker pre-fetches this many batches ahead.
    loader_kwargs = dict(
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    masking_loc = "worker (fast)" if collator_masker is not None else "main thread"
    print(f"[data] num_workers={num_workers}  pin_memory={pin_memory}  masking={masking_loc}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _make_run_name(cfg: dict) -> str:
    """
    Compose a human-readable run name from the most important config knobs.
    Format: {task}__{emb_type}__perceiver|token[__value][__time]
    Example: pretrain__learned__perceiver__value
    """
    branch = "perceiver" if cfg.get("predictor", {}).get("use_perceiver", True) else "token"
    parts = [
        cfg["training"].get("task", "pretrain"),
        cfg["model"]["embedding_type"],
        branch,
    ]
    if cfg["model"].get("use_value"):
        parts.append("value")
    if cfg["model"].get("use_time"):
        parts.append("time")
    return "__".join(parts)


def _flatten_config(cfg: dict, prefix: str = "") -> dict:
    """Flatten nested dict for wandb.config (wandb accepts nested dicts natively,
    but flattening makes it easier to filter on the dashboard)."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_config(v, prefix=key))
        else:
            out[key] = v
    return out


def init_wandb(cfg: dict, config_path: str, disabled: bool = False):
    """
    Initialise a W&B run.  Returns the run object, or a no-op stub if
    wandb is disabled or not installed.
    """
    wb_cfg = cfg.get("wandb", {})

    if disabled or not wb_cfg.get("enabled", True):
        print("[wandb] Logging disabled.")
        return None

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed — skipping. Run: pip install wandb")
        return None

    run_name = _make_run_name(cfg)
    project  = wb_cfg.get("project", "EHR-JEPA")
    entity   = wb_cfg.get("entity", None)

    config_yaml_str = yaml.dump(cfg, default_flow_style=False, sort_keys=False)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=_flatten_config(cfg),
        notes=f"```yaml\n{config_yaml_str}```",
        resume="allow",
    )
    # Also log the raw YAML as a plain-text artifact so it is always
    # retrievable from the Files tab regardless of W&B plan limits.
    config_artifact = wandb.Artifact("config", type="config")
    config_artifact.add(wandb.Table(
        columns=["yaml"],
        data=[[config_yaml_str]],
    ), name="config_yaml")
    run.log_artifact(config_artifact)

    print(f"[wandb] Run initialised: {run.url}")
    print(f"[wandb] Project: {project}  |  Name: {run_name}")
    print()
    return run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, no_wandb: bool = False) -> None:
    cfg = load_config(config_path)

    # 0. Seed
    set_seed(cfg.get("seed", None))

    # 1. Print full config
    _print_config(cfg, config_path)

    # 2. Vocab
    vocab: Vocab | None = None
    if _needs_vocab(cfg):
        vocab = _ensure_vocab(cfg)
    else:
        print("[vocab] Skipped — text_based mode uses pre-computed embeddings.")
        vocab_path = cfg["data"]["vocab_path"]
        if os.path.exists(vocab_path):
            vocab = Vocab.load(vocab_path)
            print(f"[vocab] Loaded text_based vocab '{vocab_path}' ({len(vocab)} codes)")
        else:
            print(f"[vocab] ERROR: '{vocab_path}' not found — run encode_text_embeddings.py first.")
            sys.exit(1)

    # 3. Normalizer
    normalizer: ValueNormalizer | None = None
    if _needs_normalizer(cfg):
        normalizer = _ensure_normalizer(cfg)
    else:
        print("[normalizer] Skipped — use_value=False.")
    print()

    # 4. W&B
    run = init_wandb(cfg, config_path, disabled=no_wandb)

    # 5. Model
    print("[model] Building model …")
    trainer    = build_model(cfg, vocab)
    device_str = trainer.config.device
    n_params   = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    print(f"[model] Device:               {device_str}")
    print(f"[model] Trainable parameters: {n_params:,}")
    if run is not None:
        run.summary["n_params"] = n_params
    print()

    # 6. Data
    print("[data] Building DataLoaders …")
    train_loader, val_loader = build_loaders(cfg, vocab, normalizer)
    print(f"[data] Train: {len(train_loader.dataset):,} subjects  |  {len(train_loader)} batches")
    print(f"[data] Val:   {len(val_loader.dataset):,} subjects  |  {len(val_loader)} batches")
    print()

    # 7. Optimiser — built here so wandb can watch it; trainer uses same settings
    optimizer = optim.AdamW(
        trainer.parameters(),
        lr=trainer.config.lr,
        weight_decay=trainer.config.weight_decay,
    )

    # 8. W&B callbacks
    # All metrics use global_step as x-axis so batch and epoch charts share
    # the same timeline on the W&B dashboard.
    #
    # Panel layout the user can configure in W&B:
    #   "Loss Components"         — train/loss_*, val/loss_total
    #   "Representation Health"   — train/std_dev_embeddings, val/rank_me
    #   "Optimization & Hardware" — train/learning_rate, train/grad_norm,
    #                               train/samples_per_second
    #   "Medical Context"         — train/mask_ratio, train/avg_seq_length,
    #                               val/unique_codes_seen

    def on_batch_end(epoch: int, global_step: int, metrics: dict) -> None:
        if run is not None:
            run.log({f"train/{k}": v for k, v in metrics.items()},
                    step=global_step)

    def on_epoch_end(epoch: int, metrics: dict) -> None:
        if run is not None:
            step = int(metrics.get("global_step", 0))
            # Only val metrics are logged at epoch boundaries; all train
            # metrics are already captured per-batch above.
            val_keys = {"val_loss", "std_dev_embeddings", "rank_me"}
            payload: dict = {}
            for k, v in metrics.items():
                if k not in val_keys:
                    continue
                key = k.replace("val_", "val/", 1) if k.startswith("val_") else f"val/{k}"
                payload[key] = v
            if payload:
                run.log(payload, step=step)

    # 9. Train
    print("[train] Starting training loop …")
    history = trainer.train_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        on_epoch_end=on_epoch_end,
        on_batch_end=on_batch_end,
    )

    # 10. Summary
    print()
    print("=" * 62)
    print("  Training complete.")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")
    print("=" * 62)

    if run is not None:
        run.summary["final_train_loss"] = history["train_loss"][-1]
        if history["val_loss"]:
            run.summary["final_val_loss"] = history["val_loss"][-1]
        run.finish()
        print(f"[wandb] Run finished: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EHR-JEPA pretraining")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "ehr_config.yaml"),
        help="Path to YAML config (default: configs/ehr_config.yaml)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging for this run",
    )
    args = parser.parse_args()
    main(args.config, no_wandb=args.no_wandb)
