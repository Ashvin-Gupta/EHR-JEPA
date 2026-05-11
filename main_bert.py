"""
EHR-BERT: BERT-style MLM pretraining baseline.

Reuses as much of the existing EHR-JEPA infrastructure as possible:
  - Data loading / collation (MEDSDataset, build_loaders utilities)
  - EventEmbedding and EHRTransformerEncoder (same architecture)
  - Downstream linear probe pipeline (LinearProbe, train_linear_probe)
  - DDP helpers, W&B initialisation, config loading, seed setting

What is new:
  - MLMCollator: random token masking (12.5% [MASK], 2.5% random)
  - BERTEHRModel: adds a learnable [CLS] token and an MLM prediction head
  - BERTTrainer:  MLM training loop with inline probe evaluation
  - FrozenBERTEncoder: CLS-based feature extractor for the linear probe

Usage:
    python main_bert.py                              # default config
    python main_bert.py --config configs/bert_config.yaml
    torchrun --standalone --nproc_per_node=4 main_bert.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Reuse utilities from main.py (no code duplication)
# ---------------------------------------------------------------------------
from main import (
    _init_ddp,
    load_config,
    set_seed,
    _print_config,
    _needs_vocab,
    _needs_normalizer,
    _ensure_vocab,
    _ensure_normalizer,
    build_loaders,
    build_probe_loaders,
    init_wandb,
    _run_final_probe_test,
)

# ---------------------------------------------------------------------------
# BERT-specific imports
# ---------------------------------------------------------------------------
from data.mlm_collator import MLMCollator
from data.meds_dataset import MEDSDataset
from data.normalizer import ValueNormalizer
from data.vocab import Vocab
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.bert_trainer import BERTConfig, BERTEHRModel, BERTTrainer


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_bert_model(cfg: dict, vocab: Vocab | None) -> BERTTrainer:
    """Build BERTTrainer from config."""
    m   = cfg["model"]
    t   = cfg.get("transformer", {})
    tr  = cfg.get("training", {})

    d_model    = m.get("d_model", 768)
    emb_type   = m["embedding_type"]
    use_value  = bool(m.get("use_value", False))
    use_time   = bool(m.get("use_time", False))

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

    encoder = EHRTransformerEncoder(TransformerEncoderConfig(
        n_layers=t.get("n_layers", 6),
        d_model=d_model,
        n_heads=t.get("n_heads", 8),
        ffn_dim=t.get("ffn_dim", 1024),
        dropout=t.get("dropout", 0.1),
    ))

    model = BERTEHRModel(
        embedding=embedding,
        encoder=encoder,
        vocab_size=vocab_size,
    )

    bert_cfg = BERTConfig(
        vocab_size=vocab_size,
        lr=tr.get("lr", 1e-4),
        weight_decay=tr.get("weight_decay", 1e-2),
        scheduler=tr.get("scheduler", "cosine_warmup"),
        warmup_ratio=tr.get("warmup_ratio", 0.05),
        min_lr_ratio=tr.get("min_lr_ratio", 0.1),
        grad_clip=tr.get("grad_clip", 0.0),
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 1),
        early_stopping_patience=tr.get("early_stopping_patience", 5),
        early_stopping_metric=tr.get("early_stopping_metric", "val_loss"),
        checkpoint_dir=tr.get("checkpoint_dir", ""),
        n_epochs=tr.get("n_epochs", 10),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    return BERTTrainer(model=model, config=bert_cfg)


# ---------------------------------------------------------------------------
# DataLoader factory — wraps build_loaders with MLM collation
# ---------------------------------------------------------------------------

def build_mlm_loaders(
    cfg: dict,
    vocab: Vocab | None,
    normalizer: ValueNormalizer | None,
    train_sampler: DistributedSampler | None = None,
):
    """
    Build train/val DataLoaders with MLM masking.

    Delegates to build_loaders for the MEDSDataset construction and
    DataLoader kwargs, then swaps in MLMCollator.
    """
    from data.collator import MEDSCollator
    from torch.utils.data import DataLoader

    tr       = cfg.get("training", {})
    data_cfg = cfg["data"]
    mlm_cfg  = cfg.get("mlm", {})

    max_seq_len  = tr.get("max_seq_len", 512)
    batch_size   = tr.get("batch_size", 32)
    num_workers  = tr.get("num_workers", 4)
    pin_memory   = bool(tr.get("pin_memory", True)) and torch.cuda.is_available()
    time_unit    = tr.get("time_unit", "hours")
    cache_dir    = data_cfg.get("cache_dir", None)

    if vocab is None:
        raise RuntimeError("vocab is required.")

    # mask_token_idx is one past the last real vocabulary entry
    mask_token_idx = vocab.vocab_size

    collator = MLMCollator(
        pad_idx=vocab.unk_idx,
        mask_token_idx=mask_token_idx,
        vocab_size=vocab.vocab_size,
        max_len=max_seq_len,
        mask_ratio=mlm_cfg.get("mask_ratio", 0.15),
        mask_token_frac=mlm_cfg.get("mask_token_frac", 12.5 / 15.0),
        random_frac=mlm_cfg.get("random_frac", 2.5 / 15.0),
    )

    def _ds(split: str):
        return MEDSDataset(
            data_dir=data_cfg["data_dir"],
            vocab=vocab,
            split=split,
            task="pretrain",
            max_seq_len=max_seq_len,
            normalizer=normalizer,
            time_unit=time_unit,
            cache_dir=cache_dir,
        )

    loader_kwargs = dict(
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print("[data] Loading train split …")
    train_ds = _ds("train")
    print("[data] Loading tuning split …")
    val_ds   = _ds("tuning")

    if train_sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, no_wandb: bool = False) -> None:
    # 0. DDP
    rank, local_rank, world_size, is_ddp = _init_ddp()
    is_main = (rank == 0)

    cfg = load_config(config_path)

    base_seed = cfg.get("seed", None)
    set_seed(None if base_seed is None else base_seed + rank)

    if is_main:
        _print_config(cfg, config_path)

    # 2. Vocab
    vocab: Vocab | None = None
    if _needs_vocab(cfg):
        vocab = _ensure_vocab(cfg)
    else:
        if is_main:
            print("[vocab] Skipped — text_based mode.")
        vocab_path = cfg["data"]["vocab_path"]
        if os.path.exists(vocab_path):
            vocab = Vocab.load(vocab_path)
            if is_main:
                print(f"[vocab] Loaded '{vocab_path}' ({len(vocab)} codes)")
        else:
            if is_main:
                print(f"[vocab] ERROR: '{vocab_path}' not found.")
            sys.exit(1)

    # 3. Normalizer
    normalizer: ValueNormalizer | None = None
    if _needs_normalizer(cfg):
        normalizer = _ensure_normalizer(cfg)
    elif is_main:
        print("[normalizer] Skipped — use_value=False.")
    if is_main:
        print()

    # 4. W&B (rank 0 only)
    # Build a BERT-specific run name that clearly identifies it as a baseline.
    m   = cfg["model"]
    _parts = [
        "bert",
        m["embedding_type"],
        "mlm",
    ]
    if m.get("use_value"):
        _parts.append("value")
    if m.get("use_time"):
        _parts.append("time")
    bert_run_name = "__".join(_parts)
    run = init_wandb(cfg, config_path, disabled=no_wandb, run_name=bert_run_name) if is_main else None

    # 5. Model
    if is_main:
        print("[model] Building BERT model …")
    trainer   = build_bert_model(cfg, vocab)
    n_params  = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    if is_main:
        print(f"[model] Device:               {trainer.config.device}")
        if is_ddp:
            print(f"[model] DDP world size:       {world_size}")
        print(f"[model] Trainable parameters: {n_params:,}")
        if run is not None:
            run.summary["n_params"] = n_params
        print()

    # 5b. DDP wrapping
    device = torch.device(trainer.config.device)
    ddp_trainer: DDP | None = None
    if is_ddp:
        trainer = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer)
        trainer.to(device)
        ddp_trainer = DDP(
            trainer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,   # all params used in MLM forward
        )

    # 6. Data
    if is_main:
        print("[data] Building MLM DataLoaders …")
    train_sampler: DistributedSampler | None = None
    if is_ddp:
        _tr = cfg.get("training", {})
        _dc = cfg["data"]
        _train_ds = MEDSDataset(
            data_dir=_dc["data_dir"], vocab=vocab, split="train",
            task="pretrain", max_seq_len=_tr.get("max_seq_len", 512),
            normalizer=normalizer, time_unit=_tr.get("time_unit", "hours"),
            cache_dir=_dc.get("cache_dir"),
        )
        train_sampler = DistributedSampler(
            _train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
    train_loader, val_loader = build_mlm_loaders(cfg, vocab, normalizer, train_sampler)
    if is_main:
        print(f"[data] Train: {len(train_loader.dataset):,} subjects  |  {len(train_loader)} batches")
        print(f"[data] Val:   {len(val_loader.dataset):,} subjects  |  {len(val_loader)} batches")
        print()

    # 6b. Probe loaders (rank 0 only — probe runs serially)
    probe_train_loader, probe_val_loader, probe_test_loader = (
        build_probe_loaders(cfg, vocab, normalizer) if is_main
        else (None, None, None)
    )
    probe_task = cfg["data"].get("labels_task", "downstream")
    if is_main and probe_train_loader is not None:
        print(f"[probe] Inline probe enabled — task: '{probe_task}'")
        print()

    # 7. Optimiser
    optimizer = optim.AdamW(
        trainer.parameters(),
        lr=trainer.config.lr,
        weight_decay=trainer.config.weight_decay,
    )

    # 8. W&B callbacks
    def on_batch_end(epoch: int, global_step: int, metrics: dict) -> None:
        if run is not None:
            run.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)

    def on_epoch_end(epoch: int, metrics: dict) -> None:
        if run is None:
            return
        step    = int(metrics.get("global_step", 0))
        payload: dict = {}
        val_keys = {"val_loss"}
        for k, v in metrics.items():
            if k in val_keys:
                payload[f"val/{k}"] = v
        for k, v in metrics.items():
            if k.startswith("probe_"):
                metric_name = k[len("probe_"):]
                payload[f"downstream_task/{probe_task}/{metric_name}"] = v
        if payload:
            run.log(payload, step=step)

    # 9. Train
    if is_main:
        print("[train] Starting BERT training loop …")
    ds_cfg  = cfg.get("downstream", {})
    history = trainer.train_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        on_epoch_end=on_epoch_end,
        on_batch_end=on_batch_end,
        probe_train_loader=probe_train_loader,
        probe_val_loader=probe_val_loader,
        probe_n_epochs=ds_cfg.get("probe_epochs", 15),
        probe_lr=ds_cfg.get("probe_lr", 1e-3),
        probe_dropout=ds_cfg.get("probe_dropout", 0.1),
        probe_interval=ds_cfg.get("probe_interval", 1),
        ddp_module=ddp_trainer,
        is_main_process=is_main,
        train_sampler=train_sampler,
    )

    # 10. Summary
    if is_main:
        print()
        print("=" * 62)
        print("  BERT pretraining complete.")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        if history["val_loss"]:
            print(f"  Final val loss:   {history['val_loss'][-1]:.4f}")
        print("=" * 62)

        if run is not None:
            run.summary["final_train_loss"] = history["train_loss"][-1]
            if history["val_loss"]:
                run.summary["final_val_loss"] = history["val_loss"][-1]

        # 11. Final test evaluation
        _run_final_probe_test(
            cfg=cfg,
            trainer=trainer,
            probe_train_loader=probe_train_loader,
            probe_test_loader=probe_test_loader,
            probe_task=probe_task,
            run=run,
        )

        if run is not None:
            run.finish()
            print(f"[wandb] Run finished: {run.url}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EHR-BERT MLM pretraining")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "bert_config.yaml"),
        help="Path to YAML config (default: configs/bert_config.yaml)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    args = parser.parse_args()
    main(args.config, no_wandb=args.no_wandb)
