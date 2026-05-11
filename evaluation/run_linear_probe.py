"""
Command-line entry point for downstream binary classification.

Usage
-----
python evaluation/run_linear_probe.py \\
    --checkpoint /path/to/checkpoints/best.pt \\
    --config     configs/ehr_config.yaml \\
    [--task      hospital_mortality]   # overrides data.labels_task in config
    [--data-dir  /path/to/meds/data]   # overrides data.data_dir in config
    [--epochs    20]
    [--lr        1e-3]
    [--batch-size 64]
    [--dropout   0.1]
    [--no-wandb]
    [--save-probe /path/to/probe.pt]

Label files are resolved from the config as:
    {data.labels_base_dir}/{data.labels_task}/train_labels.parquet
    {data.labels_base_dir}/{data.labels_task}/tuning_labels.parquet
    {data.labels_base_dir}/{data.labels_task}/held_out_labels.parquet   (optional)

Each parquet file must contain columns:
    subject_id | prediction_time | label  (0 or 1)

The script:
1. Loads the YAML config and builds the JEPA model architecture.
2. Loads the checkpoint and wraps the encoder in FrozenEHREncoder.
3. Builds train / val datasets filtered to prediction_time (no leakage).
4. Trains a LinearProbe (only probe weights updated).
5. Evaluates on tuning (each epoch) and held_out (once at the end).
6. Reports final AUC and optionally saves the probe weights.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.collator import MEDSCollator
from data.meds_dataset import MEDSDataset
from data.normalizer import ValueNormalizer
from data.vocab import Vocab
from evaluation.linear_probe import (
    FrozenEHREncoder,
    LinearProbe,
    _eval_probe,
    load_frozen_encoder_from_checkpoint,
    train_linear_probe,
)
from evaluation.downstream_dataset import DownstreamDataset
from main import build_model, _ensure_vocab, _ensure_normalizer


# ---------------------------------------------------------------------------
# Downstream dataset — wraps MEDSDataset with prediction-time filtering
# ---------------------------------------------------------------------------

def _load_labels(parquet_path: str) -> dict:
    """
    Load a single `{split}_labels.parquet` file and return:
        {subject_id: (prediction_time, label)}
    """
    import polars as pl

    df = pl.read_parquet(parquet_path)
    required = {"subject_id", "prediction_time", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Label file must have columns {required}, got {df.columns}")

    result = {}
    for row in df.iter_rows(named=True):
        result[row["subject_id"]] = (row["prediction_time"], int(row["label"]))
    pos = sum(1 for _, lbl in result.values() if lbl == 1)
    print(f"  [labels] {len(result)} subjects  ({pos} positive / {len(result)-pos} negative)"
          f"  ← '{parquet_path}'")
    return result


def _resolve_label_paths(cfg: dict, task_override: str | None) -> tuple[str, str, str | None]:
    """
    Return (train_path, tuning_path, held_out_path_or_None) from config.

    data.labels_base_dir / data.labels_task / {split}_labels.parquet
    """
    base   = cfg["data"]["labels_base_dir"]
    task   = task_override or cfg["data"].get("labels_task", "")
    if not task:
        raise ValueError(
            "Specify a task via --task or set data.labels_task in the config."
        )
    task_dir   = os.path.join(base, task)
    train_path  = os.path.join(task_dir, "train_labels.parquet")
    tuning_path = os.path.join(task_dir, "tuning_labels.parquet")
    heldout_path = os.path.join(task_dir, "held_out_labels.parquet")
    return train_path, tuning_path, heldout_path if os.path.exists(heldout_path) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EHR-JEPA linear probe")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to best.pt / last.pt")
    parser.add_argument("--config",      required=True,
                        help="YAML config (same as pretraining)")
    parser.add_argument("--task",        default=None,
                        help="Label task sub-folder (e.g. icu_mortality). "
                             "Overrides data.labels_task in config.")
    parser.add_argument("--data-dir",    default=None,
                        help="Override data.data_dir from config")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--no-wandb",    action="store_true")
    parser.add_argument("--save-probe",  default=None,
                        help="Path to save trained probe weights")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    # ---- Artefacts ----
    vocab: Vocab | None = None
    if cfg["model"]["embedding_type"] == "learned":
        vocab = _ensure_vocab(cfg)
    else:
        vocab_path = cfg["data"]["vocab_path"]
        vocab = Vocab.load(vocab_path)
        print(f"[vocab] Loaded '{vocab_path}' ({len(vocab)} codes)")

    normalizer: ValueNormalizer | None = None
    if cfg["model"].get("use_value", False):
        normalizer = _ensure_normalizer(cfg)

    # ---- W&B (optional) ----
    run = None
    if not args.no_wandb and cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            run = wandb.init(
                project=cfg["wandb"].get("project", "EHR-JEPA"),
                entity=cfg["wandb"].get("entity"),
                name=f"probe_{os.path.basename(args.checkpoint)}",
                config={
                    "checkpoint": args.checkpoint,
                    "epochs":     args.epochs,
                    "lr":         args.lr,
                    "dropout":    args.dropout,
                },
            )
        except Exception as e:
            print(f"[wandb] Could not initialise: {e}")

    # ---- Build model & load checkpoint ----
    print("[model] Building JEPA model architecture …")
    trainer = build_model(cfg, vocab)
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[checkpoint] Loading from '{args.checkpoint}' …")
    encoder = load_frozen_encoder_from_checkpoint(args.checkpoint, trainer)
    encoder = encoder.to(device)
    print(f"[model] Encoder output dim: {encoder.output_dim}")

    probe = LinearProbe(encoder.output_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"[probe] Trainable parameters: {n_params:,}")

    # ---- Labels ----
    task = args.task or cfg["data"].get("labels_task", "")
    print(f"[labels] Task: '{task}'")
    train_lbl_path, tuning_lbl_path, heldout_lbl_path = _resolve_label_paths(cfg, args.task)

    train_labels = _load_labels(train_lbl_path)
    val_labels   = _load_labels(tuning_lbl_path)
    heldout_labels = _load_labels(heldout_lbl_path) if heldout_lbl_path else {}

    if not train_labels:
        print("[error] No train labels found — aborting.")
        sys.exit(1)

    # ---- Datasets & loaders ----
    max_seq_len = cfg["training"]["max_seq_len"]
    data_dir    = cfg["data"]["data_dir"]

    def _make_loader(split: str, labels: dict, shuffle: bool) -> torch.utils.data.DataLoader | None:
        if not labels:
            return None
        split_data_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_data_dir):
            print(f"[data] MEDS split '{split}' not found at '{split_data_dir}' — skipping.")
            return None
        ds = MEDSDataset(
            data_dir=split_data_dir,
            vocab=vocab,
            normalizer=normalizer,
            task="prediction",
        )
        downstream_ds = DownstreamDataset(ds, labels)
        if len(downstream_ds) == 0:
            print(f"[data] No labelled subjects found in MEDS '{split}' split — skipping.")
            return None
        collator = MEDSCollator(
            pad_idx=vocab.unk_idx if vocab else 0,
            max_len=max_seq_len,
            task="prediction",
        )
        return torch.utils.data.DataLoader(
            downstream_ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )

    train_loader   = _make_loader("train",    train_labels,   shuffle=True)
    val_loader     = _make_loader("tuning",   val_labels,     shuffle=False)
    heldout_loader = _make_loader("held_out", heldout_labels, shuffle=False) if heldout_labels else None

    if train_loader is None:
        print("[error] No training data found — aborting.")
        sys.exit(1)

    print(f"[data] Train subjects: {len(train_loader.dataset)}")
    if val_loader:
        print(f"[data] Val subjects:   {len(val_loader.dataset)}")
    if heldout_loader:
        print(f"[data] Held-out subjects: {len(heldout_loader.dataset)}")

    # ---- W&B callback ----
    def on_epoch_end(epoch: int, metrics: dict) -> None:
        if run is not None:
            run.log({f"probe/{k}": v for k, v in metrics.items()},
                    step=metrics["epoch"])

    # ---- Train ----
    print("\n[probe] Starting linear probe training …")
    history, _ = train_linear_probe(
        encoder=encoder,
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        on_epoch_end=on_epoch_end,
    )

    # ---- Held-out evaluation (if available) ----
    heldout_auc = None
    if heldout_loader is not None:
        print("\n[probe] Evaluating on held-out set …")
        heldout_loss, heldout_m = _eval_probe(
            encoder, probe, heldout_loader,
            torch.nn.BCEWithLogitsLoss(), torch.device(device)
        )
        print(f"  Held-out  auroc={heldout_m['auroc']:.4f}  aupr={heldout_m['aupr']:.4f}"
              f"  recall={heldout_m['recall']:.4f}  precision={heldout_m['precision']:.4f}"
              f"  accuracy={heldout_m['accuracy']:.4f}")
        if run is not None:
            run.log({f"probe/heldout_{k}": v for k, v in heldout_m.items()})
            run.log({"probe/heldout_loss": heldout_loss})

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Linear probe training complete.")
    print(f"  Final train AUROC:  {history['train_auroc'][-1]:.4f}")
    if history["val_auroc"]:
        print(f"  Best  val AUROC:    {max(history['val_auroc']):.4f}")
        print(f"  Final val AUROC:    {history['val_auroc'][-1]:.4f}")
        print(f"  Final val AUPR:     {history['val_aupr'][-1]:.4f}")
    print("=" * 60)

    if args.save_probe:
        torch.save({"probe_state": probe.state_dict(), "history": history}, args.save_probe)
        print(f"[probe] Saved to '{args.save_probe}'")

    if run is not None:
        run.summary["best_val_auroc"]   = max(history["val_auroc"]) if history["val_auroc"] else None
        run.summary["final_train_auroc"] = history["train_auroc"][-1]
        run.finish()


if __name__ == "__main__":
    main()
