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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.normalizer import ValueNormalizer
from data.vocab import Vocab
from evaluation.bert_supervised import BERTSupervisedClassifier
from evaluation.linear_probe import _compute_all_metrics
from evaluation.supervised_perceiver import SupervisedPerceiverClassifier
from main import (
    _ensure_normalizer,
    _ensure_vocab,
    _needs_normalizer,
    build_probe_loaders,
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


def _build_jepa_supervised(cfg: dict, vocab: Vocab) -> SupervisedPerceiverClassifier:
    m = cfg["model"]
    t = cfg.get("transformer", {})
    lp = cfg.get("latent_pooling", {})
    p = cfg.get("predictor", {})
    if not bool(p.get("use_perceiver", True)):
        raise ValueError("Supervised perceiver evaluation requires predictor.use_perceiver: true")

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
    pooler = LatentCrossAttentionPool(d_model, n_latents=n_latents, n_heads=n_heads)

    de = cfg.get("downstream_eval", {})
    head_type = str(de.get("head_type", "linear"))
    head_dropout = float(de.get("head_dropout", 0.1))
    if head_type not in ("linear", "mlp"):
        raise ValueError("downstream_eval.head_type must be 'linear' or 'mlp'")

    return SupervisedPerceiverClassifier(
        embedding=embedding,
        encoder=encoder,
        pooler=pooler,
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


def _load_jepa_backbone(model: SupervisedPerceiverClassifier, path: Optional[str]) -> None:
    if not path:
        return
    sd = load_jepa_backbone_state_dict(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Pooler + head stay randomly initialised
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
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    for _epoch in range(n_epochs):
        model.train()
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
            loss = crit(logits, labels)
            loss.backward()
            opt.step()

    val_metrics: Dict[str, float] = {}
    if val_loader is not None:
        val_metrics = _eval_loader(model, val_loader, device, crit)

    test_metrics: Dict[str, float] = {}
    if test_loader is not None:
        test_metrics = _eval_loader(model, test_loader, device, crit)
    return val_metrics, test_metrics


@torch.no_grad()
def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    crit: nn.Module,
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
        loss = crit(logits, labels)
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

    if do_fraction_sweep:
        for frac in fractions:
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
            val_m, test_m = _train_one_run(
                m, train_loader, probe_val, probe_test, device, n_epochs, lr, wd
            )
            print(f"  val:  {val_m}")
            print(f"  test: {test_m}")
    else:
        val_m, test_m = _train_one_run(
            model, train_full, probe_val, probe_test, device, n_epochs, lr, wd
        )
        print(f"\nval:  {val_m}")
        print(f"test: {test_m}")


if __name__ == "__main__":
    main()
