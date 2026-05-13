"""
BERT-style EHR pretraining via Masked Language Modelling.

Architecture
------------
  EventEmbedding  →  [CLS prepended]  →  EHRTransformerEncoder  →  MLM head
                                                   ↓
                                           CLS token output  (for downstream probe)

Forward returns: (mlm_loss, cls_embedding)
    mlm_loss      — CrossEntropyLoss over masked positions (scalar)
    cls_embedding — FloatTensor (B, d_model)  from the [CLS] position

Training signal
---------------
Only masked/randomly-replaced positions contribute to the MLM loss
(mlm_labels == -100 at unmasked positions, which cross-entropy ignores).

CLS token
---------
A learnable nn.Parameter (shape [d_model]) is broadcast over the batch and
prepended to the embedded sequence.  It is assigned RoPE position ID 0; all
real event tokens are shifted to positions 1 … L.  This keeps RoPE meaningful
without reserving a vocabulary slot.

Inline linear probe
-------------------
After each pretraining epoch a fresh LinearProbe is trained on top of the
frozen CLS embedding and evaluated on the downstream task.  This mirrors the
inline evaluation in JEPATrainer and uses the same helper functions from
evaluation.linear_probe.

Checkpointing
-------------
Saves best.pt  (by val MLM loss or train loss when no val set)
      last.pt  (always, end of each epoch)
      probe_best.pt  (by probe val AUROC, if probe loaders are provided)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import gc
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.event_embedding import EventEmbedding
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig

from training.checkpoint_utils import save_bert_split_checkpoints


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BERTConfig:
    # Model
    vocab_size: int = 5001          # real vocab entries (mask_token_idx = vocab_size)

    # Optimiser
    lr: float = 1e-4
    weight_decay: float = 1e-2

    # LR scheduler (same options as JEPATrainer)
    scheduler: str = "cosine_warmup"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    # Gradient clipping (0 = disabled)
    grad_clip: float = 0.0

    # Gradient accumulation: optimizer.step() every N batches.
    # Effective batch = batch_size * gradient_accumulation_steps
    gradient_accumulation_steps: int = 1

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"

    # Checkpointing
    checkpoint_dir: str = ""

    # Training
    n_epochs: int = 10
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BERTEHRModel(nn.Module):
    """
    BERT-style EHR encoder.

    Parameters
    ----------
    embedding:
        Shared EventEmbedding (same config as JEPA baseline for fair comparison).
    encoder:
        EHRTransformerEncoder.
    vocab_size:
        Number of real vocabulary entries.  The MLM head maps d_model → vocab_size.
    """

    def __init__(
        self,
        embedding: EventEmbedding,
        encoder: EHRTransformerEncoder,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.embedding      = embedding
        self.encoder        = encoder
        self.vocab_size     = vocab_size
        # mask_token_idx = vocab_size (one past the last real token).
        # This sentinel is set in input_codes by MLMCollator to mark masked
        # positions.  It is out-of-range for the embedding table, so we must
        # intercept it before the lookup and replace the position with the
        # learnable mask_embedding below.
        self.mask_token_idx = vocab_size
        d_model             = encoder.config.d_model

        # Learnable [MASK] embedding — replaces the masked position vectors
        # after the embedding lookup so the model receives a true "unknown"
        # signal at those positions.
        self.mask_embedding = nn.Parameter(torch.randn(d_model) * 0.02)

        # Learnable [CLS] token — broadcast over batch at forward time
        self.cls_token  = nn.Parameter(torch.randn(d_model) * 0.02)

        # MLM prediction head: token embeddings → vocabulary logits
        self.mlm_head   = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def _encode_with_cls(
        self,
        input_codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Embed tokens, prepend [CLS], run encoder. Returns (h, L_event) where
        h is (B, L+1, d) and L_event is the original event length (before CLS).
        """
        B, L = input_codes.shape

        is_mask = (input_codes == self.mask_token_idx)
        safe_codes = input_codes.masked_fill(is_mask, 0)

        x = self.embedding(
            safe_codes,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask.float() if value_mask is not None else None,
        )

        if is_mask.any():
            mask_vec = self.mask_embedding.view(1, 1, -1).expand(B, L, -1)
            x = torch.where(is_mask.unsqueeze(-1), mask_vec, x)

        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = attention_mask.new_ones(B, 1)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)

        pos_ids = torch.arange(L + 1, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.encoder(x, attention_mask=full_mask, position_ids=pos_ids)
        return h, L

    def encode_cls_embedding(
        self,
        input_codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CLS vector (B, d_model) for supervised fine-tuning (no MLM loss)."""
        h, _L = self._encode_with_cls(
            input_codes, attention_mask, values, z_scores, delta_times, value_mask
        )
        return h[:, 0, :]

    @property
    def output_dim(self) -> int:
        return self.encoder.config.d_model

    def forward(
        self,
        input_codes: torch.Tensor,
        attention_mask: torch.Tensor,
        mlm_labels: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_codes:    LongTensor  (B, L) — masked codes (from MLMCollator)
        attention_mask: LongTensor  (B, L) — 1=real, 0=pad
        mlm_labels:     LongTensor  (B, L) — original codes at masked positions,
                                              -100 elsewhere
        values / z_scores / delta_times / value_mask: optional FloatTensors (B, L)

        Returns
        -------
        (mlm_loss, cls_embedding)
            mlm_loss      — scalar CrossEntropyLoss over masked positions
            cls_embedding — FloatTensor (B, d_model)
        """
        h, _L = self._encode_with_cls(
            input_codes, attention_mask, values, z_scores, delta_times, value_mask
        )
        cls_embedding = h[:, 0, :]

        token_h = h[:, 1:, :]
        logits = self.mlm_head(token_h)
        mlm_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            mlm_labels.view(-1),
            ignore_index=-100,
        )

        return mlm_loss, cls_embedding


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BERTTrainer(nn.Module):
    """
    Thin container around BERTEHRModel with a train_loop that mirrors the
    interface of JEPATrainer.train_loop for easy drop-in comparison.
    """

    def __init__(self, model: BERTEHRModel, config: BERTConfig) -> None:
        super().__init__()
        self.model  = model
        self.config = config

    # Convenience passthrough so DDP wrapping works naturally
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ------------------------------------------------------------------
    # LR scheduler (copied verbatim from JEPATrainer)
    # ------------------------------------------------------------------

    def _build_scheduler(self, optimizer, total_steps: int):
        cfg = self.config
        if cfg.scheduler == "none":
            return None
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
        min_lr = cfg.lr * cfg.min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine   = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
            return min_lr / cfg.lr + (1 - min_lr / cfg.lr) * cosine

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Inline probe (mirrors JEPATrainer._run_inline_probe)
    # ------------------------------------------------------------------

    def _run_inline_probe(
        self,
        probe_train_loader: DataLoader,
        probe_val_loader: Optional[DataLoader],
        n_epochs: int,
        lr: float,
        dropout: float,
        device: torch.device,
    ) -> Dict[str, float]:
        from evaluation.linear_probe import LinearProbe, train_linear_probe
        from evaluation.frozen_bert_encoder import FrozenBERTEncoder

        encoder = FrozenBERTEncoder(self.model).to(device)
        probe   = LinearProbe(encoder.output_dim, dropout=dropout).to(device)

        history, final_val = train_linear_probe(
            encoder=encoder,
            probe=probe,
            train_loader=probe_train_loader,
            val_loader=probe_val_loader,
            n_epochs=n_epochs,
            lr=lr,
            device=str(device),
            verbose=True,
        )

        train_metrics: Dict[str, float] = {}
        for key in ("train_loss", "train_auroc", "train_aupr",
                    "train_recall", "train_precision", "train_accuracy"):
            vals = history.get(key, [])
            if vals:
                train_metrics[key] = vals[-1]

        del encoder, probe
        gc.collect()

        combined: Dict[str, float] = {f"val_{k}": v for k, v in final_val.items()}
        combined.update(train_metrics)
        return combined

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train_loop(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
        on_batch_end: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        # ---- Inline probe ----
        probe_train_loader: Optional[DataLoader] = None,
        probe_val_loader:   Optional[DataLoader] = None,
        probe_n_epochs:     int   = 15,
        probe_lr:           float = 1e-3,
        probe_dropout:      float = 0.1,
        probe_interval:     int   = 1,   # run probe every N epochs; always runs on epoch 1
        inline_probe_during_pretrain: bool = True,
        # ---- DDP ----
        ddp_module: Optional[nn.Module] = None,
        is_main_process: bool = True,
        train_sampler = None,
    ) -> Dict[str, List[float]]:
        import time as _time

        cfg      = self.config
        _forward = ddp_module if ddp_module is not None else self

        if optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

        if ddp_module is not None:
            # Model was already placed on the correct per-GPU device and wrapped
            # by DDP before train_loop was called. Infer the device from the
            # DDP module so all input tensors go to the right GPU.
            device = next(ddp_module.parameters()).device
        else:
            device = torch.device(cfg.device)
            self.to(device)

        total_steps = cfg.n_epochs * len(train_loader)
        scheduler   = self._build_scheduler(optimizer, total_steps)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}

        best_metric      = float("inf")
        best_ckpt_metric = float("inf")
        best_probe_auroc = -float("inf")
        patience_left    = cfg.early_stopping_patience
        stopped_early    = False

        ckpt_dir = cfg.checkpoint_dir.strip() if cfg.checkpoint_dir else ""
        if ckpt_dir and is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"[train] Checkpoints: {ckpt_dir}")

        if is_main_process:
            print(f"[train] Scheduler:   {cfg.scheduler}")
            print(f"[train] Grad clip:   {cfg.grad_clip if cfg.grad_clip > 0 else 'disabled'}")
            print(f"[train] Grad accum:  {cfg.gradient_accumulation_steps} steps")
            print()

        global_step = 0
        optimizer.zero_grad()           # start with clean gradients
        for epoch in range(cfg.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.train()
            if ddp_module is not None:
                ddp_module.train()

            epoch_loss = 0.0
            n_batches  = 0

            for batch in train_loader:
                t0 = _time.perf_counter()

                input_codes = batch["input_codes"].to(device, non_blocking=True)
                attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
                mlm_labels  = batch["mlm_labels"].to(device, non_blocking=True)
                values      = batch["values"].to(device, non_blocking=True)      if "values"      in batch else None
                z_scores    = batch["z_scores"].to(device, non_blocking=True)    if "z_scores"    in batch else None
                delta_times = batch["delta_times"].to(device, non_blocking=True) if "delta_times" in batch else None
                value_mask  = batch["value_mask"].to(device, non_blocking=True)  if "value_mask"  in batch else None

                # Scale loss for gradient accumulation so effective gradients
                # are normalised to a single-step update.
                accum = max(1, cfg.gradient_accumulation_steps)
                loss, _ = _forward(
                    input_codes, attn_mask, mlm_labels,
                    values, z_scores, delta_times, value_mask,
                )
                (loss / accum).backward()

                batch_loss  = loss.item()
                epoch_loss += batch_loss
                n_batches  += 1

                if n_batches % accum == 0:
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                elapsed = _time.perf_counter() - t0

                # Only log on actual optimizer steps
                if is_main_process and on_batch_end is not None and n_batches % accum == 0:
                    total_batches  = len(train_loader)
                    epoch_progress = epoch + n_batches / max(total_batches, 1)
                    current_lr     = optimizer.param_groups[0]["lr"]
                    on_batch_end(epoch, global_step, {
                        "epoch":              epoch_progress,
                        "loss_mlm":           batch_loss,
                        "learning_rate":      current_lr,
                        "samples_per_second": input_codes.shape[0] / max(elapsed, 1e-6),
                    })

            avg_train  = epoch_loss / max(n_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train)
            history["lr"].append(current_lr)

            if is_main_process:
                epoch_metrics: Dict[str, float] = {"global_step": global_step}

                # Validation
                val_line = ""
                if val_loader is not None:
                    print("  [val] Running validation … ", end="", flush=True)
                    avg_val = self._eval_epoch(val_loader, device)
                    print(f"val_loss={avg_val:.4f}")
                    history["val_loss"].append(avg_val)
                    epoch_metrics["val_loss"] = avg_val
                    val_line = f"  val={avg_val:.4f}"

                print(
                    f"Epoch {epoch+1}/{cfg.n_epochs}  "
                    f"train={avg_train:.4f}{val_line}  lr={current_lr:.2e}"
                )

                if on_epoch_end is not None:
                    on_epoch_end(epoch, epoch_metrics)

                # Inline probe — epoch 1 always; then every probe_interval epochs
                _run_probe = (
                    inline_probe_during_pretrain
                    and probe_train_loader is not None
                    and ((epoch + 1) == 1 or (epoch + 1) % max(1, probe_interval) == 0)
                )
                probe_metrics: Dict[str, float] = {}
                if _run_probe:
                    print(f"  [probe] Running inline linear probe for epoch {epoch+1} …")
                    _pt0 = _time.perf_counter()
                    probe_metrics = self._run_inline_probe(
                        probe_train_loader, probe_val_loader,
                        probe_n_epochs, probe_lr, probe_dropout, device,
                    )
                    probe_runtime = _time.perf_counter() - _pt0
                    probe_metrics["runtime_s"] = probe_runtime
                    print(
                        f"  [probe] Done in {probe_runtime:.1f}s  "
                        f"val_auroc={probe_metrics.get('val_auroc', 0):.4f}  "
                        f"val_aupr={probe_metrics.get('val_aupr', 0):.4f}"
                    )
                    if on_epoch_end is not None:
                        on_epoch_end(
                            epoch,
                            {"global_step": global_step,
                             **{f"probe_{k}": v for k, v in probe_metrics.items()}},
                        )

                # Checkpointing
                if ckpt_dir:
                    ckpt_monitor = epoch_metrics.get("val_loss", avg_train)
                    ckpt_payload = {
                        "epoch":           epoch + 1,
                        "global_step":     global_step,
                        "val_loss":        epoch_metrics.get("val_loss", None),
                        "train_loss":      avg_train,
                        "model_state":     self.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }
                    torch.save(ckpt_payload, os.path.join(ckpt_dir, "last.pt"))
                    save_bert_split_checkpoints(ckpt_payload["model_state"], ckpt_dir)
                    if ckpt_monitor < best_ckpt_metric:
                        best_ckpt_metric = ckpt_monitor
                        torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pt"))
                        print(f"  [ckpt] Saved best.pt  (monitor={ckpt_monitor:.4f})")
                    probe_auroc = probe_metrics.get("val_auroc", None) if _run_probe else None
                    if probe_auroc is not None and probe_auroc > best_probe_auroc:
                        best_probe_auroc = probe_auroc
                        torch.save(ckpt_payload, os.path.join(ckpt_dir, "probe_best.pt"))
                        print(f"  [ckpt] Saved probe_best.pt  (val_auroc={probe_auroc:.4f})")

                # Early stopping
                if cfg.early_stopping_patience > 0:
                    monitor = epoch_metrics.get(cfg.early_stopping_metric)
                    if monitor is not None:
                        if monitor < best_metric:
                            best_metric   = monitor
                            patience_left = cfg.early_stopping_patience
                        else:
                            patience_left -= 1
                            print(
                                f"  [early stopping] No improvement for "
                                f"{cfg.early_stopping_patience - patience_left}/"
                                f"{cfg.early_stopping_patience} epochs"
                            )
                            if patience_left == 0:
                                print(f"  [early stopping] Stopping at epoch {epoch+1}.")
                                stopped_early = True
                                break

        history["stopped_early"] = stopped_early  # type: ignore[assignment]
        return history

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, device: torch.device) -> float:
        """Returns average MLM loss over the val set."""
        self.eval()
        total, n = 0.0, 0
        for batch in loader:
            input_codes = batch["input_codes"].to(device, non_blocking=True)
            attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
            mlm_labels  = batch["mlm_labels"].to(device, non_blocking=True)
            values      = batch["values"].to(device, non_blocking=True)      if "values"      in batch else None
            z_scores    = batch["z_scores"].to(device, non_blocking=True)    if "z_scores"    in batch else None
            delta_times = batch["delta_times"].to(device, non_blocking=True) if "delta_times" in batch else None
            value_mask  = batch["value_mask"].to(device, non_blocking=True)  if "value_mask"  in batch else None

            loss, _ = self.model(
                input_codes, attn_mask, mlm_labels,
                values, z_scores, delta_times, value_mask,
            )
            total += loss.item()
            n     += 1

        self.train()
        return total / max(n, 1)

