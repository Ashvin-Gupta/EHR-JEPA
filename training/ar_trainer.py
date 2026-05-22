"""
Autoregressive (GPT-style) EHR pretraining via next-token prediction.

Architecture
------------
  EventEmbedding  →  [CLS | events | EOS]  →  EHRTransformerEncoder (causal)
                                                   ↓
                                           LM head  →  next-token loss
                                                   ↓
                                           CLS @ segment start  (downstream)

Each trajectory is framed as [CLS | tok_1 … tok_L | EOS].  Packed batches
concatenate multiple trajectories: [CLS|seq1|EOS|CLS|seq2|EOS|…] with a
segment-aware causal mask so tokens cannot attend across segment boundaries.

Forward returns: (ar_loss, cls_embedding)
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

from training.checkpoint_utils import save_ar_split_checkpoints
from training.trainer import _early_stopping_higher_is_better, _early_stopping_improved


@dataclass
class ARConfig:
    vocab_size: int = 5001
    lr: float = 1e-4
    weight_decay: float = 1e-2
    scheduler: str = "cosine_warmup"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    grad_clip: float = 0.0
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"
    checkpoint_dir: str = ""
    n_epochs: int = 10
    device: str = "cpu"
    probe_pooling: str = "cls"


class AREHRModel(nn.Module):
    """
    Causal transformer LM with learnable [CLS] and [EOS] tokens.

    ``eos_token_idx`` = vocab_size (one past the last real vocabulary entry).
    The LM head predicts vocab_size real tokens plus EOS (vocab_size + 1 logits).
    """

    def __init__(
        self,
        embedding: EventEmbedding,
        encoder: EHRTransformerEncoder,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.eos_token_idx = vocab_size
        d_model = encoder.config.d_model

        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.eos_token = nn.Parameter(torch.randn(d_model) * 0.02)

        self.lm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size + 1),
        )

    @property
    def output_dim(self) -> int:
        return self.encoder.config.d_model

    def _segment_metadata(
        self,
        attention_mask: torch.Tensor,
        segment_starts: Optional[torch.Tensor],
        segment_lengths: Optional[torch.Tensor],
    ) -> List[List[Tuple[int, int]]]:
        """Per-row list of (start, length) event spans."""
        B, L = attention_mask.shape
        meta: List[List[Tuple[int, int]]] = []
        for b in range(B):
            segs: List[Tuple[int, int]] = []
            if segment_starts is not None and segment_lengths is not None:
                for j in range(segment_starts.shape[1]):
                    st = int(segment_starts[b, j].item())
                    ln = int(segment_lengths[b, j].item())
                    if st < 0 or ln <= 0:
                        continue
                    segs.append((st, ln))
            if not segs:
                real = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
                if real.numel() > 0:
                    segs.append((int(real[0].item()), int(real.numel())))
            meta.append(segs)
        return meta

    def _build_sequence_tensors(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor],
        z_scores: Optional[torch.Tensor],
        delta_times: Optional[torch.Tensor],
        value_mask: Optional[torch.Tensor],
        segment_starts: Optional[torch.Tensor],
        segment_lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build padded batch of [CLS|events|EOS] per segment, concatenated when packed.

        Returns
        -------
        x, full_mask, position_ids, labels, segment_cls_indices
            x: (B, T_max, d_model)
            labels: (B, T_max) with -100 at non-predicted positions
            segment_cls_indices: (B,) index of first CLS per row (for downstream)
        """
        B = codes.shape[0]
        device = codes.device
        d_model = self.output_dim
        seg_meta = self._segment_metadata(attention_mask, segment_starts, segment_lengths)

        rows_x: List[torch.Tensor] = []
        rows_mask: List[torch.Tensor] = []
        rows_pos: List[torch.Tensor] = []
        rows_lab: List[torch.Tensor] = []
        cls_indices: List[int] = []

        for b in range(B):
            parts_emb: List[torch.Tensor] = []
            parts_mask: List[int] = []
            parts_pos: List[int] = []
            parts_lab: List[int] = []

            first_cls_idx = 0

            for seg_i, (start, length) in enumerate(seg_meta[b]):
                ev_codes = codes[b, start : start + length]
                ev_mask = attention_mask[b, start : start + length]

                ev_vals = values[b, start : start + length] if values is not None else None
                ev_z = z_scores[b, start : start + length] if z_scores is not None else None
                ev_dt = delta_times[b, start : start + length] if delta_times is not None else None
                ev_vm = value_mask[b, start : start + length] if value_mask is not None else None

                ev_emb = self.embedding(
                    ev_codes.unsqueeze(0),
                    values=ev_vals.unsqueeze(0) if ev_vals is not None else None,
                    z_scores=ev_z.unsqueeze(0) if ev_z is not None else None,
                    delta_times=ev_dt.unsqueeze(0) if ev_dt is not None else None,
                    value_mask=ev_vm.float().unsqueeze(0) if ev_vm is not None else None,
                ).squeeze(0)

                cls = self.cls_token.unsqueeze(0)
                eos = self.eos_token.unsqueeze(0)
                seg_emb = torch.cat([cls, ev_emb, eos], dim=0)

                parts_emb.append(seg_emb)
                seg_len = int(ev_mask.sum().item())
                parts_mask.extend([1] * (1 + seg_len + 1))

                pos_base = 0
                parts_pos.extend(range(pos_base, pos_base + 1 + seg_len + 1))

                labs = [-100]
                for t in range(seg_len):
                    labs.append(int(ev_codes[t].item()))
                labs.append(self.eos_token_idx)
                parts_lab.extend(labs)

            if not parts_emb:
                parts_emb.append(self.cls_token.unsqueeze(0))
                parts_mask.append(1)
                parts_pos.append(0)
                parts_lab.append(-100)

            row_x = torch.cat(parts_emb, dim=0)
            T = row_x.shape[0]
            rows_x.append(row_x)
            rows_mask.append(torch.tensor(parts_mask, device=device, dtype=torch.long))
            rows_pos.append(torch.tensor(parts_pos, device=device, dtype=torch.long))
            rows_lab.append(torch.tensor(parts_lab, device=device, dtype=torch.long))
            cls_indices.append(first_cls_idx)

        T_max = max(r.shape[0] for r in rows_x)
        x = torch.zeros(B, T_max, d_model, device=device)
        full_mask = torch.zeros(B, T_max, device=device, dtype=torch.long)
        position_ids = torch.zeros(B, T_max, device=device, dtype=torch.long)
        labels = torch.full((B, T_max), -100, device=device, dtype=torch.long)

        for b in range(B):
            t = rows_x[b].shape[0]
            x[b, :t] = rows_x[b]
            full_mask[b, :t] = rows_mask[b]
            position_ids[b, :t] = rows_pos[b]
            labels[b, :t] = rows_lab[b]

        cls_idx = torch.tensor(cls_indices, device=device, dtype=torch.long)
        return x, full_mask, position_ids, labels, cls_idx

    def _causal_segment_mask(self, T: int, seg_boundaries: List[int], device: torch.device) -> torch.Tensor:
        """
        (T, T) additive mask: causal within segments, blocked across segments.
        seg_boundaries: token indices where each segment starts (including 0).
        """
        mask = torch.full((T, T), float("-inf"), device=device)
        bounds = seg_boundaries + [T]
        for s in range(len(seg_boundaries)):
            a, bnd = bounds[s], bounds[s + 1]
            seg_len = bnd - a
            causal = torch.triu(torch.ones(seg_len, seg_len, device=device), diagonal=1)
            mask[a:bnd, a:bnd] = torch.where(causal.bool(), float("-inf"), 0.0)
        return mask

    def _encode(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        segment_starts: Optional[torch.Tensor] = None,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h, labels, full_mask, cls_indices)."""
        x, full_mask, position_ids, labels, cls_idx = self._build_sequence_tensors(
            codes,
            attention_mask,
            values,
            z_scores,
            delta_times,
            value_mask,
            segment_starts,
            segment_lengths,
        )
        B, T, _ = x.shape
        device = x.device

        attn_biases = []
        seg_meta = self._segment_metadata(attention_mask, segment_starts, segment_lengths)
        for b in range(B):
            boundaries = [0]
            offset = 0
            for _start, length in seg_meta[b]:
                boundaries.append(offset + 1 + length + 1)
                offset = boundaries[-1]
            t_b = int(full_mask[b].sum().item())
            attn_biases.append(self._causal_segment_mask(t_b, boundaries[:-1], device))

        T_max = x.shape[1]
        batch_bias = torch.full((B, T_max, T_max), float("-inf"), device=device)
        for b, bias in enumerate(attn_biases):
            t_b = bias.shape[0]
            batch_bias[b, :t_b, :t_b] = bias

        h = self.encoder(
            x,
            attention_mask=full_mask,
            position_ids=position_ids,
            attn_bias=batch_bias,
        )
        return h, labels, full_mask, cls_idx

    def encode_cls_embedding(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h, _labels, _mask, cls_idx = self._encode(
            codes, attention_mask, values, z_scores, delta_times, value_mask
        )
        return h[torch.arange(h.shape[0], device=h.device), cls_idx, :]

    def encode_pooled_embedding(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        pooling_mode: str = "cls",
    ) -> torch.Tensor:
        from models.sequence_pooling import mean_pool_sequence, parse_pooling_mode

        mode = parse_pooling_mode(pooling_mode)
        h, _labels, full_mask, cls_idx = self._encode(
            codes, attention_mask, values, z_scores, delta_times, value_mask
        )
        if mode == "cls":
            return h[torch.arange(h.shape[0], device=h.device), cls_idx, :]

        B = h.shape[0]
        event_h = []
        event_m = []
        for b in range(B):
            cls_pos = int(cls_idx[b].item())
            t = int(full_mask[b].sum().item())
            event_h.append(h[b, cls_pos + 1 : t - 1, :])
            event_m.append(full_mask[b, cls_pos + 1 : t - 1])
        max_e = max(e.shape[0] for e in event_h)
        d = h.shape[-1]
        pooled_h = torch.zeros(B, max_e, d, device=h.device)
        pooled_m = torch.zeros(B, max_e, device=h.device, dtype=full_mask.dtype)
        for b in range(B):
            le = event_h[b].shape[0]
            pooled_h[b, :le] = event_h[b]
            pooled_m[b, :le] = event_m[b]
        return mean_pool_sequence(pooled_h, pooled_m)

    def forward(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        segment_starts: Optional[torch.Tensor] = None,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, labels, _mask, cls_idx = self._encode(
            codes,
            attention_mask,
            values,
            z_scores,
            delta_times,
            value_mask,
            segment_starts,
            segment_lengths,
        )
        logits = self.lm_head(h)
        ar_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
        )
        cls_embedding = h[torch.arange(h.shape[0], device=h.device), cls_idx, :]
        return ar_loss, cls_embedding


class ARTrainer(nn.Module):
    """Thin container around AREHRModel with a BERT-compatible train_loop."""

    def __init__(self, model: AREHRModel, config: ARConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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
            cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
            return min_lr / cfg.lr + (1 - min_lr / cfg.lr) * cosine

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        from evaluation.frozen_ar_encoder import FrozenAREncoder
        from models.sequence_pooling import parse_pooling_mode

        encoder = FrozenAREncoder(
            self.model, pooling_mode=parse_pooling_mode(self.config.probe_pooling)
        ).to(device)
        probe = LinearProbe(encoder.output_dim, dropout=dropout).to(device)

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
        for key in (
            "train_loss",
            "train_auroc",
            "train_aupr",
            "train_recall",
            "train_precision",
            "train_accuracy",
        ):
            vals = history.get(key, [])
            if vals:
                train_metrics[key] = vals[-1]

        del encoder, probe
        gc.collect()

        combined: Dict[str, float] = {f"val_{k}": v for k, v in final_val.items()}
        combined.update(train_metrics)
        return combined

    def train_loop(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
        on_batch_end: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        probe_train_loader: Optional[DataLoader] = None,
        probe_val_loader: Optional[DataLoader] = None,
        probe_n_epochs: int = 15,
        probe_lr: float = 1e-3,
        probe_dropout: float = 0.1,
        probe_interval: int = 1,
        inline_probe_during_pretrain: bool = True,
        ddp_module: Optional[nn.Module] = None,
        is_main_process: bool = True,
        train_sampler=None,
    ) -> Dict[str, List[float]]:
        import time as _time

        cfg = self.config
        _forward = ddp_module if ddp_module is not None else self

        if optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

        if ddp_module is not None:
            device = next(ddp_module.parameters()).device
        else:
            device = torch.device(cfg.device)
            self.to(device)

        total_steps = cfg.n_epochs * len(train_loader)
        scheduler = self._build_scheduler(optimizer, total_steps)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}

        es_higher = _early_stopping_higher_is_better(cfg.early_stopping_metric)
        best_metric = -float("inf") if es_higher else float("inf")
        best_ckpt_metric = float("inf")
        best_probe_auroc = -float("inf")
        patience_left = cfg.early_stopping_patience
        stopped_early = False

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
        optimizer.zero_grad()
        for epoch in range(cfg.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.train()
            if ddp_module is not None:
                ddp_module.train()

            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                t0 = _time.perf_counter()

                codes = batch["codes"].to(device, non_blocking=True)
                attn_mask = batch["attention_mask"].to(device, non_blocking=True)
                seg_starts = batch.get("segment_starts")
                seg_lengths = batch.get("segment_lengths")
                if seg_starts is not None:
                    seg_starts = seg_starts.to(device, non_blocking=True)
                if seg_lengths is not None:
                    seg_lengths = seg_lengths.to(device, non_blocking=True)
                values = batch["values"].to(device, non_blocking=True) if "values" in batch else None
                z_scores = batch["z_scores"].to(device, non_blocking=True) if "z_scores" in batch else None
                delta_times = (
                    batch["delta_times"].to(device, non_blocking=True)
                    if "delta_times" in batch
                    else None
                )
                value_mask = (
                    batch["value_mask"].to(device, non_blocking=True)
                    if "value_mask" in batch
                    else None
                )

                accum = max(1, cfg.gradient_accumulation_steps)
                loss, _ = _forward(
                    codes,
                    attn_mask,
                    values,
                    z_scores,
                    delta_times,
                    value_mask,
                    seg_starts,
                    seg_lengths,
                )
                (loss / accum).backward()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                n_batches += 1

                if n_batches % accum == 0:
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                elapsed = _time.perf_counter() - t0

                if is_main_process and on_batch_end is not None and n_batches % accum == 0:
                    total_batches = len(train_loader)
                    epoch_progress = epoch + n_batches / max(total_batches, 1)
                    current_lr = optimizer.param_groups[0]["lr"]
                    on_batch_end(
                        epoch,
                        global_step,
                        {
                            "epoch": epoch_progress,
                            "loss_ar": batch_loss,
                            "learning_rate": current_lr,
                            "samples_per_second": codes.shape[0] / max(elapsed, 1e-6),
                        },
                    )

            avg_train = epoch_loss / max(n_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train)
            history["lr"].append(current_lr)

            if is_main_process:
                epoch_metrics: Dict[str, float] = {"global_step": global_step}

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
                        probe_train_loader,
                        probe_val_loader,
                        probe_n_epochs,
                        probe_lr,
                        probe_dropout,
                        device,
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
                            {
                                "global_step": global_step,
                                **{f"probe_{k}": v for k, v in probe_metrics.items()},
                            },
                        )

                if ckpt_dir:
                    ckpt_monitor = epoch_metrics.get("val_loss", avg_train)
                    ckpt_payload = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "val_loss": epoch_metrics.get("val_loss", None),
                        "train_loss": avg_train,
                        "model_state": self.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }
                    torch.save(ckpt_payload, os.path.join(ckpt_dir, "last.pt"))
                    save_ar_split_checkpoints(ckpt_payload["model_state"], ckpt_dir)
                    if ckpt_monitor < best_ckpt_metric:
                        best_ckpt_metric = ckpt_monitor
                        torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pt"))
                        print(f"  [ckpt] Saved best.pt  (monitor={ckpt_monitor:.4f})")
                    probe_auroc = probe_metrics.get("val_auroc", None) if _run_probe else None
                    if probe_auroc is not None and probe_auroc > best_probe_auroc:
                        best_probe_auroc = probe_auroc
                        torch.save(ckpt_payload, os.path.join(ckpt_dir, "probe_best.pt"))
                        print(f"  [ckpt] Saved probe_best.pt  (val_auroc={probe_auroc:.4f})")

                if cfg.early_stopping_patience > 0:
                    monitor = epoch_metrics.get(cfg.early_stopping_metric)
                    if monitor is not None:
                        if _early_stopping_improved(monitor, best_metric, es_higher):
                            best_metric = monitor
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

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, device: torch.device) -> float:
        self.eval()
        total, n = 0.0, 0
        for batch in loader:
            codes = batch["codes"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            seg_starts = batch.get("segment_starts")
            seg_lengths = batch.get("segment_lengths")
            if seg_starts is not None:
                seg_starts = seg_starts.to(device, non_blocking=True)
            if seg_lengths is not None:
                seg_lengths = seg_lengths.to(device, non_blocking=True)
            values = batch["values"].to(device, non_blocking=True) if "values" in batch else None
            z_scores = batch["z_scores"].to(device, non_blocking=True) if "z_scores" in batch else None
            delta_times = (
                batch["delta_times"].to(device, non_blocking=True)
                if "delta_times" in batch
                else None
            )
            value_mask = (
                batch["value_mask"].to(device, non_blocking=True)
                if "value_mask" in batch
                else None
            )

            loss, _ = self.model(
                codes,
                attn_mask,
                values,
                z_scores,
                delta_times,
                value_mask,
                seg_starts,
                seg_lengths,
            )
            total += loss.item()
            n += 1

        self.train()
        return total / max(n, 1)
