"""
EHR-JEPA Trainer.

Implements the correct two-pathway JEPA architecture:

  Target Pathway  — full sequence through shared encoder (always with grad so
                    L_cov can train the encoder); target span tokens sliced out.
  Context Pathway — target spans physically dropped; compact sequence with
                    original RoPE position IDs passed to the encoder.

Two predictor branches, controlled by TrainerConfig.use_perceiver:

  Branch A — Perceiver-JEPA (use_perceiver=True)
    Target spans  → LatentCrossAttentionPool → Z_tgt [B, 16, d]
    Context       → LatentCrossAttentionPool → Z_ctx [B, 16, d]
                 + TemporalSpanPrompt + LayerNorm → Predictor → Z_hat [B, 16, d]
    Loss: MSE(Z_hat, Z_tgt.detach()) + λ·CovReg(Z_tgt)
    Spans with N_span < min_span_for_perceiver are skipped.

  Branch B — Token I-JEPA (use_perceiver=False)
    Target pathway: [CLS | full sequence] → encoder → Y_tgt at target positions.
    Context pathway: target tokens dropped → compact context → encoder.
    Predictor input: full-length (B, L, d) with context encodings at context
    indices and learnable MASK (+ hours-since-first time bias) at target indices.
    Token predictor → (B, L, d); slice target positions → Y_hat.
    Optional ProjectionHead on Y_hat / Y_tgt (when use_proj_head).
    Loss: MSE(Y_hat_proj, Y_tgt_proj.detach()) + λ·CovReg(Y_tgt_proj)

    Downstream (token branch): encoder + pretrained CLS only (no pooler / predictor).

Both branches share the same L_total = L_pred + λ·L_cov.
The target encoder NEVER uses no_grad — it receives gradient only via L_cov.
Stop-grad is applied inside jepa_prediction_loss (detach on target).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from loss.covariance_reg import SIGRegLoss
from loss.jepa_loss import (
    future_time_decay_weights,
    jepa_prediction_loss,
    jepa_prediction_loss_token_masked,
    jepa_prediction_loss_weighted,
)
from masking.causal_future_masking import CausalFutureMasker
from masking.span_masking import SpanMasker
from models.cls_encoding import encode_embeddings_with_cls
from models.event_embedding import EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.predictor import HoursSinceFirstEmbedding, Predictor, TemporalSpanPrompt
from models.projection_head import ProjectionHead
from models.attention_masks import (
    build_causal_single_partial_causal_mask_batch,
    build_causal_single_quadrant_mask_batch,
)
from models.transformer_encoder import EHRTransformerEncoder

from training.checkpoint_utils import save_jepa_split_checkpoints


def _pre_mask_dict_from_batch(batch: Dict) -> Optional[Dict]:
    """Build trainer pre_mask from collator batch keys (span vs causal)."""
    if "mask_causal_contexts" in batch:
        return {
            "mask_causal_contexts": batch["mask_causal_contexts"],
            "mask_causal_targets": batch["mask_causal_targets"],
            "mask_causal_span_times": batch["mask_causal_span_times"],
        }
    if "mask_context_indices" in batch:
        out = {
            "mask_context_indices": batch["mask_context_indices"],
            "mask_target_spans": batch["mask_target_spans"],
            "mask_span_times": batch["mask_span_times"],
        }
        if "mask_target_delta_minutes" in batch:
            out["mask_target_delta_minutes"] = batch["mask_target_delta_minutes"]
        if "mask_cutpoint_indices" in batch:
            out["mask_cutpoint_indices"] = batch["mask_cutpoint_indices"]
        if "mask_context_start_indices" in batch:
            out["mask_context_start_indices"] = batch["mask_context_start_indices"]
        return out
    return None


def _compute_causal_single_monitoring(
    all_context_indices: List[List[int]],
    all_target_spans: List[List[List[int]]],
    cutpoints: Optional[List[int]] = None,
    context_starts: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Causal-single mask geometry for W&B (two-index design).

    Active region [s, e] with cut t: context [s,t], target (t,e].

    causal_cut_position_ratio:
        (t - s) / (e - s) — split point along the active window.

    causal_cut_over_context_index_span:
        (t - s + 1) / (e - s + 1) — share of active index range used as context.

    causal_context_token_fraction / causal_target_token_fraction:
        |context| / (|context|+|target|) and complement.
    """
    cut_positions: List[float] = []
    cut_over_ctx: List[float] = []
    ctx_fracs: List[float] = []
    tgt_fracs: List[float] = []

    for b, ctx in enumerate(all_context_indices):
        if not ctx:
            continue
        spans = all_target_spans[b] if b < len(all_target_spans) else [[]]
        tgt_flat: List[int] = []
        for sp in spans:
            tgt_flat.extend(sp)
        tgt_set = set(tgt_flat)
        s = min(ctx)
        if context_starts is not None and b < len(context_starts) and context_starts[b] >= 0:
            s = int(context_starts[b])
        t = max(ctx)
        if cutpoints is not None and b < len(cutpoints) and cutpoints[b] >= 0:
            t = int(cutpoints[b])
        e = max(max(ctx), max(tgt_set) if tgt_set else max(ctx))

        n_ctx = len(ctx)
        n_tgt = len(tgt_set)
        denom = max(n_ctx + n_tgt, 1)
        ctx_fracs.append(n_ctx / denom)
        tgt_fracs.append(n_tgt / denom)

        active_span = max(e - s, 1)
        cut_positions.append((t - s) / float(active_span))
        active_index_len = e - s + 1
        cut_over_ctx.append((t - s + 1) / float(max(active_index_len, 1)))

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "causal_cut_position_ratio": _mean(cut_positions),
        "causal_cut_over_context_index_span": _mean(cut_over_ctx),
        "causal_context_token_fraction": _mean(ctx_fracs),
        "causal_target_token_fraction": _mean(tgt_fracs),
    }


def _zero_loss_connected(t: torch.Tensor) -> torch.Tensor:
    """Scalar zero with grad_fn so train_loop can always call backward()."""
    if t.numel() > 0:
        return t.sum() * 0.0
    return t.new_zeros((), requires_grad=True)


def _format_forward_debug(dbg: Dict) -> str:
    """One-line summary of the last forward() debug snapshot."""
    parts = [f"path={dbg.get('zero_path', 'ok')}"]
    for key in (
        "pre_mask",
        "num_spans",
        "n_mask_ok",
        "B",
        "n_real_tokens",
        "len_target_spans_list",
        "tgt_tensor_shape",
        "n_valid_bs",
        "B_eff",
        "avg_nc",
        "avg_nt",
        "l_pred",
        "l_cov",
        "rank_me_recomputed",
        "rank_me",
    ):
        if key in dbg:
            parts.append(f"{key}={dbg[key]}")
    return "  ".join(parts)


def _early_stopping_higher_is_better(metric_name: str) -> bool:
    """True when a larger monitor value is better (e.g. AUROC, accuracy)."""
    name = metric_name.lower()
    for token in ("auroc", "aupr", "accuracy", "rank_me", "f1", "recall", "precision"):
        if token in name:
            return True
    return False


def _early_stopping_improved(
    monitor: float, best_metric: float, higher_is_better: bool
) -> bool:
    if higher_is_better:
        return monitor > best_metric
    return monitor < best_metric


def _rank_me_from_rows(z_rows: torch.Tensor) -> float:
    """
    Effective rank (RankMe) from embedding rows (n, d).

    Runs SVD on CPU in float64 to avoid CUDA cusolver 'failed to converge'
    warnings / NaNs from the fast batched driver on fp32 tensors.
    """
    if z_rows.ndim != 2 or z_rows.shape[0] < 2:
        return 0.0
    z = z_rows.detach().float()
    if not torch.isfinite(z).all():
        return 0.0
    try:
        z_cpu = z.cpu().double()
        _, s, _ = torch.linalg.svd(z_cpu, full_matrices=False)
        s = s.clamp(min=0.0).float()
        denom = s.sum() + 1e-8
        p = s / denom
        return float(torch.exp(-(p * torch.log(p + 1e-8)).sum()).item())
    except Exception:
        return 0.0


def _total_grad_norm(parameters) -> float:
    """L2 norm over all parameter gradients (0 if no grads)."""
    norms = [
        p.grad.detach().norm(2)
        for p in parameters
        if p.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def _sample_target_rows_for_rank_me(
    span_t: torch.Tensor,
    pad_mask: torch.Tensor,
    max_rows: int,
) -> torch.Tensor:
    """
    Real (non-pad) target embeddings for RankMe.

    span_t: (B, L, d); pad_mask: (B, L) with True on padded positions.
    """
    if span_t.ndim != 3 or pad_mask.shape != span_t.shape[:2]:
        return span_t.new_zeros(0, span_t.shape[-1])
    real = span_t[~pad_mask]
    if real.shape[0] < 2:
        return real
    if real.shape[0] <= max_rows:
        return real
    idx = torch.randperm(real.shape[0], device=real.device)[:max_rows]
    return real[idx]


@dataclass
class TrainerConfig:
    # Branch selection
    use_perceiver: bool = True

    # Branch A: skip target spans shorter than this (too short to pool meaningfully)
    min_span_for_perceiver: int = 15

    # Projection heads (Linear + BatchNorm1d) applied after target perceiver and
    # predictor outputs.  Breaks the unit-sphere constraint from the final
    # LayerNorm and gives the anti-collapse objective a free representation space.
    # Recommended: True.  Set False to ablate.
    use_proj_head: bool = True

    lambda_cov: float = 0.1
    # Branch B + causal_single: L_pred uses W=exp(-lambda*delta_minutes) per target token.
    # lambda in 1/minutes; 0 disables (plain MSE).  W is clamped below at future_time_decay_weight_floor.
    future_time_decay_lambda: float = 0.0
    future_time_decay_weight_floor: float = 0.05
    # causal_single: minimum future-window events (align with masking.min_target_events).
    min_target_events: int = 10
    # causal_single Branch B token_predictor self-attention:
    #   "bidirectional" — full attention among real tokens (default)
    #   "quadrant" — top-left (CLS+context) bidirectional; block context→target;
    #                target→context full; target↔target diagonal only
    #   "partial_causal" — same as quadrant except top-left is lower-triangular
    #                      (causal along [CLS | context] compact order)
    causal_single_predictor_attn: str = "bidirectional"

    # Optimiser
    lr: float = 1e-4
    weight_decay: float = 1e-2

    # Gradient clipping — set to 0 to disable
    grad_clip: float = 1.0

    # LR scheduler
    # Choices: "cosine_warmup" | "cosine" | "linear_warmup" | "none"
    scheduler: str = "cosine_warmup"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    # Early stopping — set patience to 0 to disable
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"

    # RankMe on training steps: compute every N optimizer steps (0 = eval only).
    rank_me_every_n_steps: int = 50
    rank_me_train_max_rows: int = 256

    # Mixed precision (CUDA only). bf16 is preferred on Ampere+; float16 uses GradScaler.
    use_amp: bool = True
    amp_dtype: str = "bf16"

    # W&B batch metrics: log every N optimizer steps (1 = every step).
    wandb_log_every_n_steps: int = 10

    # Checkpointing — set to "" to disable
    # Best model (by val_loss, or train_loss if no val set) is saved to
    # {checkpoint_dir}/best.pt.  End-of-training model saved to
    # {checkpoint_dir}/last.pt.
    checkpoint_dir: str = ""

    # General
    n_epochs: int = 10
    device: str = "cpu"

    # Forward-path debugging (main process only in train_loop).
    debug_jepa: bool = False
    debug_jepa_first_batches: int = 5
    debug_jepa_every_n_steps: int = 0
    debug_jepa_on_zero_loss: bool = True

    # masking.strategy from YAML ("span_budget" | "causal_single" | "causal_future").
    # causal_single uses span batch keys (one cut); causal_future uses multi-cut keys.
    masking_strategy: str = "span_budget"

    # Inline linear probe pooling when not using perceiver: "cls" | "mean_pool"
    probe_pooling: str = "cls"


class JEPATrainer(nn.Module):
    """
    Container for all JEPA modules.

    Parameters
    ----------
    embedding:
        EventEmbedding module.
    encoder:
        Single shared EHRTransformerEncoder — used for BOTH target and context
        pathways (same weight tensor, two forward passes per batch).
    prompt:
        TemporalSpanPrompt (Branch A / Perceiver only).
    time_embed:
        HoursSinceFirstEmbedding — additive wall-clock bias in token predictor (Branch B).
    predictor:
        Shallow Predictor transformer operating on latent tokens (Branch A).
    token_predictor:
        Shallow EHRTransformerEncoder operating on context + mask tokens (Branch B).
    context_pooler:
        LatentCrossAttentionPool for the context sequence (Branch A only; None for B).
    target_pooler:
        LatentCrossAttentionPool for target spans (Branch A only; None for B).
    cov_loss:
        SIGRegLoss — anti-collapse regularizer on target embeddings (both branches).
    masker:
        SpanMasker or CausalFutureMasker (fallback when pre_mask is None).
    config:
        TrainerConfig.
    """

    def __init__(
        self,
        embedding: EventEmbedding,
        encoder: EHRTransformerEncoder,
        prompt: TemporalSpanPrompt,
        time_embed: HoursSinceFirstEmbedding,
        predictor: Predictor,
        token_predictor: EHRTransformerEncoder,
        context_pooler: Optional[LatentCrossAttentionPool],
        target_pooler: Optional[LatentCrossAttentionPool],
        cov_loss: SIGRegLoss,
        masker: SpanMasker | CausalFutureMasker,
        config: TrainerConfig,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.prompt = prompt
        self.time_embed = time_embed
        self.predictor = predictor
        self.token_predictor = token_predictor
        self.context_pooler = context_pooler
        self.target_pooler = target_pooler
        self.cov_loss = cov_loss
        self.masker = masker
        self.masking_strategy = config.masking_strategy
        self.config = config

        # Training step for SIGReg RNG sync across DDP ranks (set each batch).
        self._cov_global_step: int = 0
        # Whether SIGReg should synchronize its statistic across DDP ranks.
        # Disable during rank-0-only validation to avoid collective mismatch.
        self._cov_sync_ddp: bool = True

        d_model = encoder.config.d_model
        # [CLS] for target-pathway encoding and downstream evaluation (token branch)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        # Learnable mask token for Branch B predictor slots at target positions
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        # Projection heads (Linear + BN1d) — applied after target perceiver and
        # predictor outputs to break the unit-sphere constraint of the final LN.
        # Registered as None when disabled so state_dict round-trips cleanly.
        if config.use_proj_head:
            self.target_proj: Optional[ProjectionHead] = ProjectionHead(d_model)
            self.pred_proj:   Optional[ProjectionHead] = ProjectionHead(d_model)
        else:
            self.target_proj = None
            self.pred_proj   = None

        # Side-channel populated during each forward() call.
        # The train loop reads these after calling self.forward() so monitoring
        # metrics from inside the forward (target std-dev, mask ratio) are
        # available without changing the public return signature to 5+ values.
        self._batch_mon: Dict[str, float] = {}
        self._last_rank_me: float = 0.0
        self._forward_debug: Dict = {}

    def _dbg(self, **kwargs) -> None:
        """Merge debug fields for the current forward (train_loop may print)."""
        self._forward_debug.update(kwargs)

    def _log_jepa_debug(
        self,
        *,
        global_step: int,
        epoch: int,
        batch_in_epoch: int,
        l_pred: float,
        l_cov: float,
        l_total: float,
        force: bool = False,
    ) -> None:
        """Print forward-path diagnostics on the main process."""
        cfg = self.config
        if not cfg.debug_jepa and not force:
            if not cfg.debug_jepa_on_zero_loss:
                return
            if l_pred != 0.0 or l_cov != 0.0:
                return

        should_print = force or cfg.debug_jepa
        if cfg.debug_jepa and not should_print:
            if batch_in_epoch < cfg.debug_jepa_first_batches:
                should_print = True
            if (
                cfg.debug_jepa_every_n_steps > 0
                and global_step % cfg.debug_jepa_every_n_steps == 0
            ):
                should_print = True
        if cfg.debug_jepa_on_zero_loss and (l_pred == 0.0 and l_cov == 0.0):
            should_print = True
        if not should_print:
            return

        dbg = dict(self._forward_debug)
        print(
            f"[jepa-debug] epoch={epoch} step={global_step} batch={batch_in_epoch} "
            f"l_pred={l_pred:.6f} l_cov={l_cov:.6f} l_total={l_total:.6f} "
            f"grad={dbg.get('l_total_requires_grad', '?')}",
            flush=True,
        )
        print(f"  {_format_forward_debug(dbg)}", flush=True)
        mon = self._batch_mon
        print(
            f"  mon: avg_ctx={mon.get('avg_context_length', 0):.2f} "
            f"avg_tgt={mon.get('avg_target_span_length', 0):.2f} "
            f"valid_mask_frac={mon.get('causal_valid_mask_fraction', float('nan')):.3f} "
            f"std_dev={mon.get('std_dev_embeddings', 0):.4f} "
            f"rank_me={mon.get('rank_me', 0):.1f} "
            f"N_in={mon.get('_N_input', 0)} N_tgt={mon.get('_N_target', 0)}",
            flush=True,
        )
        if dbg.get("num_spans_per_sample_summary"):
            print(f"  spans/batch: {dbg['num_spans_per_sample_summary']}", flush=True)
        if dbg.get("sample_mask_preview"):
            print(f"  sample[0]: {dbg['sample_mask_preview']}", flush=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        hours_since_first: Optional[torch.Tensor] = None,
        pre_mask: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        codes:           LongTensor  (B, L)
        attention_mask:  LongTensor  (B, L) — 1=real, 0=pad
        values:          FloatTensor (B, L) | None
        z_scores:        FloatTensor (B, L) | None
        delta_times:     FloatTensor (B, L) | None
        value_mask:      LongTensor  (B, L) | None
        hours_since_first:
                         FloatTensor (B, L) | None — hours since first window event;
                         used by Branch B token predictor (additive, on top of RoPE).
        pre_mask:        Span or causal dict from MEDSCollator, or None.
                         Span keys: mask_context_indices, mask_target_spans,
                         mask_span_times.  Causal keys: mask_causal_contexts,
                         mask_causal_targets, mask_causal_span_times.

        Returns
        -------
        (L_pred, L_cov, L_total) — scalar tensors.
        Also populates self._batch_mon with monitoring scalars (no extra GPU→CPU
        transfers — values are already detached floats).
        """
        B, L = codes.shape
        device = codes.device
        self._forward_debug = {}
        n_real_batch = int(attention_mask.sum().item())
        self._dbg(
            masking_strategy=self.config.masking_strategy,
            use_perceiver=self.config.use_perceiver,
            B=B,
            L=L,
            n_real_tokens=n_real_batch,
            pre_mask=pre_mask is not None,
            min_target_events_cfg=self.config.min_target_events,
        )

        # 1a. Embeddings
        x = self.embedding(
            codes,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask.float() if value_mask is not None else None,
        )  # (B, L, d)

        # 1b. Post-MLP embedding std-dev monitoring.
        #
        # Both metrics measure the std of the embedding vectors output by
        # EventEmbedding (after the residual MLP + LayerNorm), computed over
        # real (non-padding) token positions and all d_model dimensions.
        #
        # std_dev_values — non-zero only when use_value=True (MLP includes value/z_score).
        #   Diagnoses value-MLP health: if it collapses to 0, the residual
        #   connection keeps embeddings alive but MLP contributes nothing.
        #
        # std_dev_times  — non-zero only when use_time=True (MLP includes delta_time).
        #   Same diagnostic for the time branch.
        #
        # Both are 0.0 when the respective feature flag is disabled, making
        # it easy to see which mode is active in W&B.
        real_mask = attention_mask.bool()
        x_real = x[real_mask]   # (N_real, d_model)

        _std_values: float = 0.0
        if z_scores is not None and x_real.numel() > 0:
            _std_values = x_real.detach().float().std().item()

        _std_times: float = 0.0
        if delta_times is not None and x_real.numel() > 0:
            _std_times = x_real.detach().float().std().item()

        if hours_since_first is None:
            hours_since_first = torch.arange(
                L, device=device, dtype=torch.float
            ).unsqueeze(0).expand(B, -1)

        if pre_mask is not None and "mask_causal_contexts" in pre_mask:
            return self._forward_causal_multi_cut(
                x,
                attention_mask,
                pre_mask,
                device,
                hours_since_first,
                _std_values,
                _std_times,
            )

        # 2. Span masking — use pre-computed results from the DataLoader worker
        #    when available; fall back to on-the-fly masking otherwise.
        all_target_delta_minutes: Optional[List[List[List[float]]]] = None
        mask_cutpoints: Optional[List[int]] = None
        mask_context_starts: Optional[List[int]] = None
        if pre_mask is not None:
            all_context_indices = pre_mask["mask_context_indices"]
            all_target_spans = pre_mask["mask_target_spans"]
            all_span_times = pre_mask.get("mask_span_times")
            if all_span_times is None:
                all_span_times = [
                    [(0.0, 0.0)] * max(len(all_target_spans[b]), 1)
                    for b in range(len(all_target_spans))
                ]
            all_target_delta_minutes = pre_mask.get("mask_target_delta_minutes")
            mask_cutpoints = pre_mask.get("mask_cutpoint_indices")
            mask_context_starts = pre_mask.get("mask_context_start_indices")
        else:
            if isinstance(self.masker, CausalFutureMasker):
                cc: List[List[List[int]]] = []
                tt: List[List[List[int]]] = []
                stt: List[List[Tuple[float, float]]] = []
                for b in range(B):
                    cr = self.masker(
                        seq_len=L,
                        attention_mask=attention_mask[b],
                        times_hours=None,
                    )
                    cc.append(cr.contexts)
                    tt.append(cr.target_spans)
                    stt.append(cr.span_times)
                return self._forward_causal_multi_cut(
                    x,
                    attention_mask,
                    {
                        "mask_causal_contexts": cc,
                        "mask_causal_targets": tt,
                        "mask_causal_span_times": stt,
                    },
                    device,
                    _std_values,
                    _std_times,
                )
            all_context_indices = []
            all_target_spans = []
            all_span_times = []
            mask_cutpoints = []
            mask_context_starts = []
            for b in range(B):
                result = self.masker(
                    seq_len=L,
                    attention_mask=attention_mask[b],
                    times=None,
                )
                all_context_indices.append(result.context_indices)
                all_target_spans.append(result.target_spans)
                all_span_times.append(result.span_times)
                cp = getattr(result, "cutpoint_index", None)
                mask_cutpoints.append(int(cp) if cp is not None else -1)
                cs = getattr(result, "context_start_index", None)
                mask_context_starts.append(int(cs) if cs is not None else -1)
            if not any(i >= 0 for i in mask_cutpoints):
                mask_cutpoints = None
            if not any(i >= 0 for i in mask_context_starts):
                mask_context_starts = None

        # --- Masking stats from the *full* collator mask (before batch truncation) ---
        # We later slice every sample to min(num_spans) so Perceiver tensors align
        # across the batch.  W&B ratios must still reflect the true mask budget
        # (~mask_ratio * N_input), not the truncated span list.
        B_size_pre = attention_mask.shape[0]
        N_model_pre = int(attention_mask.sum().item())
        per_sample_target_area_full = [
            len({p for span in spans for p in span})
            for spans in all_target_spans
        ]
        N_target_full = sum(per_sample_target_area_full)
        N_context_full = N_model_pre - N_target_full

        # Align span count across batch.  Ignore samples with no spans (causal_single
        # failures return target_spans=[]); a single empty row must not force
        # num_spans=0 and zero the whole batch.
        num_spans_per_sample = [len(spans) for spans in all_target_spans]
        positive_span_counts = [n for n in num_spans_per_sample if n > 0]
        num_spans = min(positive_span_counts) if positive_span_counts else 0
        n_mask_ok = sum(1 for n in num_spans_per_sample if n > 0)

        avg_ctx_len = sum(len(ctx) for ctx in all_context_indices) / max(B_size_pre, 1)
        avg_tgt_area = sum(per_sample_target_area_full) / max(B_size_pre, 1)

        span_summary = {}
        for n in set(num_spans_per_sample):
            span_summary[n] = span_summary.get(n, 0) + 1
        self._dbg(
            num_spans=num_spans,
            n_mask_ok=n_mask_ok,
            num_spans_per_sample_summary=span_summary,
        )
        if all_context_indices:
            b0 = 0
            self._dbg(
                sample_mask_preview={
                    "ctx_len": len(all_context_indices[b0]),
                    "tgt_spans": len(all_target_spans[b0]),
                    "tgt_len": (
                        len(all_target_spans[b0][0])
                        if all_target_spans[b0]
                        else 0
                    ),
                    "ctx_head": all_context_indices[b0][:5],
                    "tgt_head": (
                        all_target_spans[b0][0][:5]
                        if all_target_spans[b0] and all_target_spans[b0][0]
                        else []
                    ),
                }
            )

        if num_spans == 0:
            self._dbg(zero_path="num_spans_zero")
            z = _zero_loss_connected(x)
            self._batch_mon = {
                "std_dev_embeddings": 0.0, "rank_me": 0.0,
                "std_dev_values": _std_values, "std_dev_times": _std_times,
                "_N_input": N_model_pre, "_N_target": N_target_full,
                "_N_context": N_context_full,
                "avg_context_length": avg_ctx_len,
                "avg_target_span_length": avg_tgt_area,
            }
            if self.config.masking_strategy == "causal_single":
                n_ok = sum(1 for n in num_spans_per_sample if n > 0)
                self._batch_mon["causal_valid_mask_fraction"] = n_ok / max(
                    B_size_pre, 1
                )
            return z, z, z

        all_target_spans = [spans[:num_spans] for spans in all_target_spans]
        all_span_times   = [st[:num_spans]    for st    in all_span_times]
        if all_target_delta_minutes is not None:
            all_target_delta_minutes = [dm[:num_spans] for dm in all_target_delta_minutes]

        # 3. Target pathway — [CLS | events], always with grad
        target_enc_out = encode_embeddings_with_cls(
            self.encoder, self.cls_token, x, attention_mask
        )  # (B, L+1, d); event index i → position i+1
        target_spans_list, _target_span_pad_masks = self._extract_target_spans(
            target_enc_out, all_target_spans, index_offset=1
        )  # list of num_spans tensors, each (B, N_span_s, d)
        if target_spans_list:
            self._dbg(
                len_target_spans_list=len(target_spans_list),
                tgt_tensor_shape=tuple(target_spans_list[0].shape),
                tgt_rows_with_tokens=int(
                    (target_spans_list[0].abs().sum(dim=-1) > 0).any(dim=-1).sum().item()
                ),
            )
        else:
            self._dbg(len_target_spans_list=0, zero_path="extract_target_spans_empty")

        # 4. Context pathway — compact extraction with original RoPE position IDs
        ctx_out, ctx_pos_ids, ctx_mask = self._extract_context(
            x, all_context_indices, device
        )  # (B, N_ctx, d), (B, N_ctx), (B, N_ctx)
        context_enc_out = self.encoder(
            ctx_out, attention_mask=ctx_mask, position_ids=ctx_pos_ids
        )  # (B, N_ctx, d)

        # 5. Branch routing
        if self.config.use_perceiver:
            l_pred, l_cov = self._forward_perceiver(
                context_enc_out,
                ctx_mask,
                target_spans_list,
                _target_span_pad_masks,
                all_span_times,
            )
        else:
            if self.config.masking_strategy == "causal_single":
                l_pred, l_cov = self._forward_token_causal_single(
                    context_enc_out,
                    ctx_pos_ids,
                    ctx_mask,
                    target_spans_list,
                    all_target_spans,
                    hours_since_first,
                    all_target_delta_minutes=all_target_delta_minutes,
                )
            else:
                l_pred, l_cov = self._forward_token(
                    context_enc_out,
                    ctx_pos_ids,
                    ctx_mask,
                    target_spans_list,
                    all_target_spans,
                    hours_since_first,
                    L,
                    all_target_delta_minutes=all_target_delta_minutes,
                )

        l_total = l_pred + self.config.lambda_cov * l_cov

        # --- Monitoring metrics (detached, no overhead) ---
        # Average std-dev across target-span embedding dimensions
        # (collapse indicator: should stay well above 0)
        with torch.no_grad():
            if target_spans_list:
                span0 = target_spans_list[0]
                pad0 = (
                    _target_span_pad_masks[0]
                    if _target_span_pad_masks
                    else torch.zeros(
                        span0.shape[0],
                        span0.shape[1],
                        dtype=torch.bool,
                        device=span0.device,
                    )
                )
                rank_rows = _sample_target_rows_for_rank_me(
                    span0, pad0, self.config.rank_me_train_max_rows
                )
                all_tgt = rank_rows if rank_rows.shape[0] > 0 else span0.reshape(-1, span0.shape[-1])
                std_dev = all_tgt.std(dim=0).mean().item()
                n_rm = self.config.rank_me_train_max_rows
                every = self.config.rank_me_every_n_steps
                if every > 0 and self._cov_global_step % every == 0:
                    rank_me = _rank_me_from_rows(rank_rows)
                    self._last_rank_me = rank_me
                else:
                    rank_me = self._last_rank_me
            else:
                all_tgt = x.new_zeros((1, x.shape[-1]))
                std_dev = 0.0
                rank_me = self._last_rank_me

        # Raw token counts — use full-mask totals (see N_target_full above).
        # Ratios in train loop: target_ratio = N_target_full / N_input,
        # context_ratio = N_context_full / sum(N_sequence) (per-batch).

        rank_recomputed = bool(
            target_spans_list
            and self.config.rank_me_every_n_steps > 0
            and self._cov_global_step % self.config.rank_me_every_n_steps == 0
        )

        self._batch_mon = {
            "std_dev_embeddings":     std_dev,
            "rank_me":                rank_me,
            "rank_me_updated":        rank_recomputed,
            # Feature std-devs (0.0 when feature is disabled):
            #   std_dev_values = std of z_scores (should be ~1 if normalizer healthy)
            #   std_dev_times  = std of log(1+hours) (should be ~1)
            "std_dev_values":         _std_values,
            "std_dev_times":          _std_times,
            # Batch sums: N_input = real tokens in window; N_target/N_context from
            # full span mask (not truncated to min(num_spans) used in forward).
            "_N_input":               N_model_pre,
            "_N_target":              N_target_full,
            "_N_context":             N_context_full,
            "avg_context_length":     avg_ctx_len,
            "avg_target_span_length": avg_tgt_area,
            "_tgt_embs_for_rank": (
                rank_rows.detach()
                if not self.training and target_spans_list
                else None
            ),
        }
        if self.config.masking_strategy == "causal_single":
            n_ok = sum(1 for n in num_spans_per_sample if n > 0)
            self._batch_mon["causal_valid_mask_fraction"] = n_ok / max(B_size_pre, 1)
            self._batch_mon.update(
                _compute_causal_single_monitoring(
                    all_context_indices,
                    all_target_spans,
                    cutpoints=mask_cutpoints,
                    context_starts=mask_context_starts,
                )
            )

        self._dbg(
            l_pred=float(l_pred.detach().item()),
            l_cov=float(l_cov.detach().item()),
            l_total_requires_grad=bool(l_total.requires_grad),
            rank_me_recomputed=rank_recomputed,
            rank_me=float(rank_me),
        )
        if float(l_pred.detach().item()) == 0.0 and float(l_cov.detach().item()) == 0.0:
            if "zero_path" not in self._forward_debug:
                self._dbg(zero_path="loss_zero_no_tag")

        return l_pred, l_cov, l_total

    def _forward_causal_multi_cut(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        pre_mask: Dict,
        device: torch.device,
        hours_since_first: torch.Tensor,
        _std_values: float,
        _std_times: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        S independent (context_s, target_s) pairs per batch row.
        One full-sequence target encoder forward; S context encoder passes;
        mean loss over s (same aggregation as multi-span span masking).

        Rows with empty context or target shorter than min_span_for_perceiver are
        dropped **per cut** (not whole-batch skip).  Including an empty-context row
        in the pooler sets key_padding_mask=all-True for that row → NaN in Z_ctx.
        """
        B = x.shape[0]
        cc_list: List[List[List[int]]] = pre_mask["mask_causal_contexts"]
        tc_list: List[List[List[int]]] = pre_mask["mask_causal_targets"]
        st_list: List[List[Tuple[float, float]]] = pre_mask["mask_causal_span_times"]
        S = len(cc_list[0])

        B_size_pre = attention_mask.shape[0]
        N_model_pre = int(attention_mask.sum().item())
        N_target_full = sum(len(tc_list[b][s]) for b in range(B) for s in range(S))
        N_context_full = sum(len(cc_list[b][s]) for b in range(B) for s in range(S))

        target_enc_out = encode_embeddings_with_cls(
            self.encoder, self.cls_token, x, attention_mask
        )

        pred_losses: List[torch.Tensor] = []
        cov_losses: List[torch.Tensor] = []
        all_tgt_embs: List[torch.Tensor] = []
        min_span = self.config.min_span_for_perceiver
        n_cuts_used = 0

        for s in range(S):
            valid_bs = [
                b
                for b in range(B)
                if len(cc_list[b][s]) > 0 and len(tc_list[b][s]) >= min_span
            ]
            if not valid_bs:
                continue

            n_cuts_used += 1
            x_sub = x[valid_bs]
            target_enc_sub = target_enc_out[valid_bs]

            all_context_indices = [cc_list[b][s] for b in valid_bs]
            all_target_spans = [[tc_list[b][s]] for b in valid_bs]
            all_span_times = [[st_list[b][s]] for b in valid_bs]

            ctx_out, ctx_pos_ids, ctx_mask = self._extract_context(
                x_sub, all_context_indices, device
            )
            context_enc_out = self.encoder(
                ctx_out, attention_mask=ctx_mask, position_ids=ctx_pos_ids
            )

            target_spans_list, target_span_pad_masks = self._extract_target_spans(
                target_enc_sub, all_target_spans, index_offset=1
            )

            if self.config.use_perceiver:
                l_p, l_c = self._forward_perceiver(
                    context_enc_out,
                    ctx_mask,
                    target_spans_list,
                    target_span_pad_masks,
                    all_span_times,
                )
            else:
                h_sub = hours_since_first[valid_bs]
                l_p, l_c = self._forward_token(
                    context_enc_out,
                    ctx_pos_ids,
                    ctx_mask,
                    target_spans_list,
                    all_target_spans,
                    h_sub,
                    x_sub.shape[1],
                )
            pred_losses.append(l_p)
            cov_losses.append(l_c)
            if target_spans_list:
                all_tgt_embs.append(target_spans_list[0].reshape(-1, x.shape[-1]))

        if not pred_losses:
            z = _zero_loss_connected(target_enc_out)
            self._batch_mon = {
                "std_dev_embeddings": 0.0, "rank_me": 0.0,
                "std_dev_values": _std_values, "std_dev_times": _std_times,
                "_N_input": N_model_pre, "_N_target": N_target_full,
                "_N_context": N_context_full,
                "avg_context_length": 0.0, "avg_target_span_length": 0.0,
                "_causal_cuts_skipped": S,
                "_causal_cuts_used": 0,
            }
            return z, z, z

        l_pred = torch.stack(pred_losses).mean()
        l_cov = torch.stack(cov_losses).mean()
        l_total = l_pred + self.config.lambda_cov * l_cov

        with torch.no_grad():
            if target_spans_list and target_span_pad_masks:
                rank_rows = _sample_target_rows_for_rank_me(
                    target_spans_list[0],
                    target_span_pad_masks[0],
                    self.config.rank_me_train_max_rows,
                )
                all_tgt = (
                    rank_rows
                    if rank_rows.shape[0] > 0
                    else target_spans_list[0].reshape(-1, x.shape[-1])
                )
                std_dev = all_tgt.std(dim=0).mean().item()
                every = self.config.rank_me_every_n_steps
                if every > 0 and self._cov_global_step % every == 0:
                    rank_me = _rank_me_from_rows(rank_rows)
                    self._last_rank_me = rank_me
                else:
                    rank_me = self._last_rank_me
            else:
                all_tgt = x.new_zeros((1, x.shape[-1]))
                std_dev = 0.0
                rank_me = self._last_rank_me

        avg_ctx_len = N_context_full / max(B_size_pre * S, 1)
        avg_tgt_area = N_target_full / max(B_size_pre * S, 1)

        self._batch_mon = {
            "std_dev_embeddings": std_dev,
            "rank_me": rank_me,
            "std_dev_values": _std_values,
            "std_dev_times": _std_times,
            "_N_input": N_model_pre,
            "_N_target": N_target_full,
            "_N_context": N_context_full,
            "avg_context_length": avg_ctx_len,
            "avg_target_span_length": avg_tgt_area,
            "_causal_cuts_used": n_cuts_used,
            "_tgt_embs_for_rank": (
                rank_rows.detach()
                if not self.training and target_spans_list and target_span_pad_masks
                else None
            ),
        }

        return l_pred, l_cov, l_total

    # ------------------------------------------------------------------
    # Shared extraction helpers
    # ------------------------------------------------------------------

    def _extract_context(
        self,
        x: torch.Tensor,
        all_context_indices: List[List[int]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Physically extract context tokens into a compact padded tensor.
        Preserves original integer positions for RoPE.

        Returns
        -------
        x_ctx:      (B, max_ctx_len, d)
        pos_ids:    (B, max_ctx_len) — original sequence positions
        ctx_mask:   (B, max_ctx_len) — 1=real, 0=pad
        """
        from torch.nn.utils.rnn import pad_sequence

        B, _, d = x.shape
        idx_tensors = [
            torch.tensor(ci, dtype=torch.long, device=device)
            for ci in all_context_indices
        ]
        lengths = torch.tensor(
            [t.numel() for t in idx_tensors], device=device, dtype=torch.long
        )
        max_ctx_len = max(int(lengths.max().item()), 1)
        idx_pad = pad_sequence(idx_tensors, batch_first=True, padding_value=0)
        idx_exp = idx_pad.unsqueeze(-1).expand(-1, -1, d)
        x_ctx = torch.gather(x, 1, idx_exp)
        pos_ids = idx_pad
        arange = torch.arange(max_ctx_len, device=device).unsqueeze(0)
        ctx_mask = (arange < lengths.unsqueeze(1)).to(torch.long)
        return x_ctx, pos_ids, ctx_mask

    def _extract_target_spans(
        self,
        encoder_out: torch.Tensor,
        all_target_spans: List[List[List[int]]],
        index_offset: int = 0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        For each span index s, build a padded tensor [B, max_span_len_s, d]
        containing the target encoder outputs at the span positions.

        Returns (span_tensors, pad_masks) where pad_masks[b, j] is True for
        padded positions (ignored by LatentCrossAttentionPool).
        """
        B, _L, d = encoder_out.shape
        device = encoder_out.device
        if not all_target_spans:
            return [], []
        positive_span_counts = [len(s) for s in all_target_spans if len(s) > 0]
        if not positive_span_counts:
            return [], []
        # Align with forward(): do not require batch row 0 to have a mask.
        num_spans = min(positive_span_counts)
        result: List[torch.Tensor] = []
        pad_masks: List[torch.Tensor] = []

        for s in range(num_spans):
            span_lens = [
                len(all_target_spans[b][s])
                for b in range(B)
                if s < len(all_target_spans[b])
            ]
            max_span_len = max(span_lens) if span_lens else 0
            if max_span_len == 0:
                continue

            span_t = encoder_out.new_zeros(B, max_span_len, d)
            pad_mask = torch.ones(B, max_span_len, dtype=torch.bool, device=device)
            for b in range(B):
                if s >= len(all_target_spans[b]):
                    continue
                span_idx = all_target_spans[b][s]
                n = len(span_idx)
                if n == 0:
                    continue
                idx_t = torch.tensor(span_idx, dtype=torch.long, device=device)
                span_t[b, :n] = encoder_out[b][idx_t + index_offset]
                pad_mask[b, :n] = False
            result.append(span_t)
            pad_masks.append(pad_mask)

        return result, pad_masks

    # ------------------------------------------------------------------
    # Branch A: Perceiver-JEPA
    # ------------------------------------------------------------------

    def _forward_perceiver(
        self,
        context_enc_out: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_spans_list: List[torch.Tensor],
        target_span_pad_masks: List[torch.Tensor],
        all_span_times: List[List[Tuple[float, float]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each span:
          - Pool target span → Z_tgt [B, 16, d]  (skip if span < min_span_for_perceiver)
          - Pool context → Z_ctx [B, 16, d]
          - Add temporal prompt + LayerNorm → Z_prompted
          - Predictor → Z_hat [B, 16, d]
          - L_pred += MSE(Z_hat, Z_tgt.detach())
          - L_cov  += CovReg(Z_tgt)
        Returns averaged (l_pred, l_cov).
        """
        B = context_enc_out.shape[0]
        device = context_enc_out.device
        num_spans = len(target_spans_list)
        min_span = self.config.min_span_for_perceiver

        # Pool context once — same for all spans
        Z_ctx = self.context_pooler(
            context_enc_out,
            key_padding_mask=(ctx_mask == 0),
        )  # (B, n_latents, d)

        pred_losses: List[torch.Tensor] = []
        cov_losses:  List[torch.Tensor] = []

        for s, span_tokens in enumerate(target_spans_list):
            span_len = span_tokens.shape[1]
            if span_len < min_span:
                continue

            pad_mask = (
                target_span_pad_masks[s]
                if s < len(target_span_pad_masks)
                else torch.zeros(
                    span_tokens.shape[0],
                    span_tokens.shape[1],
                    dtype=torch.bool,
                    device=span_tokens.device,
                )
            )

            # Target perceiver → (B, n_latents, d)
            Z_tgt = self.target_pooler(span_tokens, key_padding_mask=pad_mask)

            # Temporal prompt for this span
            coords = self._span_coords_for_span(all_span_times, s, device)  # (B, 2)
            prompt = self.prompt(coords.unsqueeze(1))   # (B, 1, d)
            Z_prompted = self.predictor.prompt_norm(
                Z_ctx + prompt
            )  # (B, n_latents, d)

            # Predictor → (B, n_latents, d)
            Z_hat = self.predictor.transformer(Z_prompted)

            # Projection heads (Linear + BN1d) break the LN unit-sphere constraint
            # and give the anti-collapse objective a free representation space.
            if self.target_proj is not None:
                Z_tgt = self.target_proj(Z_tgt)
            if self.pred_proj is not None:
                Z_hat = self.pred_proj(Z_hat)

            pred_losses.append(jepa_prediction_loss(Z_hat, Z_tgt))
            cov_losses.append(
                self.cov_loss(
                    Z_tgt,
                    global_step=self._cov_global_step,
                    sync_ddp=self._cov_sync_ddp,
                )
            )

        if not pred_losses:
            z = _zero_loss_connected(context_enc_out)
            return z, z

        l_pred = torch.stack(pred_losses).mean()
        l_cov  = torch.stack(cov_losses).mean()
        return l_pred, l_cov

    def _proj_token_targets(
        self, y_hat: torch.Tensor, y_tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linear+BN projection for Branch B (same role as perceiver proj heads)."""
        if self.target_proj is not None:
            y_tgt = self.target_proj(y_tgt)
        if self.pred_proj is not None:
            y_hat = self.pred_proj(y_hat)
        return y_hat, y_tgt

    # ------------------------------------------------------------------
    # Branch B: Token I-JEPA
    # ------------------------------------------------------------------

    def _scatter_compact_to_full(
        self,
        compact: torch.Tensor,
        pos_ids: torch.Tensor,
        compact_mask: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Place compact (B, N, d) tokens at original event indices in (B, L, d)."""
        B, _N, d = compact.shape
        device = compact.device
        full = compact.new_zeros(B, seq_len, d)
        full_attn = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        for b in range(B):
            m = compact_mask[b].bool()
            if not m.any():
                continue
            idx = pos_ids[b, m].long()
            full[b].index_copy_(0, idx, compact[b, m])
            full_attn[b].index_copy_(
                0, idx, torch.ones(idx.shape[0], dtype=torch.long, device=device)
            )
        return full, full_attn

    def _span_index_tensor(
        self,
        all_target_spans: List[List[List[int]]],
        span_idx: int,
        max_span_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Padded (B, max_span_len) event indices for span span_idx."""
        B = len(all_target_spans)
        idx = torch.zeros(B, max_span_len, dtype=torch.long, device=device)
        for b in range(B):
            span = all_target_spans[b][span_idx]
            n = min(len(span), max_span_len)
            if n > 0:
                idx[b, :n] = torch.tensor(span[:n], dtype=torch.long, device=device)
        return idx

    def _add_time_bias_to_context_scatter(
        self,
        x_full: torch.Tensor,
        ctx_pos_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        hours_since_first: torch.Tensor,
    ) -> None:
        """Add hours-since-first bias to scattered context encodings (in-place)."""
        B, nc_max = ctx_pos_ids.shape
        for b in range(B):
            nc = int(ctx_mask[b].sum().item())
            if nc == 0:
                continue
            pos = ctx_pos_ids[b, :nc]
            h = hours_since_first[b, pos]
            x_full[b, pos] = x_full[b, pos] + self.time_embed(h)

    def _place_mask_tokens_at_span(
        self,
        x_full: torch.Tensor,
        attn_full: torch.Tensor,
        all_target_spans: List[List[List[int]]],
        span_idx: int,
        hours_since_first: torch.Tensor,
        max_span_len: int,
    ) -> None:
        """Write learnable MASK (+ per-position hours-since-first bias) at targets."""
        B = x_full.shape[0]
        for b in range(B):
            span = all_target_spans[b][span_idx]
            n = min(len(span), max_span_len)
            if n == 0:
                continue
            for j in range(n):
                p = span[j]
                h = hours_since_first[b, p]
                x_full[b, p] = self.mask_token + self.time_embed(h)
                attn_full[b, p] = 1

    def _extract_span_outputs(
        self,
        full_out: torch.Tensor,
        all_target_spans: List[List[List[int]]],
        span_idx: int,
        max_span_len: int,
    ) -> torch.Tensor:
        """Gather predictor outputs at target span positions → (B, max_span_len, d)."""
        B, _, d = full_out.shape
        device = full_out.device
        idx = self._span_index_tensor(
            all_target_spans, span_idx, max_span_len, device
        )
        b_ix = torch.arange(B, device=device).unsqueeze(1).expand(B, max_span_len)
        return full_out[b_ix, idx]

    def _pack_target_delta_minutes(
        self,
        all_target_spans: List[List[List[int]]],
        all_delta_minutes: List[List[List[float]]],
        span_idx: int,
        max_len: int,
        device: torch.device,
        batch_rows: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """(B_rows, max_len) minutes since cut; zeros on pad / missing entries."""
        rows = (
            list(range(len(all_target_spans)))
            if batch_rows is None
            else batch_rows
        )
        B = len(rows)
        out = torch.zeros(B, max_len, device=device, dtype=torch.float32)
        for bi, b in enumerate(rows):
            span = all_target_spans[b][span_idx]
            n = min(len(span), max_len)
            if n == 0:
                continue
            deltas = (
                all_delta_minutes[b][span_idx]
                if span_idx < len(all_delta_minutes[b])
                else []
            )
            if deltas:
                n_copy = min(n, len(deltas))
                out[bi, :n_copy] = torch.as_tensor(
                    deltas[:n_copy], device=device, dtype=torch.float32
                )
        return out

    def _target_span_weights_from_delta_minutes(
        self,
        all_target_spans: List[List[List[int]]],
        all_delta_minutes: List[List[List[float]]],
        span_idx: int,
        max_span_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """W_j = max(floor, exp(-lambda * delta_minutes_j)); 0 on pad positions."""
        delta_t = self._pack_target_delta_minutes(
            all_target_spans,
            all_delta_minutes,
            span_idx,
            max_span_len,
            device,
        )
        span_lens = torch.tensor(
            [
                min(len(all_target_spans[b][span_idx]), max_span_len)
                for b in range(len(all_target_spans))
            ],
            device=device,
            dtype=torch.float32,
        )
        pos = torch.arange(max_span_len, device=device).unsqueeze(0)
        pad_mask = pos < span_lens.unsqueeze(1)
        return future_time_decay_weights(
            delta_t,
            self.config.future_time_decay_lambda,
            self.config.future_time_decay_weight_floor,
            mask=pad_mask,
        )

    def _forward_token_causal_single(
        self,
        context_enc_out: torch.Tensor,
        ctx_pos_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_spans_list: List[torch.Tensor],
        all_target_spans: List[List[List[int]]],
        hours_since_first: torch.Tensor,
        all_target_delta_minutes: Optional[List[List[List[float]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Branch B for causal_single: compact [CLS | context_enc | MASK@targets].

        Target encoder runs on the full sequence (y_tgt).  Context prefix is
        re-encoded; pretrained CLS + learnable MASK tokens (RoPE at event
        positions; CLS at position 0; + hours-since-first additive bias) feed
        the token predictor.  Quadrant
        attention: CLS/context cannot attend to MASK slots; MASK slots may
        attend to CLS and context.  Only batch rows with non-empty context
        and enough target tokens are included.

        Downstream probing still uses encoder [CLS | events] (target pathway);
        predictor CLS is trained with context-only visibility to future MASKs.
        """
        if not target_spans_list:
            self._dbg(zero_path="causal_single_no_target_spans_list")
            z = _zero_loss_connected(context_enc_out)
            return z, z

        y_tgt_full = target_spans_list[0]
        B, _, d = y_tgt_full.shape
        device = context_enc_out.device
        span_idx = 0
        min_tgt = self.config.min_target_events

        ctx_lens = ctx_mask.sum(dim=1)
        tgt_lens = torch.tensor(
            [
                len(all_target_spans[b][span_idx])
                if span_idx < len(all_target_spans[b])
                else 0
                for b in range(B)
            ],
            device=device,
            dtype=torch.long,
        )
        lengths_c = ctx_lens.tolist()
        lengths_t = tgt_lens.tolist()
        valid_mask = (ctx_lens > 0) & (tgt_lens >= min_tgt)
        valid_bs = valid_mask.nonzero(as_tuple=False).flatten().tolist()

        if not valid_bs:
            self._dbg(
                zero_path="causal_single_no_valid_bs",
                n_valid_bs=0,
                B=B,
                min_tgt_required=min_tgt,
                lengths_t_sample=lengths_t[: min(8, len(lengths_t))],
                lengths_c_sample=lengths_c[: min(8, len(lengths_c))],
            )
            z = _zero_loss_connected(context_enc_out)
            return z, z

        avg_nc = sum(lengths_c[b] for b in valid_bs) / len(valid_bs)
        avg_nt = sum(lengths_t[b] for b in valid_bs) / len(valid_bs)
        self._dbg(
            n_valid_bs=len(valid_bs),
            B_eff=len(valid_bs),
            avg_nc=round(avg_nc, 2),
            avg_nt=round(avg_nt, 2),
            min_tgt_required=min_tgt,
        )

        max_compact = max(1 + lengths_c[b] + lengths_t[b] for b in valid_bs)
        max_n_tgt = max(lengths_t[b] for b in valid_bs)
        B_eff = len(valid_bs)

        x_cat = context_enc_out.new_zeros(B_eff, max_compact, d)
        pos_cat = torch.zeros(B_eff, max_compact, dtype=torch.long, device=device)
        attn_cat = torch.zeros(B_eff, max_compact, dtype=torch.long, device=device)
        hours_batch = torch.zeros(B_eff, max_compact, device=device, dtype=torch.float)

        for bi, b in enumerate(valid_bs):
            nc, nt = lengths_c[b], lengths_t[b]
            compact_len = 1 + nc + nt
            hours_batch[bi, 0] = 0.0
            if nc > 0:
                hours_batch[bi, 1 : 1 + nc] = hours_since_first[b, ctx_pos_ids[b, :nc]]
            tgt_idx = all_target_spans[b][span_idx]
            if nt > 0:
                tgt_pos = torch.tensor(tgt_idx[:nt], device=device, dtype=torch.long)
                hours_batch[bi, 1 + nc : compact_len] = hours_since_first[b, tgt_pos]

            x_cat[bi, 0] = self.cls_token
            pos_cat[bi, 0] = 0
            attn_cat[bi, 0] = 1
            if nc > 0:
                x_cat[bi, 1 : 1 + nc] = context_enc_out[b, :nc]
                pos_cat[bi, 1 : 1 + nc] = ctx_pos_ids[b, :nc]
                attn_cat[bi, 1 : 1 + nc] = 1
            if nt > 0:
                x_cat[bi, 1 + nc : compact_len] = self.mask_token
                pos_cat[bi, 1 + nc : compact_len] = tgt_pos
                attn_cat[bi, 1 + nc : compact_len] = 1

        time_bias = self.time_embed(hours_batch.reshape(-1)).view(B_eff, max_compact, d)
        x_cat = x_cat + time_bias * attn_cat.unsqueeze(-1).to(x_cat.dtype)

        if not torch.isfinite(x_cat).all():
            hsf_finite = bool(torch.isfinite(hours_since_first).all())
            self._dbg(
                zero_path="causal_single_nonfinite_x_cat",
                hours_since_first_finite=hsf_finite,
            )
            z = _zero_loss_connected(context_enc_out)
            return z, z

        attn_mode = self.config.causal_single_predictor_attn
        if attn_mode not in ("bidirectional", "quadrant", "partial_causal"):
            raise ValueError(
                f"causal_single_predictor_attn must be 'bidirectional', 'quadrant', "
                f"or 'partial_causal', got {attn_mode!r}"
            )

        attn_bias = None
        if attn_mode in ("quadrant", "partial_causal"):
            lc = [lengths_c[b] for b in valid_bs]
            lt = [lengths_t[b] for b in valid_bs]
            build_mask = (
                build_causal_single_partial_causal_mask_batch
                if attn_mode == "partial_causal"
                else build_causal_single_quadrant_mask_batch
            )
            attn_bias = build_mask(
                lc, lt, max_compact, include_cls=True, device=device
            )

        out = self.token_predictor(
            x_cat,
            attention_mask=attn_cat,
            position_ids=pos_cat,
            attn_bias=attn_bias,
        )

        y_hat = y_tgt_full.new_zeros(B_eff, max_n_tgt, d)
        y_tgt = y_tgt_full.new_zeros(B_eff, max_n_tgt, d)
        token_mask = torch.zeros(B_eff, max_n_tgt, device=device)
        loss_weights = torch.zeros(B_eff, max_n_tgt, device=device)

        use_time_decay = (
            self.config.future_time_decay_lambda > 0
            and all_target_delta_minutes is not None
        )

        for bi, b in enumerate(valid_bs):
            nc, nt = lengths_c[b], lengths_t[b]
            y_hat[bi, :nt] = out[bi, 1 + nc : 1 + nc + nt]
            y_tgt[bi, :nt] = y_tgt_full[b, :nt]
            token_mask[bi, :nt] = 1.0

        if use_time_decay:
            delta_t = self._pack_target_delta_minutes(
                all_target_spans,
                all_target_delta_minutes,
                span_idx,
                max_n_tgt,
                device,
                batch_rows=valid_bs,
            )
            loss_weights = future_time_decay_weights(
                delta_t,
                self.config.future_time_decay_lambda,
                self.config.future_time_decay_weight_floor,
                mask=token_mask,
            )

        if not torch.isfinite(y_hat).all() or not torch.isfinite(y_tgt).all():
            self._dbg(zero_path="causal_single_nonfinite_y_hat_y_tgt")
            z = _zero_loss_connected(context_enc_out)
            return z, z

        y_hat, y_tgt = self._proj_token_targets(y_hat, y_tgt)

        if use_time_decay:
            l_pred = jepa_prediction_loss_token_masked(
                y_hat, y_tgt, token_mask, weights=loss_weights
            )
        else:
            l_pred = jepa_prediction_loss_token_masked(y_hat, y_tgt, token_mask)

        if not torch.isfinite(l_pred):
            self._dbg(zero_path="causal_single_nonfinite_l_pred")
            z = _zero_loss_connected(context_enc_out)
            return z, z

        n_cov = int(token_mask.sum().item())
        if n_cov < 2:
            l_cov = y_tgt.sum() * 0.0
            self._dbg(zero_path="causal_single_cov_skipped", n_cov_tokens=n_cov)
        else:
            y_cov = y_tgt[token_mask.bool()]
            l_cov = self.cov_loss(
                y_cov,
                global_step=self._cov_global_step,
                sync_ddp=self._cov_sync_ddp,
            )
        self._dbg(
            causal_l_pred=float(l_pred.detach().item()),
            causal_l_cov=float(l_cov.detach().item()),
            n_cov_tokens=n_cov,
            token_weight_sum=float(loss_weights.sum().item()) if use_time_decay else float(token_mask.sum().item()),
        )
        return l_pred, l_cov

    def _forward_token(
        self,
        context_enc_out: torch.Tensor,
        ctx_pos_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_spans_list: List[torch.Tensor],
        all_target_spans: List[List[List[int]]],
        hours_since_first: torch.Tensor,
        seq_len: int,
        all_target_delta_minutes: Optional[List[List[List[float]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each span (span_budget etc.):
          - Scatter context encodings to context indices in (B, L, d)
          - Place MASK tokens at target indices (learnable token + hours-since-first bias)
          - Token predictor on full length L with original RoPE positions
          - Slice outputs at target indices → Y_hat; L_pred vs Y_tgt
        Returns averaged (l_pred, l_cov).
        """
        B, _, d = context_enc_out.shape
        device = context_enc_out.device

        pred_losses: List[torch.Tensor] = []
        cov_losses:  List[torch.Tensor] = []

        x_ctx_full, ctx_attn = self._scatter_compact_to_full(
            context_enc_out, ctx_pos_ids, ctx_mask, seq_len
        )
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

        self._add_time_bias_to_context_scatter(
            x_ctx_full, ctx_pos_ids, ctx_mask, hours_since_first
        )

        for s, y_tgt in enumerate(target_spans_list):
            N_span = y_tgt.shape[1]

            x_in = x_ctx_full.clone()
            attn_in = ctx_attn.clone()
            self._place_mask_tokens_at_span(
                x_in, attn_in, all_target_spans, s, hours_since_first, N_span
            )

            out = self.token_predictor(
                x_in, attention_mask=attn_in, position_ids=pos_ids
            )  # (B, L, d)
            y_hat = self._extract_span_outputs(
                out, all_target_spans, s, N_span
            )
            y_hat, y_tgt = self._proj_token_targets(y_hat, y_tgt)

            use_time_decay = (
                self.config.future_time_decay_lambda > 0
                and self.config.masking_strategy == "causal_single"
                and all_target_delta_minutes is not None
            )
            if use_time_decay:
                weights = self._target_span_weights_from_delta_minutes(
                    all_target_spans,
                    all_target_delta_minutes,
                    s,
                    N_span,
                    device,
                )
                pred_losses.append(
                    jepa_prediction_loss_weighted(y_hat, y_tgt, weights)
                )
            else:
                pred_losses.append(jepa_prediction_loss(y_hat, y_tgt))
            cov_losses.append(
                self.cov_loss(
                    y_tgt,
                    global_step=self._cov_global_step,
                    sync_ddp=self._cov_sync_ddp,
                )
            )

        if not pred_losses:
            z = _zero_loss_connected(context_enc_out)
            return z, z

        l_pred = torch.stack(pred_losses).mean()
        l_cov  = torch.stack(cov_losses).mean()
        return l_pred, l_cov

    # ------------------------------------------------------------------
    # Coordinate / position helpers
    # ------------------------------------------------------------------

    def _span_coords_for_span(
        self,
        all_span_times: List[List[Tuple[float, float]]],
        span_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build (B, 2) tensor of (midpoint, duration) for span span_idx."""
        B = len(all_span_times)
        coords = torch.zeros(B, 2, device=device)
        for b, span_times in enumerate(all_span_times):
            if span_idx < len(span_times):
                coords[b, 0] = span_times[span_idx][0]
                coords[b, 1] = span_times[span_idx][1]
        return coords

    def _span_pos_ids(
        self,
        all_target_spans: List[List[List[int]]],
        span_idx: int,
        max_span_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build (B, max_span_len) original position IDs for span span_idx."""
        B = len(all_target_spans)
        pos = torch.zeros(B, max_span_len, dtype=torch.long, device=device)
        for b in range(B):
            span = all_target_spans[b][span_idx]
            n = min(len(span), max_span_len)
            if n > 0:
                pos[b, :n] = torch.tensor(span[:n], dtype=torch.long, device=device)
        return pos

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _build_scheduler(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
    ) -> Optional[object]:
        cfg = self.config
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

        if cfg.scheduler == "none":
            return None

        if cfg.scheduler == "cosine_warmup":
            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if cfg.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=cfg.lr * cfg.min_lr_ratio
            )

        if cfg.scheduler == "linear_warmup":
            def lr_lambda_linear(step: int) -> float:
                if step < warmup_steps:
                    return step / warmup_steps
                return 1.0
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_linear)

        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")

    # ------------------------------------------------------------------
    # Inline linear-probe evaluation
    # ------------------------------------------------------------------

    def _run_inline_probe(
        self,
        probe_train_loader: DataLoader,
        probe_val_loader: Optional[DataLoader],
        n_epochs: int,
        lr: float,
        dropout: float,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> Dict[str, float]:
        """
        Train a fresh LinearProbe on top of the current (frozen) encoder weights
        and return the final validation metrics.

        When world_size > 1 (DDP active) all ranks participate:
          - Training data is sharded via DistributedSampler.
          - LinearProbe gradients are all-reduced via DDP.
          - After training, each rank evaluates its val shard; predictions are
            gathered to rank 0 before AUROC/AUPR are computed.
          - Only rank 0 returns a populated metrics dict; other ranks return {}.

        When world_size == 1 the existing single-GPU path (train_linear_probe)
        is used unchanged.
        """
        import gc
        import torch.nn as _nn
        from evaluation.linear_probe import LinearProbe, build_frozen_jepa_encoder
        from models.sequence_pooling import parse_pooling_mode

        is_dist = (world_size > 1)
        probe_pooling = parse_pooling_mode(self.config.probe_pooling)

        encoder = build_frozen_jepa_encoder(
            embedding=self.embedding,
            encoder=self.encoder,
            context_pooler=self.context_pooler,
            cls_token=self.cls_token,
            pooling_mode=probe_pooling,
        ).to(device)

        probe = LinearProbe(encoder.output_dim, dropout=dropout).to(device)

        # ---- Single-GPU fast path ------------------------------------------------
        if not is_dist:
            from evaluation.linear_probe import train_linear_probe
            history, final_val = train_linear_probe(
                encoder=encoder, probe=probe,
                train_loader=probe_train_loader, val_loader=probe_val_loader,
                n_epochs=n_epochs, lr=lr, device=str(device), verbose=True,
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

        # ---- Distributed path ---------------------------------------------------
        import torch.distributed as _dist
        from torch.nn.parallel import DistributedDataParallel as _DDP
        from torch.utils.data import DataLoader as _DL, DistributedSampler as _DS
        from evaluation.linear_probe import _compute_all_metrics

        def _rebuild_loader(loader: DataLoader, sampler) -> DataLoader:
            """Recreate a DataLoader with the given sampler (preserves all other kwargs)."""
            return _DL(
                loader.dataset,
                batch_size=loader.batch_size,
                sampler=sampler,
                collate_fn=loader.collate_fn,
                num_workers=loader.num_workers,
                pin_memory=loader.pin_memory,
                persistent_workers=loader.num_workers > 0,
                prefetch_factor=2 if loader.num_workers > 0 else None,
            )

        train_sampler = _DS(probe_train_loader.dataset, num_replicas=world_size,
                            rank=rank, shuffle=True, drop_last=True)
        dist_train_loader = _rebuild_loader(probe_train_loader, train_sampler)

        # Wrap linear probe with DDP for gradient synchronisation.
        # The encoder is frozen — it runs independently on each rank with no
        # need for all-reduce.
        ddp_probe = _DDP(probe, device_ids=[device.index], output_device=device.index)

        criterion = _nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)

        # Build distributed val loader once (reused every epoch for per-epoch val)
        dist_val_loader = None
        if probe_val_loader is not None:
            val_sampler = _DS(probe_val_loader.dataset, num_replicas=world_size,
                              rank=rank, shuffle=False, drop_last=False)
            dist_val_loader = _rebuild_loader(probe_val_loader, val_sampler)

        combined_out: Dict[str, float] = {}
        final_train_m: Dict[str, float] = {}

        for epoch in range(n_epochs):
            # ---- Training ----
            train_sampler.set_epoch(epoch)
            ddp_probe.train()
            epoch_loss, epoch_logits, epoch_labels = 0.0, [], []

            for batch in dist_train_loader:
                codes       = batch["codes"].to(device, non_blocking=True)
                attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
                labels      = batch["labels"].float().to(device, non_blocking=True)
                values      = batch["values"].to(device, non_blocking=True)      if "values"      in batch else None
                z_scores    = batch["z_scores"].to(device, non_blocking=True)    if "z_scores"    in batch else None
                delta_times = batch["delta_times"].to(device, non_blocking=True) if "delta_times" in batch else None
                value_mask  = batch["value_mask"].to(device, non_blocking=True)  if "value_mask"  in batch else None

                with torch.no_grad():
                    z = encoder(codes, attn_mask, values, z_scores, delta_times, value_mask)
                logits = ddp_probe(z)
                loss   = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()   # DDP all-reduces gradients automatically
                optimizer.step()

                epoch_loss += loss.item()
                epoch_logits.append(logits.detach().cpu())
                epoch_labels.append(labels.cpu())

            avg_train_loss = epoch_loss / max(len(dist_train_loader), 1)
            train_m = _compute_all_metrics(torch.cat(epoch_labels), torch.cat(epoch_logits))
            # Store final-epoch train metrics for the return dict
            final_train_m = {
                "train_loss":      avg_train_loss,
                **{f"train_{k}": v for k, v in train_m.items()},
            }

            # ---- Per-epoch val evaluation (all ranks forward, gather on rank 0) ----
            ddp_probe.eval()
            val_line = ""
            if dist_val_loader is not None:
                local_logits: List[torch.Tensor] = []
                local_labels: List[torch.Tensor] = []

                with torch.no_grad():
                    for batch in dist_val_loader:
                        codes       = batch["codes"].to(device, non_blocking=True)
                        attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
                        labels      = batch["labels"].float().to(device, non_blocking=True)
                        values      = batch["values"].to(device, non_blocking=True)      if "values"      in batch else None
                        z_scores    = batch["z_scores"].to(device, non_blocking=True)    if "z_scores"    in batch else None
                        delta_times = batch["delta_times"].to(device, non_blocking=True) if "delta_times" in batch else None
                        value_mask  = batch["value_mask"].to(device, non_blocking=True)  if "value_mask"  in batch else None

                        z      = encoder(codes, attn_mask, values, z_scores, delta_times, value_mask)
                        logits = probe(z)   # unwrapped probe — no DDP overhead in eval
                        local_logits.append(logits.cpu())
                        local_labels.append(labels.cpu())

                local_logits_t = torch.cat(local_logits)
                local_labels_t = torch.cat(local_labels)

                # Exchange sizes so every rank knows how many samples each other has
                local_size = torch.tensor([local_logits_t.shape[0]], dtype=torch.long, device=device)
                all_sizes  = [torch.zeros(1, dtype=torch.long, device=device)
                              for _ in range(world_size)]
                _dist.all_gather(all_sizes, local_size)
                max_size = int(max(s.item() for s in all_sizes))

                def _pad(t: torch.Tensor) -> torch.Tensor:
                    buf = t.new_zeros(max_size).to(device)
                    buf[:t.shape[0]] = t.to(device)
                    return buf

                gathered_logits = [torch.zeros(max_size, device=device) for _ in range(world_size)]
                gathered_labels = [torch.zeros(max_size, device=device) for _ in range(world_size)]
                _dist.all_gather(gathered_logits, _pad(local_logits_t))
                _dist.all_gather(gathered_labels, _pad(local_labels_t))

                if rank == 0:
                    all_logits = torch.cat([gathered_logits[i][:int(all_sizes[i].item())]
                                            for i in range(world_size)]).cpu()
                    all_labels = torch.cat([gathered_labels[i][:int(all_sizes[i].item())]
                                            for i in range(world_size)]).cpu()
                    val_m    = _compute_all_metrics(all_labels, all_logits)
                    val_loss = criterion(all_logits, all_labels).item()
                    combined_out = {"val_loss": val_loss,
                                    **{f"val_{k}": v for k, v in val_m.items()}}
                    val_line = (f"  val_loss={val_loss:.4f}"
                                f"  val_auroc={val_m['auroc']:.4f}"
                                f"  val_aupr={val_m['aupr']:.4f}")

            if rank == 0:
                print(f"  probe epoch {epoch+1}/{n_epochs}  "
                      f"loss={avg_train_loss:.4f}  auroc={train_m['auroc']:.4f}"
                      f"{val_line}")

            ddp_probe.train()  # restore train mode for next epoch

        del encoder, ddp_probe, probe
        gc.collect()
        # Merge final-epoch train metrics into val metrics for the return dict.
        # combined_out is populated on rank 0; final_train_m is local (rank 0's shard)
        # but is fine for monitoring purposes.
        combined_out.update(final_train_m)
        return combined_out  # populated on rank 0, empty dict on other ranks

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
        # ---- Inline probe evaluation ----
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
        rank: int = 0,
        world_size: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Training loop with LR scheduling, gradient clipping, and early stopping.

        Parameters
        ----------
        train_loader:     DataLoader for training data.
        val_loader:       Optional DataLoader for validation.
        optimizer:        Defaults to AdamW(weight_decay=config.weight_decay).
        on_epoch_end:     Callback(epoch, metrics_dict) — called on main process only.
        on_batch_end:     Callback(epoch, global_step, metrics_dict) — main process only.
        ddp_module:       When running under DDP, pass the DDP-wrapped version of self.
                          The forward pass is routed through it so gradients are
                          all-reduced across processes.  When None, self.forward() is
                          used directly (single-GPU / DataParallel).
        is_main_process:  True for rank-0 only.  Controls printing, checkpointing,
                          and W&B callbacks.
        train_sampler:    DistributedSampler (or None).  When provided, its set_epoch()
                          is called at the start of each epoch to reshuffle per-rank data.

        Returns
        -------
        History dict: train_loss, val_loss, lr (one per epoch), stopped_early.
        """
        cfg = self.config

        # Route the forward pass through the DDP wrapper when available so
        # NCCL all-reduces gradients across processes.  Fall back to self.
        _forward = ddp_module if ddp_module is not None else self

        if optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

        device = torch.device(cfg.device)
        self.to(device)

        total_steps = cfg.n_epochs * len(train_loader)
        scheduler   = self._build_scheduler(optimizer, total_steps)

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "lr": []
        }

        es_higher = _early_stopping_higher_is_better(cfg.early_stopping_metric)
        best_metric        = -float("inf") if es_higher else float("inf")
        best_ckpt_metric   = float("inf")   # tracks best val/train loss for checkpointing
        best_probe_auroc   = -float("inf")  # tracks best probe val_auroc for probe_best.pt
        patience_left      = cfg.early_stopping_patience
        stopped_early      = False

        # Set up checkpoint directory once so the path is always available
        ckpt_dir = cfg.checkpoint_dir.strip() if cfg.checkpoint_dir else ""
        if ckpt_dir and is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"[train] Checkpoints:     {ckpt_dir}")

        use_amp = cfg.use_amp and device.type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if cfg.amp_dtype in ("bf16", "bfloat16")
            else torch.float16
        )
        use_grad_scaler = use_amp and amp_dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
        log_every = max(1, cfg.wandb_log_every_n_steps)

        if is_main_process:
            branch = "Perceiver (A)" if cfg.use_perceiver else "Token I-JEPA (B)"
            print(f"[train] Branch:          {branch}")
            print(f"[train] Scheduler:       {cfg.scheduler}")
            print(f"[train] Grad clip:       {cfg.grad_clip if cfg.grad_clip > 0 else 'disabled'}")
            print(f"[train] Weight decay:    {cfg.weight_decay}")
            print(f"[train] lambda_cov:      {cfg.lambda_cov}")
            print(f"[train] Early stopping:  "
                  f"{'disabled' if cfg.early_stopping_patience == 0 else f'patience={cfg.early_stopping_patience}, metric={cfg.early_stopping_metric}'}")
            if cfg.debug_jepa or cfg.debug_jepa_on_zero_loss:
                print(
                    f"[train] JEPA debug:      on (first_batches={cfg.debug_jepa_first_batches}, "
                    f"every_n_steps={cfg.debug_jepa_every_n_steps}, on_zero_loss={cfg.debug_jepa_on_zero_loss})"
                )
            print(
                f"[train] AMP:             "
                f"{'off' if not use_amp else amp_dtype}  "
                f"(grad_scaler={use_grad_scaler})"
            )
            print(f"[train] W&B batch log:   every {log_every} step(s)")
            print()

        global_step = 0
        for epoch in range(cfg.n_epochs):
            # Reshuffle each rank's data slice independently each epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.train()
            if ddp_module is not None:
                ddp_module.train()
            epoch_loss   = 0.0
            epoch_l_pred = 0.0
            epoch_l_cov  = 0.0
            n_batches    = 0

            import time as _time

            for batch in train_loader:
                t0 = _time.perf_counter()

                codes       = batch["codes"].to(device, non_blocking=True)
                attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
                values      = batch.get("values")
                z_scores    = batch.get("z_scores")
                delta_times = batch.get("delta_times")
                value_mask  = batch.get("value_mask")
                hours_since_first = batch.get("hours_since_first")

                if values      is not None: values      = values.to(device, non_blocking=True)
                if z_scores    is not None: z_scores    = z_scores.to(device, non_blocking=True)
                if delta_times is not None: delta_times = delta_times.to(device, non_blocking=True)
                if value_mask  is not None: value_mask  = value_mask.to(device, non_blocking=True)
                if hours_since_first is not None:
                    hours_since_first = hours_since_first.to(device, non_blocking=True)

                # Pre-computed masking from the DataLoader worker (plain Python
                # lists — they stay on CPU, never move to device).
                pre_mask = _pre_mask_dict_from_batch(batch)

                orig_seq_lengths = batch.get("orig_seq_lengths")

                # SIGReg: same projection directions on every GPU (seed = step index).
                self._cov_global_step = global_step

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    l_pred, l_cov, l_total = _forward(
                        codes,
                        attn_mask,
                        values,
                        z_scores,
                        delta_times,
                        value_mask,
                        hours_since_first=hours_since_first,
                        pre_mask=pre_mask,
                    )
                if is_main_process:
                    self._log_jepa_debug(
                        global_step=global_step,
                        epoch=epoch,
                        batch_in_epoch=n_batches,
                        l_pred=l_pred.item(),
                        l_cov=l_cov.item(),
                        l_total=l_total.item(),
                    )
                if is_main_process and not (
                    torch.isfinite(l_pred) and torch.isfinite(l_cov) and torch.isfinite(l_total)
                ):
                    print(
                        f"[train][WARN] non-finite loss at step={global_step}: "
                        f"l_pred={float(l_pred.detach().float().cpu())} "
                        f"l_cov={float(l_cov.detach().float().cpu())} "
                        f"l_total={float(l_total.detach().float().cpu())} "
                        f"lambda_cov={cfg.lambda_cov}",
                        flush=True,
                    )
                if l_total.requires_grad:
                    if use_grad_scaler:
                        scaler.scale(l_total).backward()
                    else:
                        l_total.backward()
                elif is_main_process:
                    print(
                        f"[train][WARN] step={global_step}: loss has no grad "
                        f"(all spans/cuts skipped?) — skipping backward",
                        flush=True,
                    )

                grad_norm = 0.0
                if l_total.requires_grad:
                    if use_grad_scaler:
                        scaler.unscale_(optimizer)
                    grad_norm = _total_grad_norm(self.parameters())
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
                    if use_grad_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                elapsed = _time.perf_counter() - t0
                batch_size = codes.shape[0]

                current_lr    = optimizer.param_groups[0]["lr"]
                batch_loss    = l_total.item()
                epoch_loss   += batch_loss
                epoch_l_pred += l_pred.item()
                epoch_l_cov  += l_cov.item()
                n_batches    += 1
                # Fractional epoch: e.g. 1.45 = 45% through epoch 2
                total_batches  = len(train_loader)
                epoch_progress = epoch + n_batches / max(total_batches, 1)
                global_step  += 1

                if (
                    is_main_process
                    and on_batch_end is not None
                    and (global_step % log_every == 0 or global_step == 1)
                ):
                    batch_metrics = {
                        "epoch":       epoch_progress,
                        # --- Panel: Loss Components ---
                        "loss_total":  batch_loss,
                        "loss_pred":   l_pred.item(),
                        "loss_cov":    l_cov.item(),
                        # --- Panel: Optimization & Hardware ---
                        "learning_rate":       current_lr,
                        "grad_norm":           grad_norm,
                        "samples_per_second":  batch_size / max(elapsed, 1e-6),
                        # --- Panel: Representation Health ---
                        "std_dev_embeddings":  self._batch_mon.get("std_dev_embeddings", 0.0),
                        "rank_me":             self._batch_mon.get("rank_me", 0.0),
                        "rank_me_updated":     float(
                            self._batch_mon.get("rank_me_updated", 0.0)
                        ),
                        "std_dev_values":      self._batch_mon.get("std_dev_values", 0.0),
                        "std_dev_times":       self._batch_mon.get("std_dev_times", 0.0),
                        # --- Panel: Medical Context ---
                        # N_total: original trajectory length (pre-windowing)
                        # N_context / N_target: computed after masking
                        "avg_seq_length": (
                            orig_seq_lengths.float().mean().item()
                            if orig_seq_lengths is not None
                            else attn_mask.sum(1).float().mean().item()
                        ),
                        "avg_context_length":    self._batch_mon.get("avg_context_length", 0.0),
                        "avg_target_span_length": self._batch_mon.get("avg_target_span_length", 0.0),
                        # Definitions (batch-level sums):
                        #   N_input    = real tokens after windowing/padding
                        #   N_targets  = masked tokens
                        #   N_context  = N_input - N_targets
                        #
                        #   target_ratio  = N_targets / N_input
                        #   context_ratio = N_context / N_full_sequence
                        "target_ratio": (
                            self._batch_mon.get("_N_target", 0)
                            / max(self._batch_mon.get("_N_input", 1), 1)
                        ),
                        "context_ratio": (
                            self._batch_mon.get("_N_context", 0)
                            / max(
                                int(orig_seq_lengths.sum().item()) if orig_seq_lengths is not None
                                else self._batch_mon.get("_N_input", 1),
                                1,
                            )
                        ),
                    }
                    if self.config.masking_strategy == "causal_single":
                        for key in (
                            "causal_valid_mask_fraction",
                            "causal_cut_position_ratio",
                            "causal_cut_over_context_index_span",
                            "causal_context_token_fraction",
                            "causal_target_token_fraction",
                        ):
                            if key in self._batch_mon:
                                batch_metrics[key] = self._batch_mon[key]
                    on_batch_end(epoch, global_step, batch_metrics)

            avg_train  = epoch_loss   / max(n_batches, 1)
            avg_l_pred = epoch_l_pred / max(n_batches, 1)
            avg_l_cov  = epoch_l_cov  / max(n_batches, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train)
            history["lr"].append(current_lr)

            epoch_metrics: Dict[str, float] = {
                "global_step": global_step,
            }

            val_line = ""
            if val_loader is not None:
                if is_main_process:
                    print(f"  [val] Running validation … ", end="", flush=True)
                avg_val, val_metrics = self._eval_epoch(
                    val_loader, device, rank=rank, world_size=world_size
                )
                if is_main_process:
                    print(
                        f"val_loss={avg_val:.4f}"
                        f"  std_dev={val_metrics.get('std_dev_embeddings', 0.0):.4f}"
                        f"  rank_me={val_metrics.get('rank_me', 0.0):.1f}"
                    )
                    history["val_loss"].append(avg_val)
                    epoch_metrics["val_loss"] = avg_val
                    epoch_metrics.update(val_metrics)
                    val_line = f"  val={avg_val:.4f}"

            if is_main_process:
                print(
                    f"Epoch {epoch+1}/{cfg.n_epochs}  "
                    f"train={avg_train:.4f}  (pred={avg_l_pred:.4f}, cov={avg_l_cov:.4f})"
                    f"{val_line}  lr={current_lr:.2e}"
                )
                if on_epoch_end is not None:
                    on_epoch_end(epoch, epoch_metrics)

            # Sync after validation / logging so rank 0 cannot still be in val
            # while other ranks start the inline probe (was: barrier before val ended).
            if ddp_module is not None:
                import torch.distributed as _dist
                _dist.barrier()

            # ---- Inline probe evaluation (all ranks when DDP is active) --------
            # Run on epoch 1 unconditionally, then every probe_interval epochs.
            _run_probe_this_epoch = (
                inline_probe_during_pretrain
                and probe_train_loader is not None
                and ((epoch + 1) == 1 or (epoch + 1) % max(1, probe_interval) == 0)
            )
            probe_metrics: Dict[str, float] = {}
            if _run_probe_this_epoch:
                import time as _probe_t
                if is_main_process:
                    print(f"  [probe] Running inline linear probe for epoch {epoch+1} …")
                _pt0 = _probe_t.perf_counter()
                probe_metrics = self._run_inline_probe(
                    probe_train_loader=probe_train_loader,
                    probe_val_loader=probe_val_loader,
                    n_epochs=probe_n_epochs,
                    lr=probe_lr,
                    dropout=probe_dropout,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                )
                if is_main_process:
                    probe_runtime = _probe_t.perf_counter() - _pt0
                    probe_metrics["runtime_s"] = probe_runtime
                    print(
                        f"  [probe] Done in {probe_runtime:.1f}s  "
                        f"train_loss={probe_metrics.get('train_loss', 0):.4f}  "
                        f"val_loss={probe_metrics.get('val_loss', 0):.4f}  "
                        f"val_auroc={probe_metrics.get('val_auroc', 0):.4f}  "
                        f"val_aupr={probe_metrics.get('val_aupr', 0):.4f}"
                    )
                    if on_epoch_end is not None:
                        on_epoch_end(
                            epoch,
                            {"global_step": global_step,
                             **{f"probe_{k}": v for k, v in probe_metrics.items()}},
                        )

            # ---- Checkpointing (rank 0 only — uses gathered probe_metrics) -----
            if is_main_process and ckpt_dir:
                ckpt_monitor = epoch_metrics.get("val_loss", avg_train)
                ckpt_payload = {
                    "epoch":        epoch + 1,
                    "global_step":  global_step,
                    "val_loss":     epoch_metrics.get("val_loss", None),
                    "train_loss":   avg_train,
                    "model_state":  self.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                torch.save(ckpt_payload, os.path.join(ckpt_dir, "last.pt"))
                save_jepa_split_checkpoints(ckpt_payload["model_state"], ckpt_dir)
                if ckpt_monitor < best_ckpt_metric:
                    best_ckpt_metric = ckpt_monitor
                    torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pt"))
                    print(f"  [ckpt] Saved best.pt  (monitor={ckpt_monitor:.4f})")
                probe_auroc = probe_metrics.get("val_auroc", None) if _run_probe_this_epoch else None
                if probe_auroc is not None and probe_auroc > best_probe_auroc:
                    best_probe_auroc = probe_auroc
                    torch.save(ckpt_payload, os.path.join(ckpt_dir, "probe_best.pt"))
                    print(f"  [ckpt] Saved probe_best.pt  (val_auroc={probe_auroc:.4f})")

            # ---- Barrier 3: before next epoch ----------------------------------
            # Rank 0 may still be checkpointing (fast but non-zero).  Ensure all
            # ranks start epoch N+1 together before DDP's next forward/all-reduce.
            if ddp_module is not None:
                _dist.barrier()

            # Early stopping is evaluated on the main process (which holds
            # val metrics).  Other ranks just continue their training loop.
            if is_main_process and cfg.early_stopping_patience > 0:
                monitor = epoch_metrics.get(cfg.early_stopping_metric)
                if monitor is None:
                    pass
                elif _early_stopping_improved(monitor, best_metric, es_higher):
                    best_metric   = monitor
                    patience_left = cfg.early_stopping_patience
                else:
                    patience_left -= 1
                    print(
                        f"  [early stopping] No improvement in "
                        f"'{cfg.early_stopping_metric}' for "
                        f"{cfg.early_stopping_patience - patience_left}/"
                        f"{cfg.early_stopping_patience} epochs "
                        f"(best={best_metric:.4f})"
                    )
                    if patience_left == 0:
                        print(f"  [early stopping] Stopping at epoch {epoch+1}.")
                        stopped_early = True
                        break

        history["stopped_early"] = stopped_early  # type: ignore[assignment]
        return history

    def _rebuild_eval_loader(
        self,
        loader: DataLoader,
        rank: int,
        world_size: int,
    ) -> DataLoader:
        """Shard validation across DDP ranks so all GPUs stay busy."""
        if world_size <= 1:
            return loader
        from torch.utils.data import DataLoader as _DL
        from torch.utils.data.distributed import DistributedSampler as _DS

        sampler = _DS(loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        return _DL(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            collate_fn=loader.collate_fn,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=loader.num_workers > 0,
            prefetch_factor=2 if loader.num_workers > 0 else None,
        )

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        rank_me_max_samples: int = 2048,
        rank: int = 0,
        world_size: int = 1,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Returns
        -------
        (avg_loss, val_metrics_dict)

        When world_size > 1, each rank evaluates its val shard and scalars are
        all-reduced so metrics match a full pass (RankMe uses a small per-rank
        subsample gathered on rank 0).
        """
        self.eval()
        eval_loader = self._rebuild_eval_loader(loader, rank, world_size)
        total, n = 0.0, 0
        std_dev_sum   = 0.0
        emb_buffer: List[torch.Tensor] = []
        emb_collected = 0
        per_rank_cap = max(64, rank_me_max_samples // max(world_size, 1))
        prev_cov_sync = self._cov_sync_ddp
        self._cov_sync_ddp = False

        try:
            for batch in eval_loader:
                codes       = batch["codes"].to(device, non_blocking=True)
                attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
                values      = batch.get("values")
                z_scores    = batch.get("z_scores")
                delta_times = batch.get("delta_times")
                value_mask  = batch.get("value_mask")
                hours_since_first = batch.get("hours_since_first")
                if values      is not None: values      = values.to(device, non_blocking=True)
                if z_scores    is not None: z_scores    = z_scores.to(device, non_blocking=True)
                if delta_times is not None: delta_times = delta_times.to(device, non_blocking=True)
                if value_mask  is not None: value_mask  = value_mask.to(device, non_blocking=True)
                if hours_since_first is not None:
                    hours_since_first = hours_since_first.to(device, non_blocking=True)
                pre_mask = _pre_mask_dict_from_batch(batch)

                use_amp = self.config.use_amp and device.type == "cuda"
                amp_dtype = (
                    torch.bfloat16
                    if self.config.amp_dtype in ("bf16", "bfloat16")
                    else torch.float16
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    _, _, l_total = self.forward(
                        codes,
                        attn_mask,
                        values,
                        z_scores,
                        delta_times,
                        value_mask,
                        hours_since_first=hours_since_first,
                        pre_mask=pre_mask,
                    )
                total += l_total.item()
                n += 1

                std_dev_sum += self._batch_mon.get("std_dev_embeddings", 0.0)

                # Collect target embeddings for epoch-level RankMe
                if emb_collected < per_rank_cap:
                    tgt_embs = self._batch_mon.get("_tgt_embs_for_rank")
                    if tgt_embs is not None:
                        tgt_embs = tgt_embs.cpu()
                        need = per_rank_cap - emb_collected
                        take = min(need, tgt_embs.shape[0])
                        emb_buffer.append(tgt_embs[:take])
                        emb_collected += take
        finally:
            self._cov_sync_ddp = prev_cov_sync

        if world_size > 1:
            import torch.distributed as _dist

            stats = torch.tensor(
                [total, float(n), std_dev_sum], device=device, dtype=torch.float64
            )
            _dist.all_reduce(stats, op=_dist.ReduceOp.SUM)
            total = float(stats[0].item())
            n = int(stats[1].item())
            std_dev_sum = float(stats[2].item())

        avg_loss   = total       / max(n, 1)
        avg_std_dev = std_dev_sum / max(n, 1)

        rank_me_val = 0.0
        if world_size > 1:
            import torch.distributed as _dist

            d_model = self.encoder.config.d_model
            local_rows = (
                torch.cat(emb_buffer, dim=0).float()
                if emb_buffer
                else torch.zeros(0, d_model)
            )
            n_local = local_rows.shape[0]
            cap = per_rank_cap
            padded = torch.zeros(cap, d_model, device=device)
            if n_local > 0:
                padded[: min(n_local, cap)] = local_rows[: min(n_local, cap)].to(device)
            count_t = torch.tensor([n_local], device=device, dtype=torch.long)
            all_counts = [
                torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)
            ]
            _dist.all_gather(all_counts, count_t)
            gathered = [torch.zeros(cap, d_model, device=device) for _ in range(world_size)]
            _dist.all_gather(gathered, padded)
            if rank == 0:
                parts = []
                for r in range(world_size):
                    nr = int(all_counts[r].item())
                    if nr > 0:
                        parts.append(gathered[r][:nr].cpu())
                if parts:
                    z_all = torch.cat(parts, dim=0)
                    rank_me_val = _rank_me_from_rows(z_all[:rank_me_max_samples])
        elif emb_buffer:
            z_all = torch.cat(emb_buffer, dim=0).float()
            z_sub = z_all[:rank_me_max_samples]
            rank_me_val = _rank_me_from_rows(z_sub)

        val_metrics: Dict[str, float] = {
            "std_dev_embeddings": avg_std_dev,
            "rank_me":            rank_me_val,
        }
        return avg_loss, val_metrics
