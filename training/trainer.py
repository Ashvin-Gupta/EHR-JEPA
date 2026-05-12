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
    Learnable MASK tokens + TemporalSpanPrompt → [B, N_span, d]
    Concat with context tokens (original pos IDs) → Token Predictor
    Slice mask-token outputs → Y_hat [B, N_span, d]
    Loss: MSE(Y_hat, Y_tgt.detach()) + λ·CovReg(Y_tgt)

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
from loss.jepa_loss import jepa_prediction_loss
from masking.span_masking import SpanMasker
from models.event_embedding import EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.predictor import Predictor, TemporalSpanPrompt
from models.projection_head import ProjectionHead
from models.transformer_encoder import EHRTransformerEncoder


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

    # Checkpointing — set to "" to disable
    # Best model (by val_loss, or train_loss if no val set) is saved to
    # {checkpoint_dir}/best.pt.  End-of-training model saved to
    # {checkpoint_dir}/last.pt.
    checkpoint_dir: str = ""

    # General
    n_epochs: int = 10
    device: str = "cpu"


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
        TemporalSpanPrompt (used by both branches).
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
        SpanMasker.
    config:
        TrainerConfig.
    """

    def __init__(
        self,
        embedding: EventEmbedding,
        encoder: EHRTransformerEncoder,
        prompt: TemporalSpanPrompt,
        predictor: Predictor,
        token_predictor: EHRTransformerEncoder,
        context_pooler: Optional[LatentCrossAttentionPool],
        target_pooler: Optional[LatentCrossAttentionPool],
        cov_loss: SIGRegLoss,
        masker: SpanMasker,
        config: TrainerConfig,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.prompt = prompt
        self.predictor = predictor
        self.token_predictor = token_predictor
        self.context_pooler = context_pooler
        self.target_pooler = target_pooler
        self.cov_loss = cov_loss
        self.masker = masker
        self.config = config

        # Training step for SIGReg RNG sync across DDP ranks (set each batch).
        self._cov_global_step: int = 0

        # Learnable mask token for Branch B
        d_model = encoder.config.d_model
        self.mask_token = nn.Parameter(torch.zeros(d_model))

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
        pre_mask:        dict with keys 'mask_context_indices', 'mask_target_spans',
                         'mask_span_times' (pre-computed by MEDSCollator in worker
                         process).  When None the masker is run here on the main thread.

        Returns
        -------
        (L_pred, L_cov, L_total) — scalar tensors.
        Also populates self._batch_mon with monitoring scalars (no extra GPU→CPU
        transfers — values are already detached floats).
        """
        B, L = codes.shape
        device = codes.device

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

        # 2. Span masking — use pre-computed results from the DataLoader worker
        #    when available; fall back to on-the-fly masking otherwise.
        if pre_mask is not None:
            all_context_indices: List[List[int]] = pre_mask["mask_context_indices"]
            all_target_spans: List[List[List[int]]] = pre_mask["mask_target_spans"]
            all_span_times: List[List[Tuple[float, float]]] = pre_mask["mask_span_times"]
        else:
            all_context_indices = []
            all_target_spans = []
            all_span_times = []
            for b in range(B):
                result = self.masker(
                    seq_len=L,
                    attention_mask=attention_mask[b],
                    times=None,
                )
                all_context_indices.append(result.context_indices)
                all_target_spans.append(result.target_spans)
                all_span_times.append(result.span_times)

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

        # Use minimum num_spans across batch for uniform tensors
        num_spans_per_sample = [len(spans) for spans in all_target_spans]
        num_spans = min(num_spans_per_sample) if num_spans_per_sample else 0
        if num_spans == 0:
            zero = x.new_zeros(())
            self._batch_mon = {
                "std_dev_embeddings": 0.0, "rank_me": 0.0,
                "std_dev_values": _std_values, "std_dev_times": _std_times,
                "_N_input": 0, "_N_target": 0, "_N_context": 0,
                "avg_context_length": 0.0, "avg_target_span_length": 0.0,
            }
            return zero, zero, zero

        all_target_spans = [spans[:num_spans] for spans in all_target_spans]
        all_span_times   = [st[:num_spans]    for st    in all_span_times]

        # 3. Target pathway — full sequence, always with grad
        target_enc_out = self.encoder(x, attention_mask=attention_mask)  # (B, L, d)
        target_spans_list = self._extract_target_spans(
            target_enc_out, all_target_spans
        )  # list of num_spans tensors, each (B, N_span_s, d)

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
                context_enc_out, ctx_mask, target_spans_list, all_span_times
            )
        else:
            l_pred, l_cov = self._forward_token(
                context_enc_out, ctx_pos_ids, ctx_mask,
                target_spans_list, all_target_spans, all_span_times
            )

        l_total = l_pred + self.config.lambda_cov * l_cov

        # --- Monitoring metrics (detached, no overhead) ---
        # Average std-dev across target-span embedding dimensions
        # (collapse indicator: should stay well above 0)
        with torch.no_grad():
            if target_spans_list:
                all_tgt = torch.cat(
                    [t.reshape(-1, t.shape[-1]) for t in target_spans_list], dim=0
                )  # (N, d)
                std_dev = all_tgt.std(dim=0).mean().item()
                # RankMe per-batch: subsample ≤512 rows so SVD stays fast
                z_sample = all_tgt[:512].float()
                try:
                    _, s, _ = torch.linalg.svd(z_sample, full_matrices=False)
                    p = s / (s.sum() + 1e-8)
                    rank_me = float(torch.exp(-(p * torch.log(p + 1e-8)).sum()).item())
                except Exception:
                    rank_me = 0.0
            else:
                all_tgt = x.new_zeros((1, x.shape[-1]))
                std_dev = 0.0
                rank_me = 0.0

        # Raw token counts — use full-mask totals (see N_target_full above).
        # Ratios in train loop: target_ratio = N_target_full / N_input,
        # context_ratio = N_context_full / sum(N_sequence) (per-batch).
        avg_ctx_len = sum(len(ctx) for ctx in all_context_indices) / max(B_size_pre, 1)
        avg_tgt_area = sum(per_sample_target_area_full) / max(B_size_pre, 1)

        self._batch_mon = {
            "std_dev_embeddings":     std_dev,
            "rank_me":                rank_me,
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
            "_tgt_embs_for_rank": all_tgt.detach() if not self.training else None,
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
        B, _, d = x.shape
        max_ctx_len = max((len(ci) for ci in all_context_indices), default=1)

        x_ctx    = x.new_zeros(B, max_ctx_len, d)
        pos_ids  = torch.zeros(B, max_ctx_len, dtype=torch.long, device=device)
        ctx_mask = torch.zeros(B, max_ctx_len, dtype=torch.long, device=device)

        for b, ctx_idx in enumerate(all_context_indices):
            n = len(ctx_idx)
            if n == 0:
                continue
            idx_t = torch.tensor(ctx_idx, dtype=torch.long, device=device)
            x_ctx[b, :n]   = x[b][idx_t]
            pos_ids[b, :n]  = idx_t
            ctx_mask[b, :n] = 1

        return x_ctx, pos_ids, ctx_mask

    def _extract_target_spans(
        self,
        encoder_out: torch.Tensor,
        all_target_spans: List[List[List[int]]],
    ) -> List[torch.Tensor]:
        """
        For each span index s, build a padded tensor [B, max_span_len_s, d]
        containing the target encoder outputs at the span positions.

        Returns a list of num_spans tensors.
        """
        B, L, d = encoder_out.shape
        device = encoder_out.device
        num_spans = len(all_target_spans[0])
        result = []

        for s in range(num_spans):
            max_span_len = max(len(all_target_spans[b][s]) for b in range(B))
            if max_span_len == 0:
                result.append(encoder_out.new_zeros(B, 1, d))
                continue

            span_t = encoder_out.new_zeros(B, max_span_len, d)
            for b in range(B):
                span_idx = all_target_spans[b][s]
                n = len(span_idx)
                if n == 0:
                    continue
                idx_t = torch.tensor(span_idx, dtype=torch.long, device=device)
                span_t[b, :n] = encoder_out[b][idx_t]
            result.append(span_t)

        return result

    # ------------------------------------------------------------------
    # Branch A: Perceiver-JEPA
    # ------------------------------------------------------------------

    def _forward_perceiver(
        self,
        context_enc_out: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_spans_list: List[torch.Tensor],
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

            # Target perceiver → (B, n_latents, d)
            Z_tgt = self.target_pooler(span_tokens)

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
                self.cov_loss(Z_tgt, global_step=self._cov_global_step)
            )

        if not pred_losses:
            zero = context_enc_out.new_zeros(())
            return zero, zero

        l_pred = torch.stack(pred_losses).mean()
        l_cov  = torch.stack(cov_losses).mean()
        return l_pred, l_cov

    # ------------------------------------------------------------------
    # Branch B: Token I-JEPA
    # ------------------------------------------------------------------

    def _forward_token(
        self,
        context_enc_out: torch.Tensor,
        ctx_pos_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_spans_list: List[torch.Tensor],
        all_target_spans: List[List[List[int]]],
        all_span_times: List[List[Tuple[float, float]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each span:
          - Build MASK tokens (learnable token + temporal prompt): [B, N_span, d]
          - Concatenate with context: [B, N_ctx + N_span, d]
          - Pass through token predictor with original position IDs
          - Slice last N_span outputs → Y_hat [B, N_span, d]
          - L_pred += MSE(Y_hat, Y_tgt.detach())
          - L_cov  += CovReg(Y_tgt)
        Returns averaged (l_pred, l_cov).
        """
        B, N_ctx, d = context_enc_out.shape
        device = context_enc_out.device
        num_spans = len(target_spans_list)

        pred_losses: List[torch.Tensor] = []
        cov_losses:  List[torch.Tensor] = []

        for s, y_tgt in enumerate(target_spans_list):
            N_span = y_tgt.shape[1]

            # Temporal prompt for this span: (B, d)
            coords = self._span_coords_for_span(all_span_times, s, device)  # (B, 2)
            span_prompt = self.prompt(coords.unsqueeze(1))  # (B, 1, d)

            # MASK tokens: broadcast mask_token + span prompt
            mask_tokens = (
                self.mask_token.view(1, 1, d).expand(B, N_span, d) + span_prompt
            )  # (B, N_span, d)

            # Concatenate context + mask tokens
            x_in = torch.cat([context_enc_out, mask_tokens], dim=1)
            # (B, N_ctx + N_span, d)

            # Build position IDs: context original positions + span positions
            span_pos_ids = self._span_pos_ids(all_target_spans, s, N_span, device)
            # (B, N_span)
            pos_ids = torch.cat([ctx_pos_ids, span_pos_ids], dim=1)
            # (B, N_ctx + N_span)

            # Attention mask: context mask + 1s for mask tokens
            span_attn = torch.ones(B, N_span, dtype=torch.long, device=device)
            attn_in = torch.cat([ctx_mask, span_attn], dim=1)
            # (B, N_ctx + N_span)

            # Token predictor
            out = self.token_predictor(x_in, attention_mask=attn_in, position_ids=pos_ids)
            # (B, N_ctx + N_span, d)

            # Slice the mask-token outputs
            y_hat = out[:, N_ctx:, :]  # (B, N_span, d)

            pred_losses.append(jepa_prediction_loss(y_hat, y_tgt))
            cov_losses.append(
                self.cov_loss(y_tgt, global_step=self._cov_global_step)
            )

        if not pred_losses:
            zero = context_enc_out.new_zeros(())
            return zero, zero

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
        from evaluation.linear_probe import FrozenEHREncoder, LinearProbe

        is_dist = (world_size > 1)
        pooler  = self.context_pooler

        encoder = FrozenEHREncoder(
            embedding=self.embedding,
            encoder=self.encoder,
            pooler=pooler,
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

        best_metric        = float("inf")
        best_ckpt_metric   = float("inf")   # tracks best val/train loss for checkpointing
        best_probe_auroc   = -float("inf")  # tracks best probe val_auroc for probe_best.pt
        patience_left      = cfg.early_stopping_patience
        stopped_early      = False

        # Set up checkpoint directory once so the path is always available
        ckpt_dir = cfg.checkpoint_dir.strip() if cfg.checkpoint_dir else ""
        if ckpt_dir and is_main_process:
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"[train] Checkpoints:     {ckpt_dir}")

        if is_main_process:
            branch = "Perceiver (A)" if cfg.use_perceiver else "Token I-JEPA (B)"
            print(f"[train] Branch:          {branch}")
            print(f"[train] Scheduler:       {cfg.scheduler}")
            print(f"[train] Grad clip:       {cfg.grad_clip if cfg.grad_clip > 0 else 'disabled'}")
            print(f"[train] Weight decay:    {cfg.weight_decay}")
            print(f"[train] lambda_cov:      {cfg.lambda_cov}")
            print(f"[train] Early stopping:  "
                  f"{'disabled' if cfg.early_stopping_patience == 0 else f'patience={cfg.early_stopping_patience}, metric={cfg.early_stopping_metric}'}")
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

                if values      is not None: values      = values.to(device, non_blocking=True)
                if z_scores    is not None: z_scores    = z_scores.to(device, non_blocking=True)
                if delta_times is not None: delta_times = delta_times.to(device, non_blocking=True)
                if value_mask  is not None: value_mask  = value_mask.to(device, non_blocking=True)

                # Pre-computed masking from the DataLoader worker (plain Python
                # lists — they stay on CPU, never move to device).
                pre_mask = (
                    {
                        "mask_context_indices": batch["mask_context_indices"],
                        "mask_target_spans":    batch["mask_target_spans"],
                        "mask_span_times":      batch["mask_span_times"],
                    }
                    if "mask_context_indices" in batch else None
                )

                orig_seq_lengths = batch.get("orig_seq_lengths")

                # SIGReg: same projection directions on every GPU (seed = step index).
                self._cov_global_step = global_step

                optimizer.zero_grad()
                l_pred, l_cov, l_total = _forward(
                    codes, attn_mask, values, z_scores, delta_times, value_mask,
                    pre_mask=pre_mask,
                )
                l_total.backward()

                grad_norm = 0.0
                if cfg.grad_clip > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.parameters(), cfg.grad_clip
                    ).item()

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

                if is_main_process and on_batch_end is not None:
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

            # ---- Barrier 1: after training, before rank-0 validation -----------
            # Ranks 1-3 must not start the next training epoch while rank 0 runs
            # validation.  They wait here until rank 0 is done.
            if ddp_module is not None:
                import torch.distributed as _dist
                _dist.barrier()

            val_line = ""
            if is_main_process:
                if val_loader is not None:
                    print(f"  [val] Running validation … ", end="", flush=True)
                    avg_val, val_metrics = self._eval_epoch(val_loader, device)
                    print(
                        f"val_loss={avg_val:.4f}"
                        f"  std_dev={val_metrics.get('std_dev_embeddings', 0.0):.4f}"
                        f"  rank_me={val_metrics.get('rank_me', 0.0):.1f}"
                    )
                    history["val_loss"].append(avg_val)
                    epoch_metrics["val_loss"] = avg_val
                    epoch_metrics.update(val_metrics)
                    val_line = f"  val={avg_val:.4f}"

                print(
                    f"Epoch {epoch+1}/{cfg.n_epochs}  "
                    f"train={avg_train:.4f}  (pred={avg_l_pred:.4f}, cov={avg_l_cov:.4f})"
                    f"{val_line}  lr={current_lr:.2e}"
                )

                if on_epoch_end is not None:
                    on_epoch_end(epoch, epoch_metrics)

            # ---- Barrier 2: before probe — all ranks participate ---------------
            # Probe training is distributed: each rank processes its data shard
            # and DDP all-reduces gradients on the linear layer.  All ranks must
            # enter together.
            if ddp_module is not None:
                _dist.barrier()

            # ---- Inline probe evaluation (all ranks when DDP is active) --------
            # Run on epoch 1 unconditionally, then every probe_interval epochs.
            _run_probe_this_epoch = (
                probe_train_loader is not None
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
                elif monitor < best_metric:
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

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        rank_me_max_samples: int = 2048,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Returns
        -------
        (avg_loss, val_metrics_dict)

        val_metrics_dict keys:
          unique_codes_seen : number of distinct vocabulary IDs in this val epoch
          rank_me           : effective rank estimate of target embeddings
                              (exp of entropy over singular values; higher = richer)
        """
        self.eval()
        total, n = 0.0, 0
        std_dev_sum   = 0.0
        emb_buffer: List[torch.Tensor] = []
        emb_collected = 0

        for batch in loader:
            codes       = batch["codes"].to(device, non_blocking=True)
            attn_mask   = batch["attention_mask"].to(device, non_blocking=True)
            values      = batch.get("values")
            z_scores    = batch.get("z_scores")
            delta_times = batch.get("delta_times")
            value_mask  = batch.get("value_mask")
            if values      is not None: values      = values.to(device, non_blocking=True)
            if z_scores    is not None: z_scores    = z_scores.to(device, non_blocking=True)
            if delta_times is not None: delta_times = delta_times.to(device, non_blocking=True)
            if value_mask  is not None: value_mask  = value_mask.to(device, non_blocking=True)
            pre_mask = (
                {
                    "mask_context_indices": batch["mask_context_indices"],
                    "mask_target_spans":    batch["mask_target_spans"],
                    "mask_span_times":      batch["mask_span_times"],
                }
                if "mask_context_indices" in batch else None
            )

            _, _, l_total = self.forward(
                codes, attn_mask, values, z_scores, delta_times, value_mask,
                pre_mask=pre_mask,
            )
            total += l_total.item()
            n += 1

            std_dev_sum += self._batch_mon.get("std_dev_embeddings", 0.0)

            # Collect target embeddings for epoch-level RankMe
            if emb_collected < rank_me_max_samples:
                tgt_embs = self._batch_mon.get("_tgt_embs_for_rank")
                if tgt_embs is not None:
                    tgt_embs = tgt_embs.cpu()
                    need = rank_me_max_samples - emb_collected
                    take = min(need, tgt_embs.shape[0])
                    emb_buffer.append(tgt_embs[:take])
                    emb_collected += take

        avg_loss   = total       / max(n, 1)
        avg_std_dev = std_dev_sum / max(n, 1)

        # RankMe over accumulated val embeddings (more stable than per-batch)
        rank_me_val = 0.0
        if emb_buffer:
            z_all = torch.cat(emb_buffer, dim=0).float()
            try:
                _, s, _ = torch.linalg.svd(z_all, full_matrices=False)
                p = s / (s.sum() + 1e-8)
                rank_me_val = float(torch.exp(-(p * torch.log(p + 1e-8)).sum()).item())
            except Exception:
                rank_me_val = 0.0

        val_metrics: Dict[str, float] = {
            "std_dev_embeddings": avg_std_dev,
            "rank_me":            rank_me_val,
        }
        return avg_loss, val_metrics
