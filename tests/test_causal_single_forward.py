"""
Branch B + causal_single forward: finite loss across sequence lengths and edge cases.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loss.covariance_reg import SIGRegLoss
from masking.span_masking import SpanMasker
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.predictor import HoursSinceFirstEmbedding, Predictor, TemporalSpanPrompt
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.trainer import JEPATrainer, TrainerConfig

VOCAB = 50
D = 64
MIN_TGT = 10


def _build_trainer(
    device: str = "cpu",
    future_time_decay_lambda: float = 0.002888,
    causal_single_predictor_attn: str = "bidirectional",
) -> JEPATrainer:
    embedding = EventEmbedding(
        EmbeddingConfig(
            embedding_type="learned",
            vocab_size=VOCAB,
            d_model=D,
            use_value=True,
            use_time=True,
        )
    )
    encoder = EHRTransformerEncoder(
        TransformerEncoderConfig(
            n_layers=2, d_model=D, n_heads=4, ffn_dim=128, dropout=0.0
        )
    )
    token_predictor = EHRTransformerEncoder(
        TransformerEncoderConfig(
            n_layers=2, d_model=D, n_heads=4, ffn_dim=128, dropout=0.0
        )
    )
    cfg = TrainerConfig(
        use_perceiver=False,
        min_span_for_perceiver=15,
        min_target_events=MIN_TGT,
        masking_strategy="causal_single",
        future_time_decay_lambda=future_time_decay_lambda,
        future_time_decay_weight_floor=0.05,
        causal_single_predictor_attn=causal_single_predictor_attn,
        lambda_cov=0.1,
        device=device,
        early_stopping_patience=0,
    )
    return JEPATrainer(
        embedding=embedding,
        encoder=encoder,
        prompt=TemporalSpanPrompt(D),
        time_embed=HoursSinceFirstEmbedding(D),
        predictor=Predictor(D, n_heads=4, n_layers=2, dropout=0.0),
        token_predictor=token_predictor,
        context_pooler=None,
        target_pooler=None,
        cov_loss=SIGRegLoss(num_slices=8),
        masker=SpanMasker(),
        config=cfg,
    ).to(device)


def _causal_pre_mask(
    batch_size: int,
    seq_len: int,
    cut: int,
    n_target: int,
    *,
    empty_target_row: int | None = None,
    pad_tail: int = 0,
) -> dict:
    pre = {
        "mask_context_indices": [],
        "mask_target_spans": [],
        "mask_span_times": [],
        "mask_target_delta_minutes": [],
    }
    for b in range(batch_size):
        if empty_target_row is not None and b == empty_target_row:
            pre["mask_context_indices"].append(list(range(0, 12)))
            pre["mask_target_spans"].append([[]])
            pre["mask_span_times"].append([(0.0, 0.0)])
            pre["mask_target_delta_minutes"].append([[]])
            continue
        c = cut + (b % 5)
        ctx = list(range(0, c + 1))
        tgt = list(range(c + 1, min(c + 1 + n_target, seq_len - pad_tail)))
        if len(tgt) < MIN_TGT:
            tgt = list(range(c + 1, c + 1 + MIN_TGT))
            tgt = [min(p, seq_len - pad_tail - 1) for p in tgt]
        delta_min = [float((p - c) * 60.0) for p in tgt]
        pre["mask_context_indices"].append(ctx)
        pre["mask_target_spans"].append([tgt])
        pre["mask_span_times"].append([(2.0, 3.0)])
        pre["mask_target_delta_minutes"].append([delta_min])
    return pre


@pytest.mark.parametrize("seq_len", [32, 64, 128, 512, 1024])
def test_causal_single_finite_across_seq_lengths(seq_len: int):
    trainer = _build_trainer()
    B = 8
    codes = torch.randint(0, VOCAB, (B, seq_len))
    attn = torch.ones(B, seq_len, dtype=torch.long)
    if seq_len >= 64:
        attn[:, -8:] = 0
    pre = _causal_pre_mask(B, seq_len, cut=20, n_target=24)
    values = torch.rand(B, seq_len)
    z_scores = torch.randn(B, seq_len).clamp(-5, 5)
    delta_times = torch.rand(B, seq_len).clamp(0, 5)
    value_mask = torch.ones(B, seq_len, dtype=torch.long)
    l_pred, l_cov, l_total = trainer(
        codes, attn, values, z_scores, delta_times, value_mask, pre_mask=pre
    )
    assert torch.isfinite(l_pred), f"l_pred nan at L={seq_len}"
    assert torch.isfinite(l_cov), f"l_cov nan at L={seq_len}"
    assert torch.isfinite(l_total)
    assert l_total.requires_grad


def test_causal_single_short_padded_sequence():
    """Shorter than max_seq_len with right-pad (like collator)."""
    trainer = _build_trainer()
    L = 40
    B = 4
    codes = torch.randint(0, VOCAB, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    attn[:, 25:] = 0
    pre = _causal_pre_mask(B, L, cut=10, n_target=15, pad_tail=15)
    l_pred, l_cov, l_total = trainer(
        codes,
        attn,
        torch.rand(B, L),
        torch.randn(B, L).clamp(-3, 3),
        torch.rand(B, L),
        torch.ones(B, L, dtype=torch.long),
        pre_mask=pre,
    )
    assert torch.isfinite(l_total)


def test_causal_single_all_invalid_targets_returns_zero_loss():
    trainer = _build_trainer()
    L = 64
    B = 3
    codes = torch.randint(0, VOCAB, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    pre = {
        "mask_context_indices": [list(range(20)), list(range(15)), list(range(18))],
        "mask_target_spans": [[list(range(21, 25))], [[]], [[]]],
        "mask_span_times": [[(1.0, 1.0)], [(0.0, 0.0)], [(0.0, 0.0)]],
        "mask_target_delta_minutes": [[[60.0, 120.0]], [[]], [[]]],
    }
    l_pred, l_cov, l_total = trainer(codes, attn, pre_mask=pre)
    assert l_pred.item() == 0.0
    assert l_total.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_causal_single_cuda_large_batch():
    """Production-like: cuda, wide model, L=1024, B=16."""
    device = "cuda"
    D_prod = 128
    embedding = EventEmbedding(
        EmbeddingConfig(
            embedding_type="learned",
            vocab_size=VOCAB,
            d_model=D_prod,
            use_value=True,
            use_time=True,
        )
    )
    enc_cfg = TransformerEncoderConfig(
        n_layers=2, d_model=D_prod, n_heads=8, ffn_dim=512, dropout=0.0
    )
    trainer = JEPATrainer(
        embedding=embedding,
        encoder=EHRTransformerEncoder(enc_cfg),
        prompt=TemporalSpanPrompt(D_prod),
        time_embed=HoursSinceFirstEmbedding(D_prod),
        predictor=Predictor(D_prod, n_heads=8, n_layers=2, dropout=0.0),
        token_predictor=EHRTransformerEncoder(enc_cfg),
        context_pooler=None,
        target_pooler=None,
        cov_loss=SIGRegLoss(num_slices=8),
        masker=SpanMasker(),
        config=TrainerConfig(
            use_perceiver=False,
            min_target_events=10,
            masking_strategy="causal_single",
            future_time_decay_lambda=0.002888,
            device=device,
            early_stopping_patience=0,
        ),
    ).to(device)
    L = 1024
    B = 16
    codes = torch.randint(0, VOCAB, (B, L), device=device)
    attn = torch.ones(B, L, dtype=torch.long, device=device)
    pre = _causal_pre_mask(B, L, cut=200, n_target=40)
    values = torch.rand(B, L, device=device)
    z_scores = torch.randn(B, L, device=device).clamp(-5, 5)
    delta_times = torch.rand(B, L, device=device).clamp(0, 5)
    vm = torch.ones(B, L, dtype=torch.long, device=device)
    l_pred, l_cov, l_total = trainer(
        codes, attn, values, z_scores, delta_times, vm, pre_mask=pre
    )
    assert torch.isfinite(l_pred)
    assert torch.isfinite(l_total)
