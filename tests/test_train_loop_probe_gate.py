"""Train-loop should skip inline probe when inline_probe_during_pretrain=False."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from loss.covariance_reg import SIGRegLoss
from masking.span_masking import SpanMasker
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.predictor import Predictor, TemporalSpanPrompt
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.trainer import JEPATrainer, TrainerConfig


def _tiny_trainer() -> JEPATrainer:
    D, V, Ln = 16, 40, 2
    emb = EventEmbedding(
        EmbeddingConfig(
            embedding_type="learned",
            vocab_size=V,
            d_model=D,
            unk_idx=V - 1,
            use_value=False,
            use_time=False,
        )
    )
    enc = EHRTransformerEncoder(
        TransformerEncoderConfig(n_layers=1, d_model=D, n_heads=2, ffn_dim=32, dropout=0.0)
    )
    prompt = TemporalSpanPrompt(D)
    pred = Predictor(D, n_heads=2, n_layers=1, dropout=0.0)
    tok = EHRTransformerEncoder(
        TransformerEncoderConfig(n_layers=1, d_model=D, n_heads=2, ffn_dim=32, dropout=0.0)
    )
    pool = LatentCrossAttentionPool(D, n_latents=Ln, n_heads=2)
    cov = SIGRegLoss(num_slices=4)
    masker = SpanMasker(0.2, 1, 3, seed=0)
    cfg = TrainerConfig(
        use_perceiver=True,
        min_span_for_perceiver=1,
        use_proj_head=False,
        device="cpu",
        n_epochs=1,
        early_stopping_patience=0,
        checkpoint_dir="",
    )
    return JEPATrainer(
        embedding=emb,
        encoder=enc,
        prompt=prompt,
        predictor=pred,
        token_predictor=tok,
        context_pooler=pool,
        target_pooler=pool,
        cov_loss=cov,
        masker=masker,
        config=cfg,
    )


def test_inline_probe_skipped_when_disabled():
    trainer = _tiny_trainer()
    trainer._run_inline_probe = MagicMock(side_effect=AssertionError("probe should not run"))

    B, L = 2, 32
    codes = torch.randint(0, 39, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)

    class _Loader:
        def __init__(self) -> None:
            self._batch = {
                "codes": codes,
                "attention_mask": attn,
                "mask_context_indices": [[0, 1]] * B,
                "mask_target_spans": [[[2, 3]]] * B,
                "mask_span_times": [[(0.0, 1.0)]] * B,
            }

        def __iter__(self):
            yield self._batch

        def __len__(self) -> int:
            return 1

    loader = _Loader()
    opt = torch.optim.AdamW(trainer.parameters(), lr=1e-3)
    trainer.train_loop(
        train_loader=loader,
        val_loader=None,
        optimizer=opt,
        probe_train_loader=loader,
        probe_val_loader=None,
        probe_n_epochs=1,
        probe_interval=1,
        inline_probe_during_pretrain=False,
    )
    trainer._run_inline_probe.assert_not_called()
