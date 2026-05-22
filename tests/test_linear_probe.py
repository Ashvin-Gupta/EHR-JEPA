"""
Smoke tests for evaluation/linear_probe.py.

Covers:
  - FrozenEHREncoder.output_dim (the bug that caused AttributeError)
  - FrozenEHREncoder forward pass shape
  - FrozenEHREncoder with pooler=None (mean-pool fallback)
  - LinearProbe forward pass shape
  - _compute_all_metrics on known inputs
  - _roc_auc / _au_pr on edge cases
  - train_linear_probe end-to-end (tiny synthetic dataset)
  - _run_inline_probe via JEPATrainer (the full inline path)
"""

import os
import sys
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.linear_probe import (
    FrozenEHREncoder,
    LinearProbe,
    _au_pr,
    _compute_all_metrics,
    _roc_auc,
    train_linear_probe,
)
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig

# -----------------------------------------------------------------------
# Shared small-scale constants
# -----------------------------------------------------------------------
D         = 32
N_LATENTS = 4
VOCAB     = 50
B         = 4
L         = 20


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_encoder() -> EHRTransformerEncoder:
    return EHRTransformerEncoder(
        TransformerEncoderConfig(n_layers=1, d_model=D, n_heads=4, ffn_dim=64, dropout=0.0)
    )


def _make_embedding() -> EventEmbedding:
    return EventEmbedding(EmbeddingConfig(
        embedding_type="learned", vocab_size=VOCAB, d_model=D,
        use_value=False, use_time=False,
    ))


def _make_pooler() -> LatentCrossAttentionPool:
    return LatentCrossAttentionPool(d_model=D, n_latents=N_LATENTS, n_heads=4)


def _make_frozen_encoder(
    with_pooler: bool = True,
    pooling_mode: str = "mean_pool",
) -> FrozenEHREncoder:
    cls_token = torch.randn(D) * 0.02 if pooling_mode == "cls" else None
    return FrozenEHREncoder(
        embedding=_make_embedding(),
        encoder=_make_encoder(),
        pooler=_make_pooler() if with_pooler else None,
        cls_token=cls_token,
        pooling_mode=pooling_mode,  # type: ignore[arg-type]
    )


def _make_probe_dataloader(n: int = 16) -> DataLoader:
    """Synthetic DataLoader yielding dicts with codes/attention_mask/label."""
    codes = torch.randint(0, VOCAB, (n, L))
    attn  = torch.ones(n, L, dtype=torch.long)
    labels = torch.randint(0, 2, (n,)).float()

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i):
            return {"codes": codes[i], "attention_mask": attn[i], "labels": labels[i]}

    def _collate(batch):
        return {
            "codes":          torch.stack([b["codes"]          for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels":         torch.stack([b["labels"]         for b in batch]),
        }

    return DataLoader(_DS(), batch_size=4, collate_fn=_collate)


# -----------------------------------------------------------------------
# output_dim — the specific bug that was filed
# -----------------------------------------------------------------------

def test_output_dim_with_pooler():
    enc = _make_frozen_encoder(with_pooler=True)
    expected = N_LATENTS * D
    assert enc.output_dim == expected, (
        f"output_dim={enc.output_dim}, expected {expected}"
    )
    print(f"\n[test_output_dim_with_pooler]  output_dim={enc.output_dim} ✓")


def test_output_dim_without_pooler():
    enc = _make_frozen_encoder(with_pooler=False)
    assert enc.output_dim == D, (
        f"output_dim={enc.output_dim}, expected {D} (mean-pool fallback)"
    )
    print(f"\n[test_output_dim_without_pooler]  output_dim={enc.output_dim} ✓")


# -----------------------------------------------------------------------
# FrozenEHREncoder forward
# -----------------------------------------------------------------------

def test_frozen_encoder_forward_with_pooler():
    enc   = _make_frozen_encoder(with_pooler=True)
    codes = torch.randint(0, VOCAB, (B, L))
    attn  = torch.ones(B, L, dtype=torch.long)
    out   = enc(codes, attn)
    assert out.shape == (B, N_LATENTS * D), f"Got {out.shape}"
    # Frozen — no grad should flow
    assert not out.requires_grad
    print(f"\n[test_frozen_encoder_forward_with_pooler]  shape={out.shape} ✓")


def test_frozen_encoder_forward_without_pooler():
    enc   = _make_frozen_encoder(with_pooler=False)
    codes = torch.randint(0, VOCAB, (B, L))
    attn  = torch.ones(B, L, dtype=torch.long)
    out   = enc(codes, attn)
    assert out.shape == (B, D), f"Got {out.shape}"
    print(f"\n[test_frozen_encoder_forward_without_pooler]  shape={out.shape} ✓")


def test_frozen_encoder_padding_ignored():
    """Mean-pool fallback: padding tokens should not contribute to the output."""
    enc   = _make_frozen_encoder(with_pooler=False)
    codes = torch.randint(0, VOCAB, (2, L))
    attn  = torch.ones(2, L, dtype=torch.long)
    attn[1, L // 2:] = 0   # second sequence has padding in second half

    out = enc(codes, attn)
    assert out.shape == (2, D)
    print(f"\n[test_frozen_encoder_padding_ignored]  shape={out.shape} ✓")


# -----------------------------------------------------------------------
# LinearProbe
# -----------------------------------------------------------------------

def test_linear_probe_shape():
    probe  = LinearProbe(input_dim=N_LATENTS * D, dropout=0.0)
    z      = torch.randn(B, N_LATENTS * D)
    logits = probe(z)
    assert logits.shape == (B,), f"Got {logits.shape}"
    print(f"\n[test_linear_probe_shape]  logits shape={logits.shape} ✓")


def test_linear_probe_dropout():
    """Probe with dropout should still produce correct shape."""
    probe  = LinearProbe(input_dim=16, dropout=0.5)
    probe.train()
    z      = torch.randn(B, 16)
    logits = probe(z)
    assert logits.shape == (B,)
    print(f"\n[test_linear_probe_dropout]  shape={logits.shape} ✓")


# -----------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------

def test_roc_auc_perfect():
    labels = torch.tensor([1., 1., 0., 0.])
    probs  = torch.tensor([0.9, 0.8, 0.2, 0.1])
    auc = _roc_auc(labels, probs)
    assert abs(auc - 1.0) < 1e-6, f"Expected 1.0, got {auc}"
    print(f"\n[test_roc_auc_perfect]  AUROC={auc:.4f} ✓")


def test_roc_auc_degenerate():
    """Single class present → returns 0.5."""
    labels = torch.zeros(4)
    probs  = torch.rand(4)
    auc = _roc_auc(labels, probs)
    assert auc == 0.5, f"Expected 0.5, got {auc}"
    print(f"\n[test_roc_auc_degenerate]  AUROC={auc} ✓")


def test_au_pr_perfect():
    labels = torch.tensor([1., 1., 0., 0.])
    probs  = torch.tensor([0.9, 0.8, 0.2, 0.1])
    aupr = _au_pr(labels, probs)
    assert abs(aupr - 1.0) < 1e-6, f"Expected 1.0, got {aupr}"
    print(f"\n[test_au_pr_perfect]  AUPR={aupr:.4f} ✓")


def test_compute_all_metrics_perfect():
    labels = torch.tensor([1., 1., 1., 0., 0., 0.])
    logits = torch.tensor([2., 1.5, 1., -1., -1.5, -2.])
    m = _compute_all_metrics(labels, logits)
    for key in ("auroc", "aupr", "recall", "precision", "accuracy"):
        assert abs(m[key] - 1.0) < 1e-6, f"{key}={m[key]:.4f}, expected 1.0"
    print(f"\n[test_compute_all_metrics_perfect]  {m} ✓")


def test_compute_all_metrics_keys():
    labels = torch.tensor([1., 0., 1., 0.])
    logits = torch.zeros(4)
    m = _compute_all_metrics(labels, logits)
    assert set(m.keys()) == {"auroc", "aupr", "recall", "precision", "accuracy"}
    print(f"\n[test_compute_all_metrics_keys]  keys={set(m.keys())} ✓")


# -----------------------------------------------------------------------
# train_linear_probe end-to-end
# -----------------------------------------------------------------------

def test_train_linear_probe_runs():
    enc    = _make_frozen_encoder(with_pooler=True)
    probe  = LinearProbe(enc.output_dim, dropout=0.0)
    loader = _make_probe_dataloader(n=16)

    history, final_val = train_linear_probe(
        encoder=enc,
        probe=probe,
        train_loader=loader,
        val_loader=loader,   # reuse train as val for simplicity
        n_epochs=2,
        lr=1e-3,
        device="cpu",
        verbose=False,
    )

    # History should have one value per epoch
    assert len(history["train_loss"]) == 2
    assert len(history["val_auroc"]) == 2
    assert len(history["val_aupr"])  == 2

    # final_val should have all metric keys
    assert "auroc" in final_val
    assert "aupr"  in final_val
    assert "loss"  in final_val

    print(f"\n[test_train_linear_probe_runs]")
    print(f"  train_loss: {history['train_loss']}")
    print(f"  val_auroc:  {history['val_auroc']}")
    print(f"  final_val:  {final_val}")


def test_train_linear_probe_no_val():
    """train_linear_probe with no validation loader should still complete."""
    enc    = _make_frozen_encoder(with_pooler=True)
    probe  = LinearProbe(enc.output_dim)
    loader = _make_probe_dataloader(n=8)

    history, final_val = train_linear_probe(
        encoder=enc, probe=probe,
        train_loader=loader, val_loader=None,
        n_epochs=1, lr=1e-3, device="cpu", verbose=False,
    )
    assert len(history["train_loss"]) == 1
    assert final_val == {}
    print(f"\n[test_train_linear_probe_no_val]  train_loss={history['train_loss']} ✓")


# -----------------------------------------------------------------------
# _run_inline_probe via JEPATrainer
# -----------------------------------------------------------------------

def test_run_inline_probe():
    """Full inline probe path: JEPATrainer._run_inline_probe."""
    from training.trainer import JEPATrainer, TrainerConfig
    from loss.covariance_reg import SIGRegLoss
    from masking.span_masking import SpanMasker
    from models.predictor import Predictor, TemporalSpanPrompt

    d = D
    encoder  = _make_encoder()
    token_predictor = _make_encoder()
    pooler   = _make_pooler()
    prompt   = TemporalSpanPrompt(d)
    pred     = Predictor(d, n_heads=4, n_layers=1)
    cov_loss = SIGRegLoss(num_slices=8)
    masker   = SpanMasker(mask_ratio=0.3, default_num_spans=2, min_span_length=3)

    trainer = JEPATrainer(
        embedding=_make_embedding(),
        encoder=encoder,
        prompt=prompt,
        predictor=pred,
        token_predictor=token_predictor,
        context_pooler=pooler,
        target_pooler=pooler,
        cov_loss=cov_loss,
        masker=masker,
        config=TrainerConfig(use_perceiver=True),
    )

    loader = _make_probe_dataloader(n=16)
    device = torch.device("cpu")

    metrics = trainer._run_inline_probe(
        probe_train_loader=loader,
        probe_val_loader=loader,
        n_epochs=2,
        lr=1e-3,
        dropout=0.0,
        device=device,
    )

    # Should have both val and train metrics
    assert "val_auroc" in metrics, f"Missing val_auroc in {list(metrics.keys())}"
    assert "train_loss" in metrics, f"Missing train_loss in {list(metrics.keys())}"
    assert "val_loss" in metrics, f"Missing val_loss in {list(metrics.keys())}"

    print(f"\n[test_run_inline_probe]")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
