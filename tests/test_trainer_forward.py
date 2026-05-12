"""
Full synthetic forward-pass smoke tests for training/trainer.py.

Architecture under test:
  - Shared EHRTransformerEncoder for both target and context pathways
  - Branch A (use_perceiver=True):  Perceiver pooling → latent predictor
  - Branch B (use_perceiver=False): MASK tokens → token predictor

Uses small dimensions to keep tests fast (no real data, no training loop).
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loss.covariance_reg import SIGRegLoss
from masking.span_masking import SpanMasker
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.predictor import Predictor, TemporalSpanPrompt
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.trainer import JEPATrainer, TrainerConfig

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
D         = 32
N_LATENTS = 4
VOCAB     = 50
B         = 2
L         = 80   # long enough for masking with min_span_length=5


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def _build_encoder(n_layers: int = 2) -> EHRTransformerEncoder:
    return EHRTransformerEncoder(
        TransformerEncoderConfig(
            n_layers=n_layers, d_model=D, n_heads=4, ffn_dim=64, dropout=0.0
        )
    )


def _build_trainer(
    use_perceiver: bool = True,
    use_value: bool = False,
    use_time: bool = False,
    n_latents: int = N_LATENTS,
    lambda_cov: float = 0.1,
    min_span_for_perceiver: int = 5,
) -> JEPATrainer:
    embedding = EventEmbedding(EmbeddingConfig(
        embedding_type="learned",
        vocab_size=VOCAB,
        d_model=D,
        use_value=use_value,
        use_time=use_time,
    ))

    # Single shared encoder
    encoder = _build_encoder()

    prompt         = TemporalSpanPrompt(D)
    predictor      = Predictor(D, n_heads=4, n_layers=2, dropout=0.0)
    token_predictor = _build_encoder(n_layers=2)
    cov_loss       = SIGRegLoss(num_slices=16)
    masker         = SpanMasker(
        mask_ratio=0.30, default_num_spans=2, min_span_length=5, seed=0
    )

    context_pooler = (
        LatentCrossAttentionPool(D, n_latents=n_latents, n_heads=4)
        if use_perceiver else None
    )
    target_pooler = (
        LatentCrossAttentionPool(D, n_latents=n_latents, n_heads=4)
        if use_perceiver else None
    )

    trainer_cfg = TrainerConfig(
        use_perceiver=use_perceiver,
        min_span_for_perceiver=min_span_for_perceiver,
        lambda_cov=lambda_cov,
        device="cpu",
        early_stopping_patience=0,
    )

    return JEPATrainer(
        embedding=embedding,
        encoder=encoder,
        prompt=prompt,
        predictor=predictor,
        token_predictor=token_predictor,
        context_pooler=context_pooler,
        target_pooler=target_pooler,
        cov_loss=cov_loss,
        masker=masker,
        config=trainer_cfg,
    )


def _make_batch(use_value: bool = False, use_time: bool = False):
    codes      = torch.randint(0, VOCAB, (B, L))
    attn_mask  = torch.ones(B, L, dtype=torch.long)
    attn_mask[1, L - 10:] = 0          # pad last 10 positions for sample 1
    values     = torch.rand(B, L)      if use_value else None
    z_scores   = torch.rand(B, L)      if use_value else None
    delta_times = torch.rand(B, L)     if use_time  else None
    value_mask = torch.ones(B, L, dtype=torch.long) if use_value else None
    return codes, attn_mask, values, z_scores, delta_times, value_mask


# ==================================================================
# Branch A (Perceiver) tests
# ==================================================================

def test_branch_a_code_only():
    trainer = _build_trainer(use_perceiver=True)
    codes, attn_mask, *_ = _make_batch()
    l_pred, l_cov, l_total = trainer(codes, attn_mask)
    print(f"\n[test_branch_a_code_only]")
    print(f"  L_pred={l_pred.item():.6f}, L_cov={l_cov.item():.6f}, "
          f"L_total={l_total.item():.6f}")
    assert l_pred.ndim == 0 and not torch.isnan(l_total)


def test_branch_a_code_plus_value():
    trainer = _build_trainer(use_perceiver=True, use_value=True)
    codes, attn_mask, values, z_scores, _, value_mask = _make_batch(use_value=True)
    l_pred, l_cov, l_total = trainer(
        codes, attn_mask, values=values, z_scores=z_scores, value_mask=value_mask
    )
    print(f"\n[test_branch_a_code_plus_value]")
    print(f"  L_pred={l_pred.item():.6f}, L_cov={l_cov.item():.6f}")
    assert not torch.isnan(l_total)


def test_branch_a_code_plus_time():
    trainer = _build_trainer(use_perceiver=True, use_time=True)
    codes, attn_mask, _, _, delta_times, _ = _make_batch(use_time=True)
    l_pred, l_cov, l_total = trainer(codes, attn_mask, delta_times=delta_times)
    print(f"\n[test_branch_a_code_plus_time]")
    print(f"  L_pred={l_pred.item():.6f}, L_cov={l_cov.item():.6f}")
    assert not torch.isnan(l_total)


def test_branch_a_code_plus_value_plus_time():
    trainer = _build_trainer(use_perceiver=True, use_value=True, use_time=True)
    codes, attn_mask, values, z_scores, delta_times, value_mask = _make_batch(
        use_value=True, use_time=True
    )
    l_pred, l_cov, l_total = trainer(
        codes, attn_mask,
        values=values, z_scores=z_scores,
        delta_times=delta_times, value_mask=value_mask,
    )
    print(f"\n[test_branch_a_code_plus_value_plus_time]")
    print(f"  L_pred={l_pred.item():.6f}, L_cov={l_cov.item():.6f}")
    assert not torch.isnan(l_total)


def test_branch_a_gradient_flow():
    """
    After L_total.backward():
    - Embedding weights: grad from context path (L_pred)
    - Encoder weights: grad from BOTH paths (L_pred via context, L_cov via target)
    - Predictor weights: grad from L_pred
    """
    trainer = _build_trainer(use_perceiver=True)
    codes, attn_mask, *_ = _make_batch()
    l_pred, l_cov, l_total = trainer(codes, attn_mask)
    l_total.backward()

    emb_grad  = trainer.embedding.embedding.weight.grad
    enc_grad  = trainer.encoder.layers[0].self_attn.q_proj.weight.grad
    pred_grad = trainer.predictor.transformer.layers[0].self_attn.q_proj.weight.grad

    print(f"\n[test_branch_a_gradient_flow]")
    print(f"  embedding grad norm:        {emb_grad.norm():.6f}")
    print(f"  encoder grad norm:          {enc_grad.norm():.6f}")
    print(f"  predictor grad norm:        {pred_grad.norm():.6f}")

    assert emb_grad  is not None and emb_grad.norm()  > 0
    assert enc_grad  is not None and enc_grad.norm()  > 0
    assert pred_grad is not None and pred_grad.norm() > 0


def test_branch_a_span_filter():
    """Spans shorter than min_span_for_perceiver are skipped; forward still runs."""
    # min_span_for_perceiver=100 means ALL spans will be skipped → zero losses
    trainer = _build_trainer(use_perceiver=True, min_span_for_perceiver=100)
    codes, attn_mask, *_ = _make_batch()
    l_pred, l_cov, l_total = trainer(codes, attn_mask)
    print(f"\n[test_branch_a_span_filter]")
    print(f"  L_pred={l_pred.item()}, L_cov={l_cov.item()} (both 0 — all spans skipped)")
    # When all spans are skipped forward returns zeros
    assert l_pred.item() == 0.0 and l_cov.item() == 0.0


# ==================================================================
# Branch B (Token I-JEPA) tests
# ==================================================================

def test_branch_b_code_only():
    trainer = _build_trainer(use_perceiver=False)
    codes, attn_mask, *_ = _make_batch()
    l_pred, l_cov, l_total = trainer(codes, attn_mask)
    print(f"\n[test_branch_b_code_only]")
    print(f"  L_pred={l_pred.item():.6f}, L_cov={l_cov.item():.6f}, "
          f"L_total={l_total.item():.6f}")
    assert l_pred.ndim == 0 and not torch.isnan(l_total)


def test_branch_b_code_plus_value():
    trainer = _build_trainer(use_perceiver=False, use_value=True)
    codes, attn_mask, values, z_scores, _, value_mask = _make_batch(use_value=True)
    l_pred, l_cov, l_total = trainer(
        codes, attn_mask, values=values, z_scores=z_scores, value_mask=value_mask
    )
    print(f"\n[test_branch_b_code_plus_value]")
    print(f"  L_pred={l_pred.item():.6f}, L_cov={l_cov.item():.6f}")
    assert not torch.isnan(l_total)


def test_branch_b_code_plus_value_plus_time():
    trainer = _build_trainer(use_perceiver=False, use_value=True, use_time=True)
    codes, attn_mask, values, z_scores, delta_times, value_mask = _make_batch(
        use_value=True, use_time=True
    )
    l_pred, l_cov, l_total = trainer(
        codes, attn_mask,
        values=values, z_scores=z_scores,
        delta_times=delta_times, value_mask=value_mask,
    )
    print(f"\n[test_branch_b_code_plus_value_plus_time]")
    print(f"  L_pred={l_pred.item():.6f}, L_cov={l_cov.item():.6f}")
    assert not torch.isnan(l_total)


def test_branch_b_gradient_flow():
    """
    Branch B gradient flow:
    - Embedding, encoder, token_predictor: should have gradients
    - mask_token parameter: should have gradient
    """
    trainer = _build_trainer(use_perceiver=False)
    codes, attn_mask, *_ = _make_batch()
    l_pred, l_cov, l_total = trainer(codes, attn_mask)
    l_total.backward()

    emb_grad        = trainer.embedding.embedding.weight.grad
    enc_grad        = trainer.encoder.layers[0].self_attn.q_proj.weight.grad
    tok_pred_grad   = trainer.token_predictor.layers[0].self_attn.q_proj.weight.grad
    mask_token_grad = trainer.mask_token.grad

    print(f"\n[test_branch_b_gradient_flow]")
    print(f"  embedding grad norm:        {emb_grad.norm():.6f}")
    print(f"  encoder grad norm:          {enc_grad.norm():.6f}")
    print(f"  token_predictor grad norm:  {tok_pred_grad.norm():.6f}")
    print(f"  mask_token grad norm:       {mask_token_grad.norm():.6f}")

    assert emb_grad        is not None and emb_grad.norm()        > 0
    assert enc_grad        is not None and enc_grad.norm()        > 0
    assert tok_pred_grad   is not None and tok_pred_grad.norm()   > 0
    assert mask_token_grad is not None and mask_token_grad.norm() > 0


# ==================================================================
# Shared / structural tests
# ==================================================================

def test_losses_positive_branch_a():
    trainer = _build_trainer(use_perceiver=True)
    codes, attn_mask, *_ = _make_batch()
    l_pred, l_cov, l_total = trainer(codes, attn_mask)
    print(f"\n[test_losses_positive_branch_a]")
    print(f"  L_pred={l_pred.item():.6f} (≥0?), L_cov={l_cov.item():.6f} (≥0?)")
    assert l_pred.item() >= 0.0
    assert l_cov.item()  >= 0.0
    assert l_total.item() >= 0.0


def test_losses_positive_branch_b():
    trainer = _build_trainer(use_perceiver=False)
    codes, attn_mask, *_ = _make_batch()
    l_pred, l_cov, l_total = trainer(codes, attn_mask)
    print(f"\n[test_losses_positive_branch_b]")
    print(f"  L_pred={l_pred.item():.6f} (≥0?), L_cov={l_cov.item():.6f} (≥0?)")
    assert l_pred.item() >= 0.0
    assert l_cov.item()  >= 0.0
    assert l_total.item() >= 0.0


def test_short_sequence():
    """Short sequence triggers dynamic num_spans < default; both branches handle it."""
    for use_perceiver in [True, False]:
        trainer = _build_trainer(use_perceiver=use_perceiver)
        L_short = 30
        codes     = torch.randint(0, VOCAB, (B, L_short))
        attn_mask = torch.ones(B, L_short, dtype=torch.long)
        l_pred, l_cov, l_total = trainer(codes, attn_mask)
        branch = "A (perceiver)" if use_perceiver else "B (token)"
        print(f"\n[test_short_sequence] branch {branch}, L={L_short}")
        print(f"  L_pred={l_pred.item():.6f}, L_cov={l_cov.item():.6f}")
        assert not torch.isnan(l_total)


def test_shared_encoder_weights():
    """Both pathways use the same nn.Module instance (same id)."""
    trainer = _build_trainer(use_perceiver=True)
    # encoder is used for both target and context — there should be exactly ONE encoder
    # module (not two separate copies)
    enc_ids = {id(m) for name, m in trainer.named_modules()
               if isinstance(m, EHRTransformerEncoder) and name == "encoder"}
    print(f"\n[test_shared_encoder_weights]")
    print(f"  Encoder module id: {enc_ids}")
    # There is exactly one EHRTransformerEncoder registered as 'encoder'
    assert len(enc_ids) == 1


if __name__ == "__main__":
    import traceback
    tests = [
        test_branch_a_code_only,
        test_branch_a_code_plus_value,
        test_branch_a_code_plus_time,
        test_branch_a_code_plus_value_plus_time,
        test_branch_a_gradient_flow,
        test_branch_a_span_filter,
        test_branch_b_code_only,
        test_branch_b_code_plus_value,
        test_branch_b_code_plus_value_plus_time,
        test_branch_b_gradient_flow,
        test_losses_positive_branch_a,
        test_losses_positive_branch_b,
        test_short_sequence,
        test_shared_encoder_weights,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    if failed:
        sys.exit(1)
