"""Tests for SupervisedPerceiverClassifier and BERTSupervisedClassifier."""

from __future__ import annotations

import torch

from evaluation.bert_supervised import BERTSupervisedClassifier
from evaluation.supervised_cls import SupervisedCLSClassifier
from evaluation.supervised_perceiver import SupervisedPerceiverClassifier
from models.event_embedding import EmbeddingConfig, EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.transformer_encoder import EHRTransformerEncoder, TransformerEncoderConfig
from training.bert_trainer import BERTEHRModel


def _tiny_vocab_embedding(d_model: int = 16, vocab_size: int = 32) -> EventEmbedding:
    return EventEmbedding(
        EmbeddingConfig(
            embedding_type="learned",
            vocab_size=vocab_size,
            d_model=d_model,
            unk_idx=vocab_size - 1,
            use_value=False,
            use_time=False,
        )
    )


def _tiny_encoder(d_model: int = 16) -> EHRTransformerEncoder:
    return EHRTransformerEncoder(
        TransformerEncoderConfig(
            n_layers=1, d_model=d_model, n_heads=2, ffn_dim=32, dropout=0.0
        )
    )


def test_low_data_subset_size_formula():
    import math

    n = 1000
    for frac, expected in ((0.01, 10), (0.05, 50), (1.0, 1000)):
        k = min(n, max(1, int(math.ceil(frac * n))))
        assert k == expected


def test_supervised_perceiver_forward_backward():
    d = 16
    emb = _tiny_vocab_embedding(d)
    enc = _tiny_encoder(d)
    pool = LatentCrossAttentionPool(d, n_latents=2, n_heads=2)
    model = SupervisedPerceiverClassifier(emb, enc, pool, head_type="mlp", head_dropout=0.0)
    B, L = 2, 12
    codes = torch.randint(0, 31, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    labels = torch.tensor([0.0, 1.0])
    logits = model(codes, attn)
    assert logits.shape == (B,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert emb.embedding.weight.grad is not None


def test_supervised_cls_forward_backward():
    d = 16
    emb = _tiny_vocab_embedding(d)
    enc = _tiny_encoder(d)
    cls_token = torch.nn.Parameter(torch.randn(d) * 0.02)
    model = SupervisedCLSClassifier(emb, enc, cls_token, head_type="linear", head_dropout=0.0)
    B, L = 2, 10
    codes = torch.randint(0, 31, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    labels = torch.tensor([0.0, 1.0])
    logits = model(codes, attn)
    assert logits.shape == (B,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert cls_token.grad is not None


def test_bert_supervised_forward_backward():
    d = 16
    vocab_size = 20
    emb = EventEmbedding(
        EmbeddingConfig(
            embedding_type="learned",
            vocab_size=vocab_size,
            d_model=d,
            unk_idx=vocab_size - 1,
            use_value=False,
            use_time=False,
        )
    )
    enc = _tiny_encoder(d)
    bert = BERTEHRModel(emb, enc, vocab_size=vocab_size)
    model = BERTSupervisedClassifier(bert, head_type="linear", head_dropout=0.0)
    B, L = 2, 8
    codes = torch.randint(0, vocab_size - 1, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    labels = torch.tensor([1.0, 0.0])
    logits = model(codes, attn)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert bert.cls_token.grad is not None
