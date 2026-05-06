"""
Tests for models/event_embedding.py

No pre-trained weights required — uses random tensors for text_based mode.
Prints embedding shapes and first vector for visual inspection.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from models.event_embedding import EventEmbedding, EmbeddingConfig


VOCAB_SIZE = 20   # small test vocab
D_MODEL = 32
ENCODER_DIM = 64   # simulated ClinicalBERT hidden size
BATCH = 2
SEQ_LEN = 10


def make_codes() -> torch.Tensor:
    """Random code indices in range [0, VOCAB_SIZE)."""
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


# ------------------------------------------------------------------
# Learned mode
# ------------------------------------------------------------------

def test_learned_forward_shape():
    config = EmbeddingConfig(
        embedding_type="learned",
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
    )
    model = EventEmbedding(config)
    codes = make_codes()
    out = model(codes)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
        f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"
    )
    print(f"\n[test_learned_forward_shape] PASS — output shape: {out.shape}")


def test_learned_embedding_table_size():
    config = EmbeddingConfig(
        embedding_type="learned",
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
    )
    model = EventEmbedding(config)
    assert model.embedding.num_embeddings == VOCAB_SIZE
    assert model.embedding.embedding_dim == D_MODEL
    print(f"[test_learned_embedding_table_size] PASS — "
          f"table: ({model.embedding.num_embeddings}, {model.embedding.embedding_dim})")


def test_learned_is_trainable():
    config = EmbeddingConfig(
        embedding_type="learned",
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
    )
    model = EventEmbedding(config)
    assert model.embedding.weight.requires_grad, "Learned embedding should be trainable"
    print("[test_learned_is_trainable] PASS")


def test_learned_sample_output():
    config = EmbeddingConfig(
        embedding_type="learned",
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
    )
    model = EventEmbedding(config)
    codes = make_codes()
    out = model(codes)
    print(f"\n--- Learned embedding sample ---")
    print(f"  Input codes[0]: {codes[0].tolist()}")
    print(f"  Output[0, 0]: {out[0, 0].detach().tolist()}")
    print(f"  Output shape:  {out.shape}")


# ------------------------------------------------------------------
# Text-based mode
# ------------------------------------------------------------------

def make_pretrained_file(vocab_size: int, hidden_dim: int) -> str:
    """Save a random embedding tensor to a temp .pt file."""
    tensor = torch.randn(vocab_size, hidden_dim)
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(tensor, f.name)
    f.close()
    return f.name


def test_text_based_forward_shape():
    pt_file = make_pretrained_file(VOCAB_SIZE, ENCODER_DIM)
    try:
        config = EmbeddingConfig(
            embedding_type="text_based",
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            encoder_hidden_dim=ENCODER_DIM,
            code_embeddings_path=pt_file,
            unk_idx=VOCAB_SIZE - 1,
        )
        model = EventEmbedding(config)
        codes = make_codes()
        out = model(codes)

        assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
            f"Expected ({BATCH}, {SEQ_LEN}, {D_MODEL}), got {out.shape}"
        )
        print(f"\n[test_text_based_forward_shape] PASS — output shape: {out.shape}")
    finally:
        os.unlink(pt_file)


def test_text_based_embeddings_frozen():
    """The raw embedding weights should be frozen; only projection is trainable."""
    pt_file = make_pretrained_file(VOCAB_SIZE, ENCODER_DIM)
    try:
        config = EmbeddingConfig(
            embedding_type="text_based",
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            encoder_hidden_dim=ENCODER_DIM,
            code_embeddings_path=pt_file,
            unk_idx=VOCAB_SIZE - 1,
        )
        model = EventEmbedding(config)
        assert not model.embedding.weight.requires_grad, (
            "text_based embedding weights should be frozen"
        )
        assert model.projection.weight.requires_grad, (
            "projection layer should be trainable"
        )
        print("[test_text_based_embeddings_frozen] PASS")
    finally:
        os.unlink(pt_file)


def test_text_based_projection_shape():
    pt_file = make_pretrained_file(VOCAB_SIZE, ENCODER_DIM)
    try:
        config = EmbeddingConfig(
            embedding_type="text_based",
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            encoder_hidden_dim=ENCODER_DIM,
            code_embeddings_path=pt_file,
            unk_idx=VOCAB_SIZE - 1,
        )
        model = EventEmbedding(config)
        assert model.projection.in_features == ENCODER_DIM
        assert model.projection.out_features == D_MODEL
        print(f"[test_text_based_projection_shape] PASS — "
              f"projection: {ENCODER_DIM} → {D_MODEL}")
    finally:
        os.unlink(pt_file)


def test_text_based_unk_row_filled():
    """A zero UNK row in the .pt file should be replaced with the mean of other rows."""
    tensor = torch.randn(VOCAB_SIZE, ENCODER_DIM)
    unk_idx = VOCAB_SIZE - 1
    tensor[unk_idx] = 0.0   # force zero UNK row

    pt_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(tensor, pt_file.name)
    pt_file.close()

    try:
        config = EmbeddingConfig(
            embedding_type="text_based",
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            encoder_hidden_dim=ENCODER_DIM,
            code_embeddings_path=pt_file.name,
            unk_idx=unk_idx,
        )
        model = EventEmbedding(config)
        unk_emb = model.embedding.weight[unk_idx]
        assert unk_emb.abs().sum().item() > 0, (
            "UNK row should have been replaced with mean, not kept as zero"
        )
        print(f"[test_text_based_unk_row_filled] PASS — UNK norm: {unk_emb.norm().item():.4f}")
    finally:
        os.unlink(pt_file.name)


def test_text_based_wrong_vocab_size_raises():
    """Mismatched vocab size in .pt file should raise ValueError."""
    pt_file = make_pretrained_file(VOCAB_SIZE + 5, ENCODER_DIM)  # wrong size
    try:
        config = EmbeddingConfig(
            embedding_type="text_based",
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            encoder_hidden_dim=ENCODER_DIM,
            code_embeddings_path=pt_file,
            unk_idx=VOCAB_SIZE - 1,
        )
        with pytest.raises(ValueError, match="vocab_size"):
            EventEmbedding(config)
        print("[test_text_based_wrong_vocab_size_raises] PASS")
    finally:
        os.unlink(pt_file)


def test_invalid_embedding_type_raises():
    config = EmbeddingConfig(embedding_type="unknown_type", vocab_size=10, d_model=8)
    with pytest.raises(ValueError):
        EventEmbedding(config)
    print("[test_invalid_embedding_type_raises] PASS")


def test_text_based_sample_output():
    pt_file = make_pretrained_file(VOCAB_SIZE, ENCODER_DIM)
    try:
        config = EmbeddingConfig(
            embedding_type="text_based",
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            encoder_hidden_dim=ENCODER_DIM,
            code_embeddings_path=pt_file,
            unk_idx=VOCAB_SIZE - 1,
        )
        model = EventEmbedding(config)
        codes = make_codes()
        out = model(codes)
        print(f"\n--- Text-based embedding sample ---")
        print(f"  Input codes[0]: {codes[0].tolist()}")
        print(f"  Output[0, 0] (after projection): {out[0, 0].detach().tolist()[:8]} ...")
        print(f"  Output shape: {out.shape}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"  Trainable params: {trainable}  |  Frozen params: {frozen}")
    finally:
        os.unlink(pt_file)


if __name__ == "__main__":
    test_learned_forward_shape()
    test_learned_embedding_table_size()
    test_learned_is_trainable()
    test_learned_sample_output()
    test_text_based_forward_shape()
    test_text_based_embeddings_frozen()
    test_text_based_projection_shape()
    test_text_based_unk_row_filled()
    test_text_based_wrong_vocab_size_raises()
    test_invalid_embedding_type_raises()
    test_text_based_sample_output()
    print("\nAll event_embedding tests passed.")
