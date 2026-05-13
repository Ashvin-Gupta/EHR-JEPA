"""
Supervised binary classification: embedding → encoder → context perceiver → head.

No JEPA masking, target pathway, or predictor. Used for end-to-end downstream
training with gradients through the encoder and embeddings.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from models.event_embedding import EventEmbedding
from models.latent_pooling import LatentCrossAttentionPool
from models.transformer_encoder import EHRTransformerEncoder


HeadType = Literal["linear", "mlp"]


def build_classification_head(
    in_dim: int,
    d_model: int,
    head_type: HeadType,
    dropout: float,
) -> nn.Module:
    if head_type == "linear":
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )
    if head_type == "mlp":
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
    raise ValueError(f"Unknown head_type {head_type!r}, expected 'linear' or 'mlp'")


class SupervisedPerceiverClassifier(nn.Module):
    """
    Full-sequence forward: pool the encoded sequence with LatentCrossAttentionPool
    (same module type as JEPA context pooler).
    """

    def __init__(
        self,
        embedding: EventEmbedding,
        encoder: EHRTransformerEncoder,
        pooler: LatentCrossAttentionPool,
        head_type: HeadType = "linear",
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooler is None:
            raise ValueError("SupervisedPerceiverClassifier requires a LatentCrossAttentionPool")
        self.embedding = embedding
        self.encoder = encoder
        self.pooler = pooler
        d_model = encoder.config.d_model
        n_latents = pooler.latent_tokens.shape[0]
        in_dim = n_latents * d_model
        self.head = build_classification_head(
            in_dim, d_model, head_type, head_dropout
        )

    def forward(
        self,
        codes: torch.Tensor,
        attention_mask: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : FloatTensor (B,)  — BCEWithLogits targets
        """
        x = self.embedding(
            codes,
            values=values,
            z_scores=z_scores,
            delta_times=delta_times,
            value_mask=value_mask.float() if value_mask is not None else None,
        )
        h = self.encoder(x, attention_mask=attention_mask)
        pad_mask = attention_mask == 0
        z = self.pooler(h, key_padding_mask=pad_mask)
        z_flat = z.flatten(1)
        return self.head(z_flat).squeeze(-1)
