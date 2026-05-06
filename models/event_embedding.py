"""
Event embedding module.

Converts a batch of code indices (LongTensor [B, L]) into dense embedding
vectors (FloatTensor [B, L, d_model]).

Two base embedding modes, controlled by config.embedding_type:

  "learned"
    Standard nn.Embedding table of shape (vocab_size, d_model).
    Randomly initialised and trained end-to-end.

  "text_based"
    Pre-computed embeddings produced offline by a clinical language model
    (e.g. ClinicalBERT, BioBERT) and saved as a .pt file.  A trainable
    linear projection maps encoder_hidden_dim → d_model.  The raw embeddings
    are frozen; only the projection is trained.

Four operating modes for numeric/time features, controlled by
config.use_value and config.use_time:

  code only (use_value=False, use_time=False)
    No MLP.  Output = embedding(codes).

  code + value (use_value=True, use_time=False)
    MLP input: [E_code, value, z_score, value_mask]    (N_extra = 3)

  code + time (use_value=False, use_time=True)
    MLP input: [E_code, delta_time]                    (N_extra = 1)

  code + value + time (use_value=True, use_time=True)
    MLP input: [E_code, value, z_score, delta_time, value_mask]  (N_extra = 4)

MLP architecture (instantiated only when N_extra > 0):
    Linear(d_model + N_extra → d_model) → GELU → Linear(d_model → d_model)

Residual integration:
    x = LayerNorm(E_code + MLP([E_code, *active_features]))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EmbeddingConfig:
    """Flat configuration object for EventEmbedding."""
    embedding_type: str = "learned"       # "learned" | "text_based"
    vocab_size: int = 5001                # includes UNK slot
    d_model: int = 256
    # text_based specific
    code_embeddings_path: Optional[str] = None   # path to .pt tensor file
    encoder_hidden_dim: int = 768                 # ClinicalBERT default
    unk_idx: int = 5000
    # feature flags — control MLP construction
    use_value: bool = False
    use_time: bool = False


class EventEmbedding(nn.Module):
    """
    Maps code index sequences to d_model embedding vectors, optionally
    incorporating numeric values and/or time deltas via a residual MLP.

    Parameters
    ----------
    config:
        EmbeddingConfig (or any object with the same attributes).
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

        # --- Base code embedding ---
        if config.embedding_type == "learned":
            self.embedding = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.d_model,
                padding_idx=None,
            )
            self.projection: Optional[nn.Linear] = None

        elif config.embedding_type == "text_based":
            if config.code_embeddings_path is None:
                raise ValueError(
                    "config.code_embeddings_path must be set for embedding_type='text_based'"
                )
            raw: torch.Tensor = torch.load(
                config.code_embeddings_path, map_location="cpu"
            )
            if raw.shape[0] != config.vocab_size:
                raise ValueError(
                    f"Embedding file has {raw.shape[0]} rows but vocab_size={config.vocab_size}"
                )
            unk_row = raw[config.unk_idx]
            if unk_row.abs().sum().item() == 0.0:
                non_unk = torch.cat(
                    [raw[: config.unk_idx], raw[config.unk_idx + 1 :]], dim=0
                )
                raw[config.unk_idx] = non_unk.mean(dim=0)

            self.embedding = nn.Embedding.from_pretrained(
                raw.float(), freeze=True, padding_idx=None
            )
            self.projection = nn.Linear(config.encoder_hidden_dim, config.d_model)

        else:
            raise ValueError(
                f"embedding_type must be 'learned' or 'text_based', "
                f"got '{config.embedding_type}'"
            )

        # --- Numeric / time MLP ---
        n_extra = self._n_extra()
        if n_extra > 0:
            self.mlp = nn.Sequential(
                nn.Linear(config.d_model + n_extra, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.layer_norm = nn.LayerNorm(config.d_model)
        else:
            self.mlp = None        # type: ignore[assignment]
            self.layer_norm = None # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _n_extra(self) -> int:
        """Number of extra scalar features appended to E_code in the MLP."""
        if self.config.use_value and self.config.use_time:
            return 4   # value, z_score, delta_time, value_mask
        elif self.config.use_value:
            return 3   # value, z_score, value_mask
        elif self.config.use_time:
            return 1   # delta_time
        return 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        codes: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        z_scores: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        codes:
            LongTensor (B, L) — vocab indices.
        values:
            FloatTensor (B, L) — raw numeric values (0.0 for missing).
        z_scores:
            FloatTensor (B, L) — z-scored values (0.0 for missing).
        delta_times:
            FloatTensor (B, L) — log(1 + hours_since_prev).
        value_mask:
            FloatTensor (B, L) — 1.0 if value is present, 0.0 if missing.

        Returns
        -------
        FloatTensor (B, L, d_model).
        """
        x = self.embedding(codes)                  # (B, L, embed_dim)
        if self.projection is not None:
            x = self.projection(x)                 # (B, L, d_model)

        if self.mlp is None:
            return x

        # Build feature list for MLP input
        features = [x]
        if self.config.use_value:
            B, L = x.shape[:2]
            v = values if values is not None else x.new_zeros(B, L)
            z = z_scores if z_scores is not None else x.new_zeros(B, L)
            vm = (
                value_mask.float()
                if value_mask is not None
                else x.new_zeros(B, L)
            )
            features.append(v.unsqueeze(-1))
            features.append(z.unsqueeze(-1))
            if self.config.use_time:
                dt = delta_times if delta_times is not None else x.new_zeros(B, L)
                features.append(dt.unsqueeze(-1))
            features.append(vm.unsqueeze(-1))
        elif self.config.use_time:
            B, L = x.shape[:2]
            dt = delta_times if delta_times is not None else x.new_zeros(B, L)
            features.append(dt.unsqueeze(-1))

        mlp_in = torch.cat(features, dim=-1)        # (B, L, d_model + N_extra)
        x = self.layer_norm(x + self.mlp(mlp_in))  # residual + LayerNorm
        return x
