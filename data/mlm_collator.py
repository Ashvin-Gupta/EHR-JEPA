"""
BERT-style Masked Language Model collator for EHR event sequences.

Wraps MEDSCollator (which handles windowing / padding) and then applies
token-level masking to the resulting batch.

Masking scheme (applied only to real, non-padding tokens):
    mask_ratio * mask_token_frac  →  replace code with mask_token_idx
    mask_ratio * random_frac      →  replace code with a random vocab index
    remainder                     →  leave code unchanged

Default values (matching the user specification):
    mask_ratio        = 0.15   (15 % of real tokens are selected)
    mask_token_frac   = 12.5/15 ≈ 0.833  (12.5 % of total → [MASK])
    random_frac       = 2.5/15 ≈ 0.167   (2.5 % of total → random)

Output batch dict — all fields from MEDSCollator, plus:
    "input_codes"  LongTensor [B, L]  masked codes fed to the model
    "mlm_labels"   LongTensor [B, L]  original codes at selected positions,
                                       -100 elsewhere (cross-entropy ignore index)

The original "codes" field is preserved (unmasked) in the batch so callers
can inspect or evaluate against the ground truth.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from data.collator import MEDSCollator

if TYPE_CHECKING:
    from data.vocab import Vocab


class MLMCollator:
    """
    Parameters
    ----------
    pad_idx:
        Vocabulary index used for padding (passed to MEDSCollator).
    mask_token_idx:
        Vocabulary index used as the [MASK] replacement token.
        Typically set to vocab_size (a special index outside the normal range).
    vocab_size:
        Number of real vocabulary entries.  Used to sample random replacement
        tokens uniformly from [0, vocab_size).
    max_len:
        Context window length (passed to MEDSCollator).
    mask_ratio:
        Fraction of real tokens selected for masking.  Default 0.15.
    mask_token_frac:
        Of the selected tokens, this fraction are replaced with mask_token_idx.
        Default 12.5/15 ≈ 0.833.
    random_frac:
        Of the selected tokens, this fraction are replaced with a random token.
        Default 2.5/15 ≈ 0.167.
        The remaining (1 - mask_token_frac - random_frac) are kept unchanged.
    seed:
        Optional seed for reproducible masking (testing only).
    """

    def __init__(
        self,
        pad_idx: int,
        mask_token_idx: int,
        vocab_size: int,
        max_len: int,
        mask_ratio: float = 0.15,
        mask_token_frac: float = 12.5 / 15.0,
        random_frac: float = 2.5 / 15.0,
        seed: Optional[int] = None,
    ) -> None:
        self._inner = MEDSCollator(pad_idx=pad_idx, max_len=max_len, task="pretrain")
        self.mask_token_idx  = mask_token_idx
        self.vocab_size      = vocab_size
        self.mask_ratio      = mask_ratio
        self.mask_token_frac = mask_token_frac
        self.random_frac     = random_frac
        self._rng = random.Random(seed)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Step 1: delegate windowing / padding to MEDSCollator
        out = self._inner(batch)

        codes        = out["codes"]           # (B, L) original token indices
        attn_mask    = out["attention_mask"]  # (B, L) 1=real, 0=pad

        B, L = codes.shape
        input_codes = codes.clone()
        mlm_labels  = torch.full_like(codes, -100)  # -100 = ignored by cross-entropy

        for b in range(B):
            # Indices of real (non-padding) tokens
            real_positions = (attn_mask[b] == 1).nonzero(as_tuple=True)[0].tolist()
            if not real_positions:
                continue

            n_mask = max(1, round(len(real_positions) * self.mask_ratio))
            selected = self._rng.sample(real_positions, min(n_mask, len(real_positions)))

            for pos in selected:
                original_code = codes[b, pos].item()
                mlm_labels[b, pos] = original_code

                r = self._rng.random()
                if r < self.mask_token_frac:
                    input_codes[b, pos] = self.mask_token_idx
                elif r < self.mask_token_frac + self.random_frac:
                    input_codes[b, pos] = self._rng.randrange(self.vocab_size)
                # else: keep original (unchanged)

        out["input_codes"] = input_codes
        out["mlm_labels"]  = mlm_labels
        return out
