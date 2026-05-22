"""
Autoregressive (next-token) collator for EHR event sequences.

Wraps MEDSCollator (windowing / padding), then optionally packs multiple
subject sequences end-to-end into each batch row:

    [events_seq_1 | events_seq_2 | …]   (events only; CLS/EOS added in the model)

When ``pack_sequences`` is True, consecutive samples in a batch are greedily
concatenated until the next subject would exceed ``max_len`` event tokens.

Output batch dict — all fields from MEDSCollator, plus:
    "segment_starts"  LongTensor [B, S]  — event start index per segment (-1 pad)
    "segment_lengths" LongTensor [B, S] — event count per segment (-1 pad)

The model prepends [CLS] and appends [EOS] per segment and builds next-token
labels internally.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch

from data.collator import MEDSCollator


class ARCollator:
    """
    Parameters
    ----------
    pad_idx:
        Padding index (passed to MEDSCollator).
    max_len:
        Maximum event tokens per row (after packing).
    pack_sequences:
        If True, concatenate multiple subjects per batch row up to ``max_len``.
    seed:
        Optional seed for reproducible windowing / packing (tests only).
    """

    def __init__(
        self,
        pad_idx: int,
        max_len: int,
        pack_sequences: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self._inner = MEDSCollator(pad_idx=pad_idx, max_len=max_len, task="pretrain")
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.pack_sequences = pack_sequences
        self._rng = random.Random(seed)

    def _pack_batch(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Greedy pack batch rows into fewer rows (events concatenated)."""
        codes = out["codes"]
        attn = out["attention_mask"]
        B, L = codes.shape

        B, L = codes.shape
        extras = {
            k: out[k]
            for k in out
            if k not in ("codes", "attention_mask")
            and torch.is_tensor(out[k])
            and out[k].dim() == 2
            and out[k].shape[0] == B
            and out[k].shape[1] == L
        }

        packed_codes: List[torch.Tensor] = []
        packed_attn: List[torch.Tensor] = []
        packed_extras: Dict[str, List[torch.Tensor]] = {k: [] for k in extras}
        seg_starts_list: List[List[int]] = []
        seg_lens_list: List[List[int]] = []

        b = 0
        while b < B:
            row_codes: List[int] = []
            row_attn: List[int] = []
            row_extras: Dict[str, List] = {k: [] for k in extras}
            seg_starts: List[int] = []
            seg_lens: List[int] = []

            while b < B:
                real = (attn[b] == 1).nonzero(as_tuple=True)[0]
                if real.numel() == 0:
                    b += 1
                    continue
                seg_len = int(real.numel())
                if row_codes and len(row_codes) + seg_len > self.max_len:
                    break

                seg_starts.append(len(row_codes))
                seg_lens.append(seg_len)
                row_codes.extend(codes[b, real].tolist())
                row_attn.extend([1] * seg_len)

                for k, v in extras.items():
                    row_extras[k].extend(v[b, real].tolist())

                b += 1
                if not self.pack_sequences:
                    break

            if not row_codes:
                continue

            pad_n = self.max_len - len(row_codes)
            row_codes.extend([self.pad_idx] * pad_n)
            row_attn.extend([0] * pad_n)

            packed_codes.append(torch.tensor(row_codes, dtype=codes.dtype))
            packed_attn.append(torch.tensor(row_attn, dtype=attn.dtype))
            seg_starts_list.append(seg_starts)
            seg_lens_list.append(seg_lens)

            for k, v in extras.items():
                padded = row_extras[k] + [0.0] * pad_n
                packed_extras[k].append(torch.tensor(padded, dtype=v.dtype))

        if not packed_codes:
            return out

        new_B = len(packed_codes)
        max_segs = max(len(s) for s in seg_starts_list)

        seg_starts = torch.full((new_B, max_segs), -1, dtype=torch.long)
        seg_lengths = torch.full((new_B, max_segs), -1, dtype=torch.long)
        for i, (starts, lens) in enumerate(zip(seg_starts_list, seg_lens_list)):
            for j, (st, ln) in enumerate(zip(starts, lens)):
                seg_starts[i, j] = st
                seg_lengths[i, j] = ln

        new_out: Dict[str, Any] = {
            "codes": torch.stack(packed_codes),
            "attention_mask": torch.stack(packed_attn),
            "segment_starts": seg_starts,
            "segment_lengths": seg_lengths,
        }
        for k, rows in packed_extras.items():
            new_out[k] = torch.stack(rows)
        for k in out:
            if k not in new_out and k not in ("codes", "attention_mask"):
                new_out[k] = out[k]
        return new_out

    def _single_segment_metadata(self, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One segment per row: start=0, length=number of real tokens."""
        B = attn.shape[0]
        lengths = attn.sum(dim=1).long()
        seg_starts = torch.zeros(B, 1, dtype=torch.long)
        seg_lengths = lengths.unsqueeze(1)
        return seg_starts, seg_lengths

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = self._inner(batch)

        if self.pack_sequences and out["codes"].shape[0] > 1:
            out = self._pack_batch(out)
        else:
            seg_starts, seg_lengths = self._single_segment_metadata(out["attention_mask"])
            out["segment_starts"] = seg_starts
            out["segment_lengths"] = seg_lengths

        return out
