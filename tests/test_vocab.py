"""
Tests for data/vocab.py

No parquet files required — uses build_vocab_from_codes() with synthetic lists.
Prints vocab samples for visual inspection.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vocab import Vocab, UNK_TOKEN, build_vocab_from_codes


SYNTHETIC_CODES = (
    ["LAB//50882//mEq/L"] * 100 +
    ["DIAGNOSIS//ICD//9//25000"] * 80 +
    ["GENDER//F"] * 60 +
    ["GENDER//M"] * 55 +
    ["AGE"] * 200 +
    ["BMI"] * 150 +
    ["RACE//WHITE"] * 40 +
    ["LAB//51221//g/dL"] * 30 +
    ["PROCEDURE//ICD//9//9904"] * 5 +
    ["RARE_CODE_A"] * 2 +
    ["RARE_CODE_B"] * 1
)


def test_learned_top_k():
    vocab = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="learned", top_k=5)
    # Should have 5 codes + UNK = 6 total
    assert len(vocab) == 6, f"Expected 6, got {len(vocab)}"
    # Most frequent code (AGE, 200 times) must be in vocab
    assert vocab.encode("AGE") != vocab.unk_idx, "AGE should be in top-5 vocab"
    # Rare code should map to UNK
    assert vocab.encode("RARE_CODE_B") == vocab.unk_idx
    print(f"[test_learned_top_k] PASS — vocab size={len(vocab)}, unk_idx={vocab.unk_idx}")


def test_learned_unk_fallback():
    vocab = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="learned", top_k=5)
    unseen = vocab.encode("TOTALLY_UNSEEN_CODE")
    assert unseen == vocab.unk_idx, f"Unseen code should map to unk_idx={vocab.unk_idx}"
    print(f"[test_learned_unk_fallback] PASS — unseen code → {unseen}")


def test_text_based_all_codes():
    vocab = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="text_based")
    unique_codes = set(SYNTHETIC_CODES)
    # Every unique code plus UNK
    expected_size = len(unique_codes) + 1
    assert len(vocab) == expected_size, (
        f"Expected {expected_size} entries (all unique + UNK), got {len(vocab)}"
    )
    # Even rare codes should be present
    assert vocab.encode("RARE_CODE_A") != vocab.unk_idx
    assert vocab.encode("RARE_CODE_B") != vocab.unk_idx
    print(f"[test_text_based_all_codes] PASS — vocab size={len(vocab)} "
          f"(all {len(unique_codes)} unique codes + UNK)")


def test_decode_round_trip():
    vocab = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="text_based")
    for code in set(SYNTHETIC_CODES):
        idx = vocab.encode(code)
        decoded = vocab.decode(idx)
        assert decoded == code, f"Round-trip failed: {code!r} → {idx} → {decoded!r}"
    print("[test_decode_round_trip] PASS")


def test_unk_token_present():
    vocab = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="learned", top_k=10)
    assert UNK_TOKEN in vocab.code_to_idx
    assert vocab.code_to_idx[UNK_TOKEN] == vocab.unk_idx
    print(f"[test_unk_token_present] PASS — UNK_TOKEN at index {vocab.unk_idx}")


def test_save_load_round_trip():
    vocab = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="learned", top_k=8)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        vocab.save(path)
        loaded = Vocab.load(path)
        assert len(vocab) == len(loaded), "Vocab size mismatch after load"
        assert vocab.unk_idx == loaded.unk_idx, "UNK index mismatch after load"
        for code, idx in vocab.code_to_idx.items():
            assert loaded.encode(code) == idx, f"Mismatch for code {code!r}"
        print(f"[test_save_load_round_trip] PASS — saved/loaded vocab with {len(vocab)} entries")
    finally:
        os.unlink(path)


def test_vocab_size_property():
    vocab = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="learned", top_k=5)
    assert vocab.vocab_size == 6     # 5 codes + UNK
    assert vocab.num_codes == 5
    print(f"[test_vocab_size_property] PASS — vocab_size={vocab.vocab_size}, num_codes={vocab.num_codes}")


def test_sample_output():
    """Print vocab samples for visual inspection."""
    vocab_learned = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="learned", top_k=6)
    vocab_text = build_vocab_from_codes(SYNTHETIC_CODES, embedding_type="text_based")

    print("\n--- Learned vocab (top-6) ---")
    for code, idx in sorted(vocab_learned.code_to_idx.items(), key=lambda x: x[1]):
        marker = " ← UNK" if idx == vocab_learned.unk_idx else ""
        print(f"  {idx:3d}  {code}{marker}")

    print(f"\n--- Text-based vocab (all codes, {len(vocab_text)} entries) ---")
    sample_items = list(sorted(vocab_text.code_to_idx.items(), key=lambda x: x[1]))[:8]
    for code, idx in sample_items:
        marker = " ← UNK" if idx == vocab_text.unk_idx else ""
        print(f"  {idx:3d}  {code}{marker}")
    print(f"  ... ({len(vocab_text)} total entries)")


if __name__ == "__main__":
    test_learned_top_k()
    test_learned_unk_fallback()
    test_text_based_all_codes()
    test_decode_round_trip()
    test_unk_token_present()
    test_save_load_round_trip()
    test_vocab_size_property()
    test_sample_output()
    print("\nAll vocab tests passed.")
