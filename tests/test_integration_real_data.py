"""
Integration tests against the real clean_meds_debug sample data.

Runs the full pipeline end-to-end:
  parquet load → subject sequences → vocab → dataset → collator → embedding

Run with:
    python tests/test_integration_real_data.py
or:
    pytest tests/test_integration_real_data.py -v -s

DATA_DIR defaults to /home/ag619/clean_meds_debug but can be overridden:
    DATA_DIR=/other/path pytest tests/test_integration_real_data.py -v -s
"""

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR   = os.environ.get("DATA_DIR",   "/home/ag619/clean_meds_debug")
LOOKUP_DIR = os.environ.get("LOOKUP_DIR", "/home/ag619/MIMIC_data/hosp")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.meds_parser import (
    load_split,
    build_subject_sequences,
    extract_header,
    get_age_from_header,
)
from data.vocab import build_vocab
from data.code_translator import MEDSCodeTranslator
from data.meds_dataset import MEDSDataset
from data.collator import MEDSCollator
from models.event_embedding import EventEmbedding, EmbeddingConfig


# ---------------------------------------------------------------------------
# 1. Raw data loading
# ---------------------------------------------------------------------------

def test_load_real_parquets():
    print("\n" + "="*60)
    print("TEST 1: Load real parquet files")
    print("="*60)
    for split in ["train", "tuning", "held_out"]:
        df = load_split(DATA_DIR, split)
        assert len(df) > 0, f"{split} is empty"
        assert list(df.columns) == ["subject_id", "time", "code", "numeric_value"]
        # numeric_value must be float after coercion (not str)
        assert pd.api.types.is_float_dtype(df["numeric_value"]), (
            f"numeric_value should be float after coercion, got {df['numeric_value'].dtype}"
        )
        n_subjects = df["subject_id"].nunique()
        print(f"  {split:10s}: {len(df):7,} rows | {n_subjects:4} subjects | "
              f"numeric_value dtype={df['numeric_value'].dtype}")
    print("  PASS")


# ---------------------------------------------------------------------------
# 2. Subject sequence building
# ---------------------------------------------------------------------------

def test_build_sequences():
    print("\n" + "="*60)
    print("TEST 2: Build subject sequences")
    print("="*60)
    df = load_split(DATA_DIR, "train")
    seqs = build_subject_sequences(df)

    n_subjects = len(seqs)
    seq_lengths = [len(v) for v in seqs.values()]
    print(f"  Subjects:       {n_subjects}")
    print(f"  Seq len min:    {min(seq_lengths)}")
    print(f"  Seq len max:    {max(seq_lengths)}")
    print(f"  Seq len mean:   {sum(seq_lengths)/len(seq_lengths):.1f}")

    # Every sequence must start with AGE
    bad = [(sid, seqs[sid][0].code) for sid in seqs if seqs[sid][0].code != "AGE"]
    if bad:
        print(f"  WARNING: {len(bad)} subjects don't start with AGE:")
        for sid, code in bad[:5]:
            print(f"    subject {sid} starts with '{code}'")
    else:
        print(f"  All {n_subjects} subjects start with AGE  ✓")

    # Print a sample sequence
    sample_sid = sorted(seqs.keys())[0]
    sample_events = seqs[sample_sid]
    print(f"\n  Sample subject {sample_sid} ({len(sample_events)} events):")
    for i, e in enumerate(sample_events[:8]):
        ts = str(e.time) if pd.notna(e.time) else "NaT"
        print(f"    [{i:3d}] time={ts:26s}  code={e.code!r:50s}  value={e.numeric_value}")
    if len(sample_events) > 8:
        print(f"    ... ({len(sample_events) - 8} more events)")
    print("  PASS")


# ---------------------------------------------------------------------------
# 3. Header extraction
# ---------------------------------------------------------------------------

def test_header_extraction():
    print("\n" + "="*60)
    print("TEST 3: Header extraction and AGE values")
    print("="*60)
    df = load_split(DATA_DIR, "train")
    seqs = build_subject_sequences(df)

    ages = []
    header_sizes = []
    for sid, events in seqs.items():
        header = extract_header(events)
        header_sizes.append(len(header))
        age = get_age_from_header(events)
        if age is not None:
            ages.append(age)

    print(f"  Header size distribution:")
    from collections import Counter
    for size, count in sorted(Counter(header_sizes).items()):
        print(f"    {size} tokens: {count} subjects")
    print(f"  AGE stats: min={min(ages):.0f}  max={max(ages):.0f}  "
          f"mean={sum(ages)/len(ages):.1f}  n={len(ages)}")

    # Print header for a few subjects
    print(f"\n  Sample headers:")
    for sid in sorted(seqs.keys())[:3]:
        header = extract_header(seqs[sid])
        print(f"    Subject {sid}: {[e.code for e in header]}")
    print("  PASS")


# ---------------------------------------------------------------------------
# 4. Vocabulary building
# ---------------------------------------------------------------------------

def test_build_vocab():
    print("\n" + "="*60)
    print("TEST 4: Vocabulary building")
    print("="*60)

    # Learned mode (top-5000)
    vocab_learned = build_vocab(DATA_DIR, embedding_type="learned", top_k=5000)
    print(f"  Learned vocab:    {vocab_learned.vocab_size} entries "
          f"(top-5000 + UNK, unk_idx={vocab_learned.unk_idx})")

    # Text-based mode (all codes)
    vocab_text = build_vocab(DATA_DIR, embedding_type="text_based")
    print(f"  Text-based vocab: {vocab_text.vocab_size} entries "
          f"(all unique codes + UNK, unk_idx={vocab_text.unk_idx})")

    # Most common codes
    df = load_split(DATA_DIR, "train")
    top10 = df["code"].value_counts().head(30)
    print(f"\n  Top-10 most frequent codes:")
    for code, count in top10.items():
        in_learned = "✓" if vocab_learned.encode(code) != vocab_learned.unk_idx else "✗"
        print(f"    [{in_learned}] {count:7,}  {code!r}")

    # UNK rate on held-out
    df_heldout = load_split(DATA_DIR, "held_out")
    codes_heldout = df_heldout["code"].dropna().tolist()
    unk_count = sum(1 for c in codes_heldout if vocab_learned.encode(c) == vocab_learned.unk_idx)
    unk_pct = 100 * unk_count / len(codes_heldout)
    print(f"\n  UNK rate on held_out (learned vocab): {unk_pct:.2f}%  "
          f"({unk_count:,}/{len(codes_heldout):,} tokens)")
    print("  PASS")


# ---------------------------------------------------------------------------
# 5. Code translation
# ---------------------------------------------------------------------------

def _make_translator() -> MEDSCodeTranslator:
    import os
    def _p(fname):
        if not LOOKUP_DIR:
            return None
        path = os.path.join(LOOKUP_DIR, fname)
        return path if os.path.exists(path) else None

    return MEDSCodeTranslator.from_csv_files(
        labitems_file   = _p("d_labitems.csv.gz"),
        diagnoses_file  = _p("d_icd_diagnoses.csv.gz"),
        procedures_file = _p("d_icd_procedures.csv.gz"),
        items_file      = _p("d_items.csv.gz"),
    )


def test_code_translator():
    print("\n" + "="*60)
    print("TEST 5: Code translation with real lookup files")
    print("="*60)

    translator = _make_translator()
    has_labs  = bool(translator.lab_lookup)
    has_diag  = bool(translator.diag_lookup)
    has_proc  = bool(translator.proc_lookup)
    print(f"  Lookup tables loaded:  labs={has_labs} ({len(translator.lab_lookup)} entries) | "
          f"diagnoses={has_diag} ({len(translator.diag_lookup)} entries) | "
          f"procedures={has_proc} ({len(translator.proc_lookup)} entries)")

    df = load_split(DATA_DIR, "train")

    # --- 5a. One example of every distinct code prefix in the data ---
    type_examples = {}
    for code in df["code"].dropna().unique():
        prefix = code.split("//")[0] if "//" in code else code.split()[0]
        if prefix not in type_examples:
            type_examples[prefix] = code

    print(f"\n  {'CODE (raw)':<70s}  TRANSLATION")
    print(f"  {'-'*70}  {'-'*50}")
    for code in type_examples.values():
        desc = translator.translate(code)
        print(f"  {code!r:<70s}  {desc!r}")

    # --- 5b. Specific ICU vital codes from the log ---
    icu_codes = [
        "LAB//227969//UNK",      # Safety Measures
        "LAB//220045//bpm",      # Heart Rate
        "LAB//220210//insp/min", # Respiratory Rate
        "LAB//220277//%",        # O2 saturation pulseoxymetry
        "LAB//220048//UNK",      # Heart Rhythm
        "LAB//224650//UNK",      # Ectopy Type 1
        "SUBJECT_WEIGHT_AT_INFUSION//KG",
        "LAB//220180//mmHg",     # Non Invasive Blood Pressure diastolic
        "LAB//220179//mmHg",     # Non Invasive Blood Pressure systolic
        "LAB//220181//mmHg",     # Non Invasive Blood Pressure mean
    ]
    print(f"\n  --- ICU vital & chart codes (from log) ---")
    for code in icu_codes:
        desc = translator.translate(code)
        resolved = "✓" if "Lab test" not in desc and desc != code else "✗"
        print(f"  [{resolved}] {code!r:<45s}  →  {desc!r}")

    # --- 5c. Standard lab spot-checks ---
    lab_checks = [
        ("LAB//50912//mg/dL",  "Creatinine"),
        ("LAB//51221//%",      "Hematocrit"),
        ("LAB//50882//mEq/L",  "Bicarbonate"),
        ("LAB//50971//mEq/L",  "Potassium"),
        ("LAB//50983//mEq/L",  "Sodium"),
    ]
    print(f"\n  --- Standard lab codes ---")
    for code, expected_word in lab_checks:
        desc = translator.translate(code)
        ok = expected_word.lower() in desc.lower()
        print(f"  [{'✓' if ok else '✗'}] {code!r:<35s}  →  {desc!r}")
        if has_labs:
            assert ok, f"Expected '{expected_word}' in translation of {code!r}, got {desc!r}"

    # --- 5d. ICD diagnosis spot-checks ---
    diag_checks = [
        ("DIAGNOSIS//ICD//9//25000",  "Diabetes"),
        ("DIAGNOSIS//ICD//10//E119",  "diabetes"),
        ("DIAGNOSIS//ICD//10//I10",   "hypertension"),
    ]
    print(f"\n  --- ICD diagnosis codes ---")
    for code, expected_word in diag_checks:
        desc = translator.translate(code)
        ok = expected_word.lower() in desc.lower()
        print(f"  [{'✓' if ok else '✗'}] {code!r:<45s}  →  {desc!r}")

    # --- 5e. DRG spot-checks ---
    drg_codes = df[df["code"].str.startswith("DRG//", na=False)]["code"].unique()[:6]
    print(f"\n  --- DRG codes ---")
    for code in drg_codes:
        desc = translator.translate(code)
        print(f"  {code!r:<70s}")
        print(f"      →  {desc!r}")

    # --- 5f. Overall coverage ---
    all_codes = df["code"].dropna().unique().tolist()
    fallback_count = sum(
        1 for c in all_codes
        if translator.translate(c) in (c, c.replace("//", " ").replace("_", " "))
    )
    resolved_count = len(all_codes) - fallback_count
    print(f"\n  Translation coverage: {resolved_count}/{len(all_codes)} unique codes resolved "
          f"({100*resolved_count/len(all_codes):.1f}%)")
    print("  PASS")


# ---------------------------------------------------------------------------
# 6. MEDSDataset — pretrain mode
# ---------------------------------------------------------------------------

def test_dataset_pretrain():
    print("\n" + "="*60)
    print("TEST 6: MEDSDataset pretrain mode")
    print("="*60)
    vocab = build_vocab(DATA_DIR, embedding_type="learned", top_k=5000)
    ds = MEDSDataset(
        data_dir=DATA_DIR,
        vocab=vocab,
        split="train",
        task="pretrain",
        max_seq_len=512,
    )
    print(f"  Dataset size: {len(ds)} subjects")

    item = ds[0]
    print(f"\n  Sample item (subject {item['subject_id']}):")
    print(f"    keys:          {list(item.keys())}")
    print(f"    seq length:    {len(item['codes'])}")
    print(f"    label:         {item['label']}")
    print(f"    first 6 codes (raw):    {item['raw_codes'][:6]}")
    print(f"    first 6 codes (encoded): {item['codes'][:6]}")
    print(f"    first 6 values:          {item['values'][:6]}")
    print(f"    first 6 z_scores:        {item['z_scores'][:6]}")
    print(f"    first 6 delta_times:     {[f'{v:.4f}' for v in item['delta_times'][:6]]}")

    # Verify header is at front
    assert item["raw_codes"][0] == "AGE", f"First code should be AGE, got {item['raw_codes'][0]}"
    assert "z_scores" in item and "delta_times" in item, "Missing z_scores/delta_times keys"
    print("  PASS")


# ---------------------------------------------------------------------------
# 7. Collator — pretrain mode with DataLoader
# ---------------------------------------------------------------------------

def test_collator_dataloader():
    print("\n" + "="*60)
    print("TEST 7: Collator + DataLoader (pretrain)")
    print("="*60)
    MAX_LEN = 512
    BATCH = 4

    vocab = build_vocab(DATA_DIR, embedding_type="learned", top_k=5000)
    ds = MEDSDataset(
        data_dir=DATA_DIR,
        vocab=vocab,
        split="train",
        task="pretrain",
        max_seq_len=MAX_LEN,
    )
    collator = MEDSCollator(pad_idx=vocab.unk_idx, max_len=MAX_LEN, task="pretrain")
    loader = DataLoader(ds, batch_size=BATCH, collate_fn=collator, shuffle=False)

    batch = next(iter(loader))
    print(f"  Batch keys:          {list(batch.keys())}")
    print(f"  codes shape:         {batch['codes'].shape}")
    print(f"  attention_mask shape:{batch['attention_mask'].shape}")
    print(f"  values shape:        {batch['values'].shape}")
    print(f"  z_scores shape:      {batch['z_scores'].shape}")
    print(f"  delta_times shape:   {batch['delta_times'].shape}")
    print(f"  labels:              {batch['labels'].tolist()}")
    print(f"  subject_ids:         {batch['subject_ids'].tolist()}")

    assert batch["codes"].shape == (BATCH, MAX_LEN), f"Expected ({BATCH},{MAX_LEN})"
    assert batch["codes"].dtype == torch.long
    assert batch["values"].dtype == torch.float
    assert "z_scores" in batch and "delta_times" in batch

    # Show attention mask distribution (real vs pad tokens)
    for i in range(BATCH):
        n_real = batch["attention_mask"][i].sum().item()
        print(f"    Item {i}: {n_real}/{MAX_LEN} real tokens "
              f"({'windowed' if n_real==MAX_LEN else 'padded'})")
    print("  PASS")


# ---------------------------------------------------------------------------
# 8. EventEmbedding — forward pass on real batch
# ---------------------------------------------------------------------------

def test_event_embedding_forward():
    print("\n" + "="*60)
    print("TEST 8: EventEmbedding forward pass on real data")
    print("="*60)
    D_MODEL = 128
    MAX_LEN = 512
    BATCH = 2

    vocab = build_vocab(DATA_DIR, embedding_type="learned", top_k=5000)
    ds = MEDSDataset(
        data_dir=DATA_DIR,
        vocab=vocab,
        split="train",
        task="pretrain",
        max_seq_len=MAX_LEN,
    )
    collator = MEDSCollator(pad_idx=vocab.unk_idx, max_len=MAX_LEN, task="pretrain")
    loader = DataLoader(ds, batch_size=BATCH, collate_fn=collator, shuffle=False)
    batch = next(iter(loader))

    config = EmbeddingConfig(
        embedding_type="learned",
        vocab_size=vocab.vocab_size,
        d_model=D_MODEL,
    )
    model = EventEmbedding(config)
    model.eval()

    with torch.no_grad():
        embeddings = model(batch["codes"])

    print(f"  Input shape:     {batch['codes'].shape}")
    print(f"  Output shape:    {embeddings.shape}  (expected: [{BATCH}, {MAX_LEN}, {D_MODEL}])")
    print(f"  Output dtype:    {embeddings.dtype}")
    print(f"  Output mean:     {embeddings.mean().item():.4f}")
    print(f"  Output std:      {embeddings.std().item():.4f}")
    print(f"  First token emb (first 8 dims): {embeddings[0, 0, :8].tolist()}")

    assert embeddings.shape == (BATCH, MAX_LEN, D_MODEL)
    print("  PASS")


# ---------------------------------------------------------------------------
# 9. Sequence length statistics
# ---------------------------------------------------------------------------

def test_sequence_length_stats():
    print("\n" + "="*60)
    print("TEST 9: Sequence length statistics vs context window")
    print("="*60)
    CONTEXT = 512
    vocab = build_vocab(DATA_DIR, embedding_type="learned", top_k=5000)
    df = load_split(DATA_DIR, "train")
    seqs = build_subject_sequences(df)

    lengths = [len(v) for v in seqs.values()]
    n_long = sum(1 for l in lengths if l > CONTEXT)
    n_short = sum(1 for l in lengths if l <= CONTEXT)

    print(f"  Context window: {CONTEXT}")
    print(f"  Total subjects: {len(lengths)}")
    print(f"  Longer than context: {n_long} ({100*n_long/len(lengths):.1f}%) → stochastic windowing")
    print(f"  Fits in context:     {n_short} ({100*n_short/len(lengths):.1f}%) → padding")
    print(f"  Min length:  {min(lengths)}")
    print(f"  Max length:  {max(lengths)}")
    print(f"  Mean length: {sum(lengths)/len(lengths):.1f}")

    # Percentiles
    import statistics
    lengths_sorted = sorted(lengths)
    p50 = lengths_sorted[len(lengths_sorted)//2]
    p90 = lengths_sorted[int(len(lengths_sorted)*0.9)]
    p99 = lengths_sorted[int(len(lengths_sorted)*0.99)]
    print(f"  p50={p50}  p90={p90}  p99={p99}")
    print("  PASS")


# ---------------------------------------------------------------------------
# 10. Sequence length histogram
# ---------------------------------------------------------------------------

def test_sequence_length_histogram():
    print("\n" + "="*60)
    print("TEST 10: Sequence length histogram")
    print("="*60)

    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend (safe for servers)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    df = load_split(DATA_DIR, "train")
    seqs = build_subject_sequences(df)
    lengths = sorted(len(v) for v in seqs.values())

    max_len = max(lengths)
    min_len = min(lengths)
    mean_len = sum(lengths) / len(lengths)
    median_len = lengths[len(lengths) // 2]

    print(f"  Subjects:     {len(lengths)}")
    print(f"  Min length:   {min_len}")
    print(f"  Max length:   {max_len}")
    print(f"  Mean length:  {mean_len:.1f}")
    print(f"  Median:       {median_len}")

    # ---- Full range: log x (sequence length), log y (count tail) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Patient sequence length distribution", fontsize=13, fontweight="bold")

    # Left panel: full range
    ax = axes[0]
    ax.hist(lengths, bins=range(min_len, max_len + 2), color="#4C72B0", edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length (events per patient, log scale)", fontsize=11)
    ax.set_ylabel("Number of patients (log scale)", fontsize=11)
    ax.set_title(f"Full range  [1 – {max_len:,}]", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Annotate percentile lines
    for pct, label, col in [(50, "p50", "#e67e22"), (90, "p90", "#e74c3c"), (95, "p95", "#8e44ad")]:
        val = lengths[int(len(lengths) * pct / 100)]
        ax.axvline(val, color=col, linestyle="--", linewidth=1.2, label=f"{label}={val:,}")
    ax.axvline(512, color="black", linestyle=":", linewidth=1.5, label="context=512")
    ax.legend(fontsize=9)

    # Right panel: zoomed to ≤ 1500, log x, linear y
    zoom_max = 1500
    zoomed = [l for l in lengths if l <= zoom_max]
    ax2 = axes[1]
    ax2.hist(zoomed, bins=range(min_len, zoom_max + 2), color="#4C72B0", edgecolor="none")
    ax2.set_xscale("log")
    ax2.set_xlabel("Sequence length (events per patient, log scale)", fontsize=11)
    ax2.set_ylabel("Number of patients", fontsize=11)
    ax2.set_title(
        f"Zoomed: ≤ {zoom_max:,} events  ({len(zoomed)}/{len(lengths)} patients, "
        f"{100*len(zoomed)/len(lengths):.1f}%)",
        fontsize=11,
    )
    ax2.axvline(512, color="black", linestyle=":", linewidth=1.5, label="context=512")
    ax2.axvline(median_len, color="#e67e22", linestyle="--", linewidth=1.2,
                label=f"median={median_len}")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "sequence_length_histogram.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Histogram saved to: {out_path}")

    # ASCII spark-histogram for quick terminal preview (first 100 sequence lengths, bucket=1)
    print(f"\n  ASCII preview (bucket = 1 event; first 100):")
    max_bucket = 100
    buckets = {}
    for l in lengths:
        if 0 <= l < max_bucket:
            buckets[l] = buckets.get(l, 0) + 1
    if buckets:
        bar_max = max(buckets.values())
    else:
        bar_max = 1
    bar_width = 40
    for k in range(max_bucket):
        count = buckets.get(k, 0)
        bar = "█" * int(count / bar_max * bar_width) if bar_max > 0 else ""
        print(f"    {k:>3} │{bar:<{bar_width}}│ {count}")

    print("  PASS")


def _events_per_trajectory_window_counts(events, window_hours: float) -> List[int]:
    """
    For one subject, anchor at **trajectory start** ``t0`` = earliest non-null event time.
    Split wall-clock time into consecutive half-open windows of length ``W`` hours::

        [t0, t0 + W), [t0 + W, t0 + 2W), ...

    Return one integer per window index (0, 1, …, K): number of events whose
    time falls in that window. **Not** inter-event gaps: each event is placed by
    absolute time since ``t0``. Windows with no events are **0** between the
    first and last index that contains any event, so indices are contiguous
    along the stay (e.g. for W=24: “events on first day”, “second day”, …).
    """
    w = pd.Timedelta(hours=float(window_hours))
    valid = [e for e in events if pd.notna(e.time)]
    if not valid:
        return []
    t0 = min(e.time for e in valid)
    by_block = {}
    for e in valid:
        k = int((e.time - t0) / w)
        by_block[k] = by_block.get(k, 0) + 1
    max_k = max(by_block)
    return [by_block.get(i, 0) for i in range(max_k + 1)]


# ---------------------------------------------------------------------------
# 11. Events per fixed-duration time block (1–24 h), 3×2 histograms
# ---------------------------------------------------------------------------

def test_time_block_event_histograms():
    print("\n" + "="*60)
    print("TEST 11: Events per trajectory window (1–24 h blocks from earliest time)")
    print("="*60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    window_hours = (1, 2, 4, 8, 12, 24)
    df = load_split(DATA_DIR, "train")
    seqs = build_subject_sequences(df)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes_flat = axes.flatten()
    fig.suptitle(
        "Each subject: from earliest event time t₀, count events in [t₀, t₀+W), then "
        "[t₀+W, t₀+2W), … (not time-between-events).\n"
        "Histogram: across all subjects, how many such windows had each event count.",
        fontsize=11,
        fontweight="bold",
    )

    for ax, wh in zip(axes_flat, window_hours):
        all_counts: List[int] = []
        for events in seqs.values():
            all_counts.extend(_events_per_trajectory_window_counts(events, wh))

        if not all_counts:
            ax.set_title(f"{wh:g} h — no data")
            continue

        hi = max(all_counts)
        # Integer-centered bins when modest span; else fixed number of equal-width bins
        if hi <= 300:
            bins = np.arange(-0.5, hi + 1.5, 1.0)
        else:
            bins = 150
        ax.hist(
            all_counts,
            bins=bins,
            range=None if hi <= 300 else (0, hi + 1),
            color="#2ca02c",
            edgecolor="none",
        )
        ax.set_xlabel(f"Events in one {wh:g}-h window [t₀+k·{wh:g}h, t₀+(k+1)·{wh:g}h)")
        ax.set_ylabel("Number of blocks (all subjects, log scale)")
        mean_c = float(np.mean(all_counts))
        n_zero = sum(1 for c in all_counts if c == 0)
        ax.set_title(
            f"{wh:g}-h blocks  n_blocks={len(all_counts):,}  mean={mean_c:.2f}  "
            f"empty={100 * n_zero / len(all_counts):.1f}%"
        )
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "events_per_time_block_histograms.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved to: {out_path}")
    print(f"  Window sizes (h): {list(window_hours)}")
    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nUsing data directory:   {DATA_DIR}")
    print(f"Using lookup directory: {LOOKUP_DIR}")

    tests = [
        test_load_real_parquets,
        test_build_sequences,
        test_header_extraction,
        test_build_vocab,
        test_code_translator,
        test_dataset_pretrain,
        test_collator_dataloader,
        test_event_embedding_forward,
        test_sequence_length_stats,
        test_sequence_length_histogram,
        test_time_block_event_histograms,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n  FAILED: {test_fn.__name__}")
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60)
    if failed:
        sys.exit(1)
