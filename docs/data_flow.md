# EHR-JEPA: Complete Data Flow

This document traces every step of a training run, from `main.py` entry point
to the final loss scalar, showing the input and output of every function and
the tensor shapes at each stage.

---

## 1. Entry point — `main.py`

```
python main.py --config configs/ehr_config.yaml
```

```
main(config_path, no_wandb)
│
├── load_config(config_path)
│     IN:  path to YAML file
│     OUT: nested Python dict (cfg)
│
├── set_seed(cfg["seed"])
│     IN:  int | None
│     OUT: seeds Python random, NumPy, torch, torch.cuda
│
├── _ensure_vocab(cfg)           ← only when embedding_type == "learned"
│     IN:  cfg dict
│     OUT: Vocab object  (vocab.json loaded or built from training parquet files)
│
├── _ensure_normalizer(cfg)      ← only when use_value == True
│     IN:  cfg dict
│     OUT: ValueNormalizer object  (normalizer_stats.json loaded or fitted)
│
├── init_wandb(cfg, config_path)
│     IN:  cfg dict, path string
│     OUT: wandb.Run | None
│
├── build_model(cfg, vocab)
│     IN:  cfg dict, Vocab
│     OUT: JEPATrainer (nn.Module, all sub-models wired together)
│
├── build_loaders(cfg, vocab, normalizer)
│     IN:  cfg dict, Vocab, ValueNormalizer | None
│     OUT: (train_loader, val_loader)   DataLoader objects
│
└── trainer.train_loop(train_loader, val_loader, optimizer, callbacks)
      IN:  DataLoaders, AdamW optimizer, W&B callback functions
      OUT: history dict  {"train_loss": [...], "val_loss": [...], "lr": [...]}
```

---

## 2. Data loading — `build_loaders`

Called once at startup. Reads every parquet file in the split directory.

```
build_loaders(cfg, vocab, normalizer)
│
└── MEDSDataset.__init__(data_dir, vocab, split, task, max_seq_len,
│                         aces_label_path, normalizer, time_unit)
│     │
│     ├── load_split(data_dir, split)
│     │     IN:  "/path/to/data/", "train" | "tuning" | "held_out"
│     │     reads: all .parquet files in data_dir/split/
│     │     OUT:  pd.DataFrame
│     │             columns: subject_id (int64), time (datetime64),
│     │                      code (str), numeric_value (float64 | NaN)
│     │
│     └── build_subject_sequences(df)
│           IN:  flat DataFrame (all subjects concatenated)
│           - groups rows by subject_id
│           - sorts each group by time, NaT first (so header rows lead)
│           OUT: Dict[subject_id → List[Event]]
│                 Event = dataclass(time, code, numeric_value)
│
└── MEDSCollator(pad_idx, max_len, task)
      Stateless transform applied per mini-batch by DataLoader
```

---

## 3. Per-sample fetch — `MEDSDataset.__getitem__(idx)`

Called by DataLoader for each sample in a mini-batch.

```
__getitem__(idx)
│
│   sample = {"subject_id": int, "prediction_time": Timestamp|None, "label": int}
│   events = List[Event]  (full chronological sequence for the subject)
│
├── [prediction mode only]
│   ├── _apply_time_cutoff(events, prediction_time)
│   │     IN:  List[Event], pd.Timestamp
│   │     keeps: only events where event.time <= prediction_time
│   │     OUT: List[Event]   (shorter list, no future leakage)
│   │
│   └── _truncate_with_header(events)
│         IN:  List[Event]
│         IF len(events) > max_seq_len:
│           1. extract_header(events)    → first ≤4 demographic tokens
│           2. compute years_elapsed since header to start of tail window
│           3. update AGE numeric_value += round(years_elapsed)
│           4. return updated_header + events[-(max_seq_len - len(header)):]
│         OUT: List[Event]  length ≤ max_seq_len
│
├── _encode_events(events)
│   │
│   ├── vocab.encode(code) for each event
│   │     IN:  code string  e.g. "LAB//50882//mEq/L"
│   │     OUT: int (vocabulary index; unk_idx if code unseen)
│   │
│   ├── _compute_delta_times(events)
│   │     IN:  List[Event]
│   │     for each event i:
│   │       delta_i = log(1 + (time_i - time_{i-1}) / 3600)   [hours]
│   │       header events (NaT) → 0.0
│   │       first real event   → 0.0
│   │     OUT: List[float]  length = len(events)
│   │
│   └── _compute_z_scores(events)
│         IN:  List[Event]
│         calls: normalizer.transform_sequence(codes, values)
│           - winsorizes value to [p5, p95] per code
│           - z-scores: (value - mean) / std   (0.0 if std==0 or value missing)
│         OUT: List[float]  length = len(events)
│
└── returns dict:
      {
        "subject_id":  int,
        "label":       int,
        "codes":       List[int],          # vocab indices
        "raw_codes":   List[str],
        "times":       List[Timestamp],
        "values":      List[float|None],
        "z_scores":    List[float],
        "delta_times": List[float],
      }
```

---

## 4. Batch collation — `MEDSCollator.__call__(batch)`

Called by DataLoader automatically. Receives a list of dicts from `__getitem__`.

```
MEDSCollator.__call__(List[item_dict])
│
└── for each item in batch:
│     _window_or_pad(codes, values, z_scores, delta_times)
│       IN:  variable-length Python lists
│       pretrain, len > max_len:  random start i ~ U(0, len-max_len)
│                                 slice [i : i+max_len], attention_mask all 1s
│       any mode, len >= max_len: truncate to max_len
│       len < max_len:            right-pad codes with pad_idx,
│                                 pad values/z_scores/delta_times with 0.0
│                                 attention_mask: [1]*len + [0]*pad_len
│       OUT: all sequences length = max_len
│
└── stack into batch tensors:
      {
        "codes":          LongTensor  [B, max_len]
        "attention_mask": LongTensor  [B, max_len]   1=real, 0=pad
        "values":         FloatTensor [B, max_len]
        "value_mask":     LongTensor  [B, max_len]   1=value present
        "z_scores":       FloatTensor [B, max_len]
        "delta_times":    FloatTensor [B, max_len]
        "labels":         LongTensor  [B]
        "subject_ids":    LongTensor  [B]
      }
```

---

## 5. Training loop — `JEPATrainer.train_loop`

Outer loop. Calls `forward()` once per mini-batch.

```
train_loop(train_loader, val_loader, optimizer, on_epoch_end, on_batch_end)
│
│  For each epoch:
│    self.train()
│    For each batch from train_loader:
│      │
│      ├── optimizer.zero_grad()
│      │
│      ├── JEPATrainer.forward(codes, attention_mask, values, z_scores,
│      │                       delta_times, value_mask)
│      │     IN:  batch tensors (all [B, L])
│      │     OUT: (L_pred, L_cov, L_total)  scalar tensors
│      │
│      ├── L_total.backward()
│      │     gradients flow through context encoder + predictor via L_pred
│      │     gradients flow through target encoder via L_cov
│      │
│      ├── clip_grad_norm_(parameters, grad_clip)
│      ├── optimizer.step()
│      └── scheduler.step()          cosine_warmup by default
│
│    [end of epoch]
│    _eval_epoch(val_loader)          @torch.no_grad()
│      iterates val batches, averages L_total
│
│    early stopping check
│    on_epoch_end callback → W&B log
│
└── returns history dict
```

---

## 6. Model forward pass — `JEPATrainer.forward`

The core of one training step. `B` = batch size, `L` = sequence length, `d` = d_model.

```
JEPATrainer.forward(codes[B,L], attention_mask[B,L], values, z_scores,
                    delta_times, value_mask)
│
│ ─────────────────────────────────────────────────
│  STEP 1: EMBEDDINGS
│ ─────────────────────────────────────────────────
│
├── EventEmbedding.forward(codes, values, z_scores, delta_times, value_mask)
│   │
│   ├── nn.Embedding(codes)              → [B, L, embed_dim]
│   │     "learned":    embed_dim = d_model   (trainable)
│   │     "text_based": embed_dim = 768       (frozen ClinicalBERT weights)
│   │
│   ├── [text_based only] Linear(768 → d_model)  → [B, L, d_model]
│   │
│   └── [if use_value or use_time]
│         MLP: Linear(d_model + N_extra → d_model) → GELU → Linear(d → d)
│           N_extra:  code_only=0, +value=3, +time=1, +value+time=4
│           inputs concatenated: [E_code, value, z_score, (delta_time), value_mask]
│         residual: x = LayerNorm(E_code + MLP(concat))
│   │
│   OUT: x  [B, L, d_model]
│
│ ─────────────────────────────────────────────────
│  STEP 2: SPAN MASKING  (per sample, Python-side)
│ ─────────────────────────────────────────────────
│
├── for b in range(B):
│     SpanMasker.__call__(seq_len=L, attention_mask[b], times=None)
│       IN:  sequence length, per-sample 1-D attention mask
│       1. real_positions = indices where attention_mask == 1
│       2. B_mask = floor(N_real * mask_ratio)     default: 30%
│       3. num_spans:
│            if B_mask >= default_num_spans * min_span_length → 4 spans
│            else → max(1, B_mask // min_span_length)
│       4. T_span = floor(B_mask / num_spans),
│            last span gets remainder tokens
│       5. rejection-sample non-overlapping contiguous spans
│       OUT: SpanMaskResult
│              context_indices: List[int]     positions NOT in any span
│              target_spans:    List[List[int]]  one list of positions per span
│              span_times:      List[(mid, dur)]  position proxies (no real times)
│
│   Trim all samples to min(num_spans) for uniform batch processing
│
│ ─────────────────────────────────────────────────
│  STEP 3: TARGET PATHWAY
│ ─────────────────────────────────────────────────
│
├── EHRTransformerEncoder.forward(x[B,L,d], attention_mask[B,L])
│   │    ← shared encoder, full sequence, WITH GRADIENTS
│   │    (gradients only arrive here via L_cov — not L_pred)
│   │
│   ├── for each of 6 RoPETransformerLayer:
│   │     ├── LayerNorm(x)
│   │     ├── RoPEMultiheadAttention(x, key_padding_mask)
│   │     │     Q, K, V = Linear projections of x
│   │     │     Q, K rotated by RotaryEmbedding(positions 0..L-1)
│   │     │     scaled dot-product attention
│   │     │     out = Linear(concat heads)
│   │     ├── residual add + dropout
│   │     ├── LayerNorm(x)
│   │     ├── FFN: Linear(d→1024) → GELU → Linear(1024→d)
│   │     └── residual add + dropout
│   │
│   └── final LayerNorm
│   OUT: target_enc_out  [B, L, d_model]
│
├── _extract_target_spans(target_enc_out, all_target_spans)
│     IN:  [B, L, d],  List[List[List[int]]]  (B × num_spans × span_len)
│     for each span s:
│       gather rows at span indices from target_enc_out
│       pad to max_span_len_s across batch
│     OUT: List of num_spans tensors, each  [B, N_span_s, d]
│
│ ─────────────────────────────────────────────────
│  STEP 4: CONTEXT PATHWAY
│ ─────────────────────────────────────────────────
│
├── _extract_context(x, all_context_indices)
│     IN:  x[B,L,d], List[List[int]]  (context positions per sample)
│     gather only context tokens (target spans physically dropped)
│     preserve original integer positions as pos_ids for RoPE
│     pad to max_ctx_len across batch
│     OUT: x_ctx[B, N_ctx, d],  ctx_pos_ids[B, N_ctx],  ctx_mask[B, N_ctx]
│
├── EHRTransformerEncoder.forward(x_ctx, attention_mask=ctx_mask,
│                                  position_ids=ctx_pos_ids)
│   │    ← SAME shared encoder, compact context sequence, WITH GRADIENTS
│   │    RoPE uses ORIGINAL position IDs (not 0..N_ctx-1)
│   │    so positional relationships between context tokens are preserved
│   OUT: context_enc_out  [B, N_ctx, d_model]
│
│ ─────────────────────────────────────────────────
│  STEP 5A: BRANCH A — PERCEIVER  (use_perceiver=True)
│ ─────────────────────────────────────────────────
│
├── _forward_perceiver(context_enc_out, ctx_mask, target_spans_list, span_times)
│   │
│   ├── LatentCrossAttentionPool.forward(context_enc_out, key_padding_mask)
│   │     IN:  [B, N_ctx, d]
│   │     16 learnable latent tokens (queries) cross-attend over context
│   │     MultiheadAttention(query=latents, key=context, value=context)
│   │     + LayerNorm
│   │     OUT: Z_ctx  [B, 16, d]    ← fixed-size context summary
│   │
│   ├── for each span s  (skip if N_span_s < min_span_for_perceiver=15):
│   │   │
│   │   ├── LatentCrossAttentionPool.forward(target_spans_list[s])
│   │   │     IN:  [B, N_span_s, d]
│   │   │     16 learnable latent tokens cross-attend over target span tokens
│   │   │     OUT: Z_tgt  [B, 16, d]    ← fixed-size target summary
│   │   │
│   │   ├── TemporalSpanPrompt.forward(coords[B, 1, 2])
│   │   │     IN:  (midpoint_position, span_length) for this span
│   │   │     MLP: Linear(2→d) → GELU → Linear(d→d)
│   │   │     OUT: prompt  [B, 1, d]
│   │   │
│   │   ├── Z_prompted = LayerNorm(Z_ctx + prompt)   broadcast over 16 latents
│   │   │     OUT: [B, 16, d]    ← context conditioned on WHERE the span is
│   │   │
│   │   ├── Predictor.transformer.forward(Z_prompted)
│   │   │     2-layer RoPE Transformer (shallow, forces reliance on prompt)
│   │   │     IN:  [B, 16, d]
│   │   │     OUT: Z_hat  [B, 16, d]    ← predicted target representation
│   │   │
│   │   ├── jepa_prediction_loss(Z_hat, Z_tgt)
│   │   │     = MSE(Z_hat, Z_tgt.detach())
│   │   │     stop_grad on Z_tgt → gradient does NOT flow to target encoder here
│   │   │     OUT: scalar L_pred_s
│   │   │
│   │   └── CovarianceRegularizationLoss.forward(Z_tgt)
│   │         Z_tgt NOT detached → gradient DOES flow to target encoder
│   │         flatten [B, 16, d] → [B*16, d]
│   │         project via Linear(d→64)
│   │         center, compute covariance matrix [64, 64]
│   │         OUT: scalar L_cov_s = ||C - I||_F
│   │
│   └── L_pred = mean(L_pred_s),  L_cov = mean(L_cov_s)  over valid spans
│
│ ─────────────────────────────────────────────────
│  STEP 5B: BRANCH B — TOKEN I-JEPA  (use_perceiver=False)
│ ─────────────────────────────────────────────────
│
└── _forward_token(context_enc_out, ctx_pos_ids, ctx_mask,
                   target_spans_list, all_target_spans, span_times)
    │
    └── for each span s:
        │
        ├── TemporalSpanPrompt.forward(coords[B, 1, 2])
        │     IN:  (midpoint_position, span_length) for this span
        │     OUT: span_prompt  [B, 1, d]
        │
        ├── Build MASK tokens:
        │     mask_tokens = mask_token_param[d].expand(B, N_span, d) + span_prompt
        │     OUT: [B, N_span, d]   ← learnable token + temporal conditioning
        │
        ├── Concatenate:
        │     x_in = cat([context_enc_out, mask_tokens], dim=1)
        │     OUT: [B, N_ctx + N_span, d]
        │
        ├── Build position IDs:
        │     pos_ids = cat([ctx_pos_ids, span_original_positions], dim=1)
        │     OUT: [B, N_ctx + N_span]    ← preserves original temporal order
        │
        ├── EHRTransformerEncoder.forward(x_in, attn_mask, position_ids)
        │     ← token_predictor (separate shallow 2-layer encoder)
        │     self-attention allows mask tokens to query the full context
        │     OUT: [B, N_ctx + N_span, d]
        │
        ├── Slice mask-token outputs:
        │     y_hat = out[:, N_ctx:, :]
        │     OUT: Y_hat  [B, N_span, d]    ← predicted token representations
        │
        ├── jepa_prediction_loss(Y_hat, Y_tgt)
        │     Y_tgt = target_spans_list[s]  [B, N_span, d]  from target encoder
        │     = MSE(Y_hat, Y_tgt.detach())
        │     OUT: scalar L_pred_s
        │
        └── CovarianceRegularizationLoss.forward(Y_tgt)
              flatten [B, N_span, d] → [B*N_span, d]
              project, center, covariance
              OUT: scalar L_cov_s = ||C - I||_F

    L_pred = mean(L_pred_s),  L_cov = mean(L_cov_s)  over spans

─────────────────────────────────────────────────
 STEP 6: TOTAL LOSS
─────────────────────────────────────────────────

L_total = L_pred + λ * L_cov           λ = 0.1 by default

Gradient flow summary:
  L_pred  →  context encoder, predictor / token_predictor, embedding
  L_cov   →  target encoder (ONLY path gradients reach it), embedding
  Both    →  EventEmbedding MLP, projection layer

return (L_pred, L_cov, L_total)   all scalar tensors
```

---

## 7. File map

| File | Purpose |
|------|---------|
| `main.py` | Entry point, wires everything together |
| `configs/ehr_config.yaml` | All hyperparameters and paths |
| `data/meds_parser.py` | Parquet loading, Event dataclass, sequence building |
| `data/vocab.py` | Code → integer index mapping |
| `data/normalizer.py` | Winsorise + z-score per code (fit on train only) |
| `data/meds_dataset.py` | PyTorch Dataset: per-sample fetch, time cutoff, encoding |
| `data/collator.py` | PyTorch Collator: windowing, padding, batch tensor creation |
| `models/event_embedding.py` | Code embedding + optional value/time MLP |
| `models/transformer_encoder.py` | RoPE Transformer encoder (shared for target + context) |
| `models/latent_pooling.py` | Perceiver cross-attention pool (Branch A) |
| `models/predictor.py` | TemporalSpanPrompt MLP + latent Predictor (Branch A) |
| `masking/span_masking.py` | Span masking: which positions become context / target |
| `loss/jepa_loss.py` | MSE with stop-gradient on target |
| `loss/covariance_reg.py` | VICReg-style covariance regularisation |
| `training/trainer.py` | JEPATrainer: full forward pass + training loop |
| `scripts/encode_text_embeddings.py` | Offline ClinicalBERT encoding (run once before training) |
| `submissions/pretrain_gpu.sh` | SLURM GPU job script |
