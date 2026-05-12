"""
Run every test suite with full printed output — both the original integration
tests and all the new module tests.

Usage (from project root):
    python tests/run_all_tests.py              # all suites
    python tests/run_all_tests.py normalizer   # only the normalizer suite

Each suite is a label + list of test functions taken directly from the
individual test modules.  All print() calls inside each test are visible
exactly as they would be if you ran that file standalone.
"""

from __future__ import annotations

import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Import all test modules
# ---------------------------------------------------------------------------

import tests.test_normalizer            as _tn
import tests.test_event_embedding_mlp   as _te
import tests.test_transformer_encoder   as _tt
import tests.test_span_masking          as _ts
import tests.test_latent_pooling        as _tl
import tests.test_predictor             as _tp
import tests.test_losses                as _tloss
import tests.test_trainer_forward       as _tf

# Legacy integration test (runs against real data)
import tests.test_integration_real_data as _tint

# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

SUITES = {
    "normalizer": (
        "ValueNormalizer — winsorize, z-score, save/load",
        [
            _tn.test_fit_stores_stats,
            _tn.test_winsorize,
            _tn.test_zscore,
            _tn.test_zscore_std_zero,
            _tn.test_missing_value,
            _tn.test_unseen_code,
            _tn.test_save_load_roundtrip,
            _tn.test_transform_sequence,
        ],
    ),
    "event_embedding": (
        "EventEmbedding — all four MLP modes",
        [
            _te.test_code_only,
            _te.test_code_only_no_mlp,
            _te.test_code_plus_value,
            _te.test_code_plus_value_residual,
            _te.test_code_plus_time,
            _te.test_code_plus_value_plus_time,
            _te.test_extra_tensors_ignored,
            _te.test_omitted_optional_tensors,
            _te.test_text_based_frozen_weights,
        ],
    ),
    "transformer_encoder": (
        "EHRTransformerEncoder — RoPE, padding mask, position IDs",
        [
            _tt.test_forward_shape,
            _tt.test_padding_mask,
            _tt.test_rope_default_positions,
            _tt.test_rope_custom_positions,
            _tt.test_full_sequence_vs_extracted_context,
            _tt.test_no_nan_in_output,
        ],
    ),
    "span_masking": (
        "SpanMasker — dynamic spans, T_span floor, span_times",
        [
            _ts.test_output_structure,
            _ts.test_no_overlap,
            _ts.test_mask_ratio,
            _ts.test_context_plus_target_equals_N,
            _ts.test_dynamic_num_spans_short,
            _ts.test_dynamic_num_spans_normal,
            _ts.test_padding_excluded,
            _ts.test_span_times,
            _ts.test_t_span_floor,
            _ts.test_empty_sequence,
        ],
    ),
    "latent_pooling": (
        "LatentCrossAttentionPool — Perceiver-style cross-attention",
        [
            _tl.test_output_shape,
            _tl.test_key_padding_mask,
            _tl.test_bool_key_padding_mask,
            _tl.test_latents_are_learned,
            _tl.test_batched_target_spans,
            _tl.test_output_no_nan,
            _tl.test_gradient_through_latents,
        ],
    ),
    "predictor": (
        "TemporalSpanPrompt + Predictor",
        [
            _tp.test_temporal_prompt_shape,
            _tp.test_temporal_prompt_different_coords,
            _tp.test_predictor_forward_shape,
            _tp.test_prompt_conditioning_effect,
            _tp.test_flatten_reshape_roundtrip,
            _tp.test_predictor_no_nan,
            _tp.test_predictor_gradient_flows,
        ],
    ),
    "losses": (
        "JEPA prediction loss + SIGReg",
        [
            _tloss.test_jepa_loss_zero,
            _tloss.test_jepa_loss_positive,
            _tloss.test_jepa_loss_no_grad_through_target,
            _tloss.test_jepa_loss_scalar,
            _tloss.test_sigreg_loss_shape,
            _tloss.test_sigreg_deterministic_given_global_step,
            _tloss.test_sigreg_gradients_flow,
            _tloss.test_sigreg_no_trainable_submodules,
            _tloss.test_covariance_alias_import,
        ],
    ),
    "trainer_forward": (
        "JEPATrainer — full forward pass, all modes, gradient flow",
        [
            _tf.test_branch_a_code_only,
            _tf.test_branch_a_code_plus_value,
            _tf.test_branch_a_code_plus_time,
            _tf.test_branch_a_code_plus_value_plus_time,
            _tf.test_branch_a_gradient_flow,
            _tf.test_branch_a_span_filter,
            _tf.test_branch_b_code_only,
            _tf.test_branch_b_code_plus_value,
            _tf.test_branch_b_code_plus_value_plus_time,
            _tf.test_branch_b_gradient_flow,
            _tf.test_losses_positive_branch_a,
            _tf.test_losses_positive_branch_b,
            _tf.test_short_sequence,
            _tf.test_shared_encoder_weights,
        ],
    ),
    "integration": (
        "Integration — real MIMIC data end-to-end pipeline",
        [
            _tint.test_load_real_parquets,
            _tint.test_build_sequences,
            _tint.test_header_extraction,
            _tint.test_build_vocab,
            _tint.test_code_translator,
            _tint.test_dataset_pretrain,
            _tint.test_collator_dataloader,
            _tint.test_event_embedding_forward,
            _tint.test_sequence_length_stats,
            _tint.test_sequence_length_histogram,
        ],
    ),
}

# Run suites in this order when no filter is given
DEFAULT_ORDER = [
    "normalizer",
    "event_embedding",
    "transformer_encoder",
    "span_masking",
    "latent_pooling",
    "predictor",
    "losses",
    "trainer_forward",
    "integration",
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_suite(key: str) -> tuple[int, int]:
    label, tests = SUITES[key]
    width = 62
    print("\n" + "=" * width)
    print(f"  SUITE: {key.upper()}")
    print(f"  {label}")
    print("=" * width)

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception:
            print(f"\n  !! FAILED: {fn.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n  Suite result: {passed}/{passed+failed} passed")
    return passed, failed


def main(filter_keys: list[str] | None = None) -> None:
    keys = filter_keys if filter_keys else DEFAULT_ORDER

    # Validate
    unknown = [k for k in keys if k not in SUITES]
    if unknown:
        print(f"Unknown suite(s): {unknown}")
        print(f"Available: {list(SUITES.keys())}")
        sys.exit(1)

    total_passed = total_failed = 0
    for key in keys:
        p, f = run_suite(key)
        total_passed += p
        total_failed += f

    width = 62
    print("\n" + "=" * width)
    print(f"  GRAND TOTAL: {total_passed} passed, {total_failed} failed "
          f"out of {total_passed + total_failed} tests")
    print("=" * width)
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    # Optional positional args filter to specific suites, e.g.:
    #   python tests/run_all_tests.py normalizer losses
    filter_keys = sys.argv[1:] if len(sys.argv) > 1 else None
    main(filter_keys)
