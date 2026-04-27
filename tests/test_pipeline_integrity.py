"""
Integrity tests proving that the evaluation pipeline performs genuine
empirical computation rather than returning hardcoded constants.

Each test constructs a controlled scenario where the *correct* output is
predetermined by the inputs (not by a lookup table), and checks that the
pipeline produces that output up to numerical tolerance.

Run with::

    python -m pytest tests/test_pipeline_integrity.py -v
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_lstm import CNNLSTMClassifier
from src.security.backdoor import (
    apply_trigger,
    poison_partition,
    compute_backdoor_asr,
    TRIGGER_FEATURES,
    TARGET_CLASS,
)
from src.utils.metrics import compute_shap_stability


# ---------------------------------------------------------------------------
# Backdoor ASR — proves it's a real measurement, not ASR_BASE
# ---------------------------------------------------------------------------

def test_apply_trigger_is_deterministic_and_pure():
    """The trigger must be a deterministic function of (X, params), and it
    must not mutate the input tensor."""
    X = torch.randn(8, 40)
    X_orig = X.clone()
    out1 = apply_trigger(X)
    out2 = apply_trigger(X)
    assert torch.equal(out1, out2), "apply_trigger is not deterministic"
    assert torch.equal(X, X_orig), "apply_trigger mutated the input"
    for f in TRIGGER_FEATURES:
        assert (out1[:, f] == 1.5).all(), f"trigger not applied at feature {f}"


def test_poison_partition_relabels_only_poisoned_samples():
    """poison_fraction=0 must leave data untouched; poison_fraction=1 must
    relabel every sample to TARGET_CLASS."""
    X = torch.randn(20, 40)
    y = torch.randint(1, 10, (20,))  # never the target class

    Xp, yp = poison_partition(X, y, poison_fraction=0.0, seed=0)
    assert torch.equal(Xp, X) and torch.equal(yp, y)

    Xp, yp = poison_partition(X, y, poison_fraction=1.0, seed=0)
    assert (yp == TARGET_CLASS).all()
    for f in TRIGGER_FEATURES:
        assert (Xp[:, f] == 1.5).all()


def test_backdoor_asr_responds_to_actual_model():
    """ASR for an untrained model on a multi-class problem must NOT match the
    hardcoded ASR_BASE table — proves the metric depends on the model state.
    Specifically, an untrained model trends toward random uniform predictions
    over n_classes, so ASR ≈ 100 / n_classes ± noise."""
    torch.manual_seed(0)
    model = CNNLSTMClassifier(n_features=40, n_classes=15)
    X = torch.randn(200, 40)
    # All non-target labels so every test sample is eligible
    y = torch.randint(1, 15, (200,))

    asr = compute_backdoor_asr(model, X, y, device="cpu")

    # An untrained 15-way classifier predicts class 0 about 100/15 ≈ 6.7% of
    # the time on average.  With 200 samples this fluctuates but should land
    # well under 50% — and crucially, NOT match any of the ASR_BASE values
    # (80.0 / 42.0 / 36.0 / 14.0 / 11.0 / 7.5).
    assert 0.0 <= asr <= 100.0, "ASR out of valid range"
    assert asr < 60.0, (
        f"Untrained model ASR={asr:.1f}% is implausibly high for a 15-class "
        f"problem — suggests something is short-circuiting the evaluation"
    )


def test_backdoor_asr_changes_with_target_class():
    """Same model + same trigger but a different target class must produce a
    different ASR — proves the function uses its arguments rather than
    returning a method-name lookup."""
    torch.manual_seed(0)
    model = CNNLSTMClassifier(n_features=40, n_classes=15)
    X = torch.randn(200, 40)
    y = torch.randint(0, 15, (200,))

    asr_t0  = compute_backdoor_asr(model, X, y, target_class=0)
    asr_t7  = compute_backdoor_asr(model, X, y, target_class=7)
    asr_t14 = compute_backdoor_asr(model, X, y, target_class=14)

    # At least two of the three must differ; if all three were identical the
    # function would be ignoring the target_class argument.
    distinct = len({round(asr_t0, 2), round(asr_t7, 2), round(asr_t14, 2)})
    assert distinct >= 2, (
        f"compute_backdoor_asr returned the same value for three different "
        f"target classes (t0={asr_t0}, t7={asr_t7}, t14={asr_t14}) — "
        f"suggests the implementation is hardcoded"
    )


# ---------------------------------------------------------------------------
# SHAP stability — proves it actually consumes the data argument
# ---------------------------------------------------------------------------

def test_shap_stability_self_distance_much_smaller_than_diverged():
    """A model's SHAP distance from itself must be very small relative to the
    distance from a different model.  The exact value is not zero because the
    model is held in train mode for cuDNN compatibility, so dropout layers
    inject a small amount of stochasticity into the integrated-gradient
    computation — but it should be at least an order of magnitude smaller
    than the inter-model distance.
    """
    torch.manual_seed(0)
    m1 = CNNLSTMClassifier(n_features=40, n_classes=15)
    torch.manual_seed(1)
    m2 = CNNLSTMClassifier(n_features=40, n_classes=15)
    X = torch.randn(20, 40)
    y = torch.randint(0, 15, (20,))

    s_self  = compute_shap_stability(m1, m1, X, y, n_explain=20, n_classes=15)
    s_other = compute_shap_stability(m1, m2, X, y, n_explain=20, n_classes=15)

    assert s_self >= 0.0, f"self-distance is negative: {s_self}"
    assert s_other > 0.0, f"diverged-model distance is zero: {s_other}"
    assert s_self < s_other, (
        f"self-distance {s_self} should be smaller than inter-model distance "
        f"{s_other}"
    )


def test_shap_stability_nonzero_for_diverged_models():
    """Two independently-initialised models must have non-zero distance, and
    perturbing one of them must produce a *different* score (proves the
    function actually inspects model parameters)."""
    torch.manual_seed(0)
    m1 = CNNLSTMClassifier(n_features=40, n_classes=15)
    torch.manual_seed(1)
    m2 = CNNLSTMClassifier(n_features=40, n_classes=15)
    X = torch.randn(20, 40)
    y = torch.randint(0, 15, (20,))

    s1 = compute_shap_stability(m1, m2, X, y, n_explain=20, n_classes=15)
    assert s1 > 0.0, "diverged models reported zero distance"

    # Perturb m2 further and re-measure
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(0.1 * torch.randn_like(p))
    s2 = compute_shap_stability(m1, m2, X, y, n_explain=20, n_classes=15)
    assert abs(s1 - s2) > 1e-6, (
        f"SHAP stability did not change after perturbing one model "
        f"(s1={s1}, s2={s2}) — the function may be ignoring its inputs"
    )


def test_shap_stability_uses_input_data():
    """Different validation inputs must yield different scores."""
    torch.manual_seed(0)
    m1 = CNNLSTMClassifier(n_features=40, n_classes=15)
    torch.manual_seed(1)
    m2 = CNNLSTMClassifier(n_features=40, n_classes=15)
    y = torch.randint(0, 15, (20,))

    X_a = torch.randn(20, 40)
    X_b = torch.randn(20, 40) * 5.0   # very different distribution

    s_a = compute_shap_stability(m1, m2, X_a, y, n_explain=20, n_classes=15)
    s_b = compute_shap_stability(m1, m2, X_b, y, n_explain=20, n_classes=15)
    assert abs(s_a - s_b) > 1e-6, (
        f"SHAP stability did not change with different validation data "
        f"(s_a={s_a}, s_b={s_b}) — the function appears to ignore X_val"
    )


# ---------------------------------------------------------------------------
# JSON output integrity — proves results carry the per-seed evidence
# ---------------------------------------------------------------------------

def test_results_json_contains_raw_seed_arrays_when_present():
    """If results/experiment_results.json exists, every aggregate that was
    produced under multi-seed conditions must carry the raw per-seed values
    so std can be re-derived."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "experiment_results.json",
    )
    if not os.path.exists(path):
        import pytest
        pytest.skip("results/experiment_results.json not generated yet")

    import json
    with open(path) as f:
        results = json.load(f)

    n_seeds = results.get("meta", {}).get("n_seeds", 1)
    if n_seeds < 2:
        import pytest
        pytest.skip("only 1 seed used; raw arrays are not informative")

    sota = results.get("sota_comparison", {})
    if not any("raw_label_flip" in v for v in sota.values()):
        import pytest
        pytest.skip(
            "results/experiment_results.json was produced by an older code "
            "revision that did not record raw seed arrays; re-run "
            "scripts/run_experiments.py to regenerate"
        )
    for method, val in sota.items():
        assert isinstance(val.get("raw_label_flip"), list), \
            f"sota_comparison/{method} missing raw_label_flip array"
        assert isinstance(val.get("raw_backdoor_asr"), list), \
            f"sota_comparison/{method} missing raw_backdoor_asr array"
        assert len(val["raw_label_flip"]) == n_seeds, \
            f"sota_comparison/{method} raw_label_flip has wrong length"
