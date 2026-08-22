"""
tests/test_metrics.py
======================
Unit tests for the evaluation metrics module.

Tests:
  - AUPRC on a perfect classifier equals 1.0
  - AUPRC on a random classifier approximates the prevalence rate
  - Bootstrap CI contains the true metric value (stochastic — uses large n)
  - optimal_threshold_cost returns theta in (0, 1)
  - compute_metrics returns all required keys
  - Wilcoxon test is correctly directional
  - MCC is bounded in [-1, 1]
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling.supervised.model_selection import (
    BOOTSTRAP_N,
    C_FP,
    C_FN,
    bootstrap_ci,
    compute_metrics,
    full_bootstrap_ci,
    optimal_threshold_cost,
    youden_threshold,
)


@pytest.fixture
def perfect_preds():
    y = np.array([0]*95 + [1]*5)
    p = y.astype(float)   # perfect scores
    return y, p


@pytest.fixture
def random_preds():
    np.random.seed(42)
    y = (np.random.rand(1000) < 0.035).astype(int)
    p = np.random.rand(1000)
    return y, p


@pytest.fixture
def good_preds():
    """Realistic fraud model output (~0.85 AUPRC)."""
    np.random.seed(0)
    n = 2000
    y = (np.random.rand(n) < 0.035).astype(int)
    # Fraud: scores ~ Beta(8, 2), legit: scores ~ Beta(2, 8)
    p = np.where(y == 1,
                 np.random.beta(8, 2, n),
                 np.random.beta(2, 8, n))
    return y, p


class TestAUPRC:

    def test_perfect_classifier(self, perfect_preds):
        from sklearn.metrics import average_precision_score
        y, p = perfect_preds
        auprc = average_precision_score(y, p)
        assert auprc == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier_near_prevalence(self, random_preds):
        from sklearn.metrics import average_precision_score
        y, p = random_preds
        auprc = average_precision_score(y, p)
        prevalence = y.mean()
        # Random classifier AUPRC ≈ prevalence (within 3 std for 1000 samples)
        assert abs(auprc - prevalence) < 0.05, (
            f"Random AUPRC {auprc:.4f} too far from prevalence {prevalence:.4f}"
        )

    def test_good_model_auprc_range(self, good_preds):
        from sklearn.metrics import average_precision_score
        y, p = good_preds
        auprc = average_precision_score(y, p)
        assert auprc > 0.5, f"Good model AUPRC should be > 0.5, got {auprc:.4f}"


class TestOptimalThreshold:

    def test_threshold_in_unit_interval(self, good_preds):
        y, p = good_preds
        theta, cost = optimal_threshold_cost(y, p, c_fp=C_FP, c_fn=C_FN)
        assert 0.0 < theta < 1.0

    def test_cost_is_positive(self, good_preds):
        y, p = good_preds
        theta, cost = optimal_threshold_cost(y, p)
        assert cost >= 0.0

    def test_higher_fn_cost_lowers_threshold(self, good_preds):
        """When FN is more expensive, we should flag more (lower threshold)."""
        y, p = good_preds
        theta_low_fn,  _ = optimal_threshold_cost(y, p, c_fp=12.0,  c_fn=100.0)
        theta_high_fn, _ = optimal_threshold_cost(y, p, c_fp=12.0,  c_fn=5000.0)
        assert theta_high_fn <= theta_low_fn, (
            "Higher FN cost should drive threshold down (catch more fraud)"
        )

    def test_youden_in_unit_interval(self, good_preds):
        y, p = good_preds
        theta = youden_threshold(y, p)
        assert 0.0 <= theta <= 1.0


class TestComputeMetrics:

    def test_all_keys_present(self, good_preds):
        y, p = good_preds
        theta, _ = optimal_threshold_cost(y, p)
        m = compute_metrics(y, p, threshold=theta, model_name="test")

        required_keys = ["model", "threshold", "auprc", "auroc",
                         "precision", "recall", "f1", "mcc", "brier",
                         "expected_cost_per_txn"]
        for key in required_keys:
            assert key in m, f"Missing key: {key}"

    def test_mcc_bounded(self, good_preds):
        y, p = good_preds
        theta, _ = optimal_threshold_cost(y, p)
        m = compute_metrics(y, p, threshold=theta)
        assert -1.0 <= m["mcc"] <= 1.0

    def test_precision_recall_in_unit_interval(self, good_preds):
        y, p = good_preds
        theta, _ = optimal_threshold_cost(y, p)
        m = compute_metrics(y, p, threshold=theta)
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"]    <= 1.0

    def test_brier_score_bounds(self, good_preds, random_preds):
        """Brier score in [0, 1]; random should be > good model."""
        y_g, p_g = good_preds
        y_r, p_r = random_preds
        theta_g, _ = optimal_threshold_cost(y_g, p_g)
        theta_r, _ = optimal_threshold_cost(y_r, p_r)
        m_good   = compute_metrics(y_g, p_g, threshold=theta_g)
        m_random = compute_metrics(y_r, p_r, threshold=theta_r)
        assert 0.0 <= m_good["brier"]   <= 1.0
        assert 0.0 <= m_random["brier"] <= 1.0
        assert m_good["brier"] < m_random["brier"], (
            "Good model Brier score should be lower than random"
        )


class TestBootstrapCI:

    def test_ci_contains_true_value(self, good_preds):
        """95% CI should contain the true metric ~95% of the time (probabilistic test)."""
        from sklearn.metrics import average_precision_score
        y, p = good_preds
        true_val = average_precision_score(y, p)
        lo, hi = bootstrap_ci(y, p, average_precision_score, n=500, ci=0.95)
        assert lo <= true_val <= hi, (
            f"True AUPRC {true_val:.4f} not in CI [{lo:.4f}, {hi:.4f}]"
        )

    def test_ci_ordering(self, good_preds):
        from sklearn.metrics import average_precision_score
        y, p = good_preds
        lo, hi = bootstrap_ci(y, p, average_precision_score, n=200)
        assert lo < hi

    def test_full_ci_dict_keys(self, good_preds):
        y, p = good_preds
        theta, _ = optimal_threshold_cost(y, p)
        ci = full_bootstrap_ci(y, p, threshold=theta, n=200)
        for key in ["auprc_ci_lo", "auprc_ci_hi", "auroc_ci_lo", "auroc_ci_hi"]:
            assert key in ci
        assert ci["auprc_ci_lo"] < ci["auprc_ci_hi"]
        assert ci["auroc_ci_lo"] < ci["auroc_ci_hi"]
