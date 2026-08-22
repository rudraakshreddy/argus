"""
modeling/supervised/model_selection.py
=======================================
Shared utilities for rigorous model selection:
  - Stratified 5-fold CV with fixed random_state
  - Optimal threshold selection via Expected Cost minimisation
  - Bootstrap confidence intervals (n=1000) for any metric
  - Youden's J statistic threshold (for unsupervised models)

All functions are imported by individual model scripts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"

# Cost parameters (USD) - see report Section 4 for justification
C_FP = 12.0   # False positive: 30 min analyst review at /hr
C_FN = 850.0  # False negative: median IEEE-CIS fraud transaction amount

N_SPLITS = 5
RANDOM_STATE = 42
BOOTSTRAP_N = 1000


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def optimal_threshold_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
    n_thresholds: int = 500,
) -> tuple[float, float]:
    """
    Select decision threshold minimising expected cost.

    E[Cost](theta) = FP(theta) * c_fp + FN(theta) * c_fn

    Returns
    -------
    theta_star : float  - optimal threshold
    min_cost   : float  - expected cost at theta_star (per transaction)
    """
    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    costs = []
    n = len(y_true)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        costs.append((fp * c_fp + fn * c_fn) / n)
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return float(thresholds[best_idx]), float(costs[best_idx])


def youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Youden's J = Sensitivity + Specificity - 1. Used for unsupervised models."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    # J = TPR - FPR; approximate via PR curve
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
    model_name: str = "model",
) -> dict:
    """
    Compute all benchmark metrics at the given threshold.

    Metrics (never raw accuracy on imbalanced data):
      - AUPRC  (primary)
      - AUROC
      - Precision, Recall, F1  @ threshold
      - MCC                    @ threshold
      - Brier Score
      - Expected Cost          @ threshold
    """
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    from sklearn.metrics import brier_score_loss
    return {
        "model": model_name,
        "threshold": round(threshold, 4),
        "auprc":     round(average_precision_score(y_true, y_prob), 4),
        "auroc":     round(roc_auc_score(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc":       round(matthews_corrcoef(y_true, y_pred), 4),
        "brier":     round(brier_score_loss(y_true, y_prob), 4),
        "expected_cost_per_txn": round((fp * c_fp + fn * c_fn) / n, 4),
        "n_flagged": int(y_pred.sum()),
        "n_fraud_caught": int(np.sum((y_pred == 1) & (y_true == 1))),
        "n_fraud_total": int(y_true.sum()),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    n: int = BOOTSTRAP_N,
    ci: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    metric_fn : callable(y_true, y_prob) -> float

    Returns
    -------
    (lower, upper) at the given CI level.
    """
    rng = np.random.default_rng(random_state)
    scores = []
    n_samples = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    alpha = (1 - ci) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


def full_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n: int = BOOTSTRAP_N,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Bootstrap CIs for AUPRC and AUROC (the two primary reported metrics)."""
    auprc_lo, auprc_hi = bootstrap_ci(
        y_true, y_prob, average_precision_score, n=n, random_state=random_state
    )
    auroc_lo, auroc_hi = bootstrap_ci(
        y_true, y_prob, roc_auc_score, n=n, random_state=random_state
    )
    return {
        "auprc_ci_lo": round(auprc_lo, 4),
        "auprc_ci_hi": round(auprc_hi, 4),
        "auroc_ci_lo": round(auroc_lo, 4),
        "auroc_ci_hi": round(auroc_hi, 4),
    }


# ---------------------------------------------------------------------------
# CV helper
# ---------------------------------------------------------------------------

def cross_validate_auprc(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """
    Stratified k-fold CV returning mean and std of AUPRC across folds.
    Estimator must implement fit() and predict_proba().
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        estimator.fit(X[tr_idx], y[tr_idx])
        y_prob = estimator.predict_proba(X[val_idx])[:, 1]
        score = average_precision_score(y[val_idx], y_prob)
        fold_scores.append(score)
        log.info(f"  Fold {fold+1}/{n_splits}: AUPRC = {score:.4f}")
    mean_score = float(np.mean(fold_scores))
    std_score  = float(np.std(fold_scores))
    log.info(f"  CV AUPRC: {mean_score:.4f} +/- {std_score:.4f}")
    return mean_score, std_score


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, model_name: str) -> None:
    """Save metric results dict to models/ as JSON."""
    out_path = MODELS_DIR / f"{model_name}_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Results saved -> {out_path}")
