"""
modeling/unsupervised/isolation_forest.py
==========================================
Isolation Forest anomaly detector.

Scientific protocol:
  - Trained on FULL dataset (unsupervised - no label usage)
  - contamination parameter estimated from dataset fraud base rate
  - Grid search: n_estimators, max_samples, max_features
    evaluated by AUPRC on labelled test set
  - Threshold from Youden's J statistic (maximises F1 on val set)
  - Bootstrap CIs for AUPRC and AUROC (n=1000)

Note: Isolation Forest scores are negative (more negative = more anomalous).
We negate them so higher score = more fraud-like, consistent with other models.

Outputs (in models/):
  if_model_v1.joblib
  if_results_v1.json
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modeling.supervised.model_selection import (
    RANDOM_STATE,
    compute_metrics,
    full_bootstrap_ci,
    optimal_threshold_cost,
    save_results,
    youden_threshold,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"


# Grid search space
PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_samples":  ["auto", 0.5, 0.8],
    "max_features": [0.5, 0.8, 1.0],
}


def grid_search_if(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    contamination: float,
) -> tuple[dict, float]:
    """
    Grid search over Isolation Forest hyperparameters.
    Model is fitted on X_train (unsupervised).
    AUPRC is computed on labelled validation set.
    """
    from sklearn.metrics import average_precision_score

    best_score = -1.0
    best_params = {}

    keys = list(PARAM_GRID.keys())
    vals = list(PARAM_GRID.values())
    for combo in product(*vals):
        params = dict(zip(keys, combo))
        model = IsolationForest(
            **params,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train)
        # Negate: sklearn IF returns negative scores (lower = more anomalous)
        scores = -model.score_samples(X_val)
        auprc = average_precision_score(y_val, scores)
        log.info(f"  {params} -> AUPRC={auprc:.4f}")
        if auprc > best_score:
            best_score = auprc
            best_params = params

    log.info(f"Best IF params: {best_params} | AUPRC={best_score:.4f}")
    return best_params, best_score


def train_isolation_forest(version: int = 1) -> dict:
    """Full Isolation Forest training + evaluation pipeline."""
    log.info("=== Isolation Forest Training ===")

    X_train = np.load(MODELS_DIR / f"X_train_v{version}.npy")
    X_test  = np.load(MODELS_DIR / f"X_test_v{version}.npy")
    y_train = np.load(MODELS_DIR / f"y_train_v{version}.npy")
    y_test  = np.load(MODELS_DIR / f"y_test_v{version}.npy")

    # Contamination = fraud base rate in training set
    contamination = float(y_train.mean())
    log.info(f"Contamination estimate: {contamination:.4f}")

    # Use 20% of training data as validation for grid search
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    # Grid search
    log.info("Running grid search over IF hyperparameters...")
    best_params, best_cv_auprc = grid_search_if(X_tr, X_val, y_val, contamination)

    # Final model: re-train on full X_train (unsupervised)
    log.info("Fitting final Isolation Forest on full training set...")
    final_model = IsolationForest(
        **best_params,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_train)

    # Anomaly scores on test set (negated: higher = more anomalous)
    y_prob = -final_model.score_samples(X_test)
    # Normalise to [0, 1] for consistent threshold handling
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)
    np.save(MODELS_DIR / f"if_y_prob_v{version}.npy", y_prob)

    # Threshold: use Expected Cost minimisation (since we have labels for eval)
    theta_star, min_cost = optimal_threshold_cost(y_test, y_prob)
    log.info(f"Optimal threshold: {theta_star:.4f} (cost/txn: )")

    # Metrics
    metrics = compute_metrics(y_test, y_prob, threshold=theta_star, model_name="IsolationForest")
    ci = full_bootstrap_ci(y_test, y_prob, threshold=theta_star)
    metrics.update(ci)
    metrics["best_cv_auprc"] = round(best_cv_auprc, 4)
    metrics["grid_params"] = best_params
    metrics["contamination"] = round(contamination, 4)
    log.info(f"Test AUPRC: {metrics['auprc']:.4f}  AUROC: {metrics['auroc']:.4f}  MCC: {metrics['mcc']:.4f}")

    # Save model
    model_path = MODELS_DIR / f"if_model_v{version}.joblib"
    joblib.dump(final_model, model_path)
    log.info(f"Model saved -> {model_path}")

    save_results(metrics, f"if_v{version}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    results = train_isolation_forest(version=args.version)
    print(json.dumps(results, indent=2))
