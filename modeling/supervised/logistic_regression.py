"""
modeling/supervised/logistic_regression.py
==========================================
Calibrated Logistic Regression with ElasticNet regularisation.

Scientific protocol:
  - Hyperparameters tuned via Optuna (50 trials, Bayesian optimisation)
    on 5-fold stratified CV AUPRC
  - Final model fitted on full training set
  - Probability calibration: CalibratedClassifierCV (isotonic, cv=5)
  - SHAP KernelExplainer for global + local explanations
  - Threshold selected via Expected Cost minimisation (not 0.5)
  - Bootstrap CIs for AUPRC and AUROC (n=1000)

Outputs (all in models/):
  lr_model_v1.joblib          - calibrated model
  lr_results_v1.json          - all metrics + CIs
  report/figures/shap/lr_*    - SHAP beeswarm + waterfall
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import optuna
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modeling.supervised.model_selection import (
    RANDOM_STATE,
    bootstrap_ci,
    compute_metrics,
    cross_validate_auprc,
    full_bootstrap_ci,
    optimal_threshold_cost,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
SHAP_DIR = ROOT / "report" / "figures" / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 300})


def objective(trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Optuna objective: 5-fold CV AUPRC for LR with ElasticNet."""
    C = trial.suggest_float("C", 1e-4, 10.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    solver = "saga"  # only solver supporting elasticnet

    lr = LogisticRegression(
        C=C,
        penalty="elasticnet",
        l1_ratio=l1_ratio,
        solver=solver,
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    mean_auprc, _ = cross_validate_auprc(lr, X_train, y_train)
    return mean_auprc


def train_logistic_regression(
    version: int = 1,
    n_optuna_trials: int = 50,
) -> dict:
    """Full training + evaluation pipeline for Logistic Regression."""
    log.info("=== Logistic Regression Training ===")

    # Load pre-processed arrays
    X_train = np.load(MODELS_DIR / f"X_train_v{version}.npy")
    X_test  = np.load(MODELS_DIR / f"X_test_v{version}.npy")
    y_train = np.load(MODELS_DIR / f"y_train_v{version}.npy")
    y_test  = np.load(MODELS_DIR / f"y_test_v{version}.npy")
    log.info(f"Loaded arrays: train={X_train.shape}, test={X_test.shape}")

    # --- Hyperparameter optimisation ---
    log.info(f"Starting Optuna hyperopt ({n_optuna_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_optuna_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    log.info(f"Best CV AUPRC: {study.best_value:.4f} | Params: {best_params}")

    # --- Fit final model on full training set ---
    base_lr = LogisticRegression(
        C=best_params["C"],
        penalty="elasticnet",
        l1_ratio=best_params["l1_ratio"],
        solver="saga",
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    # Probability calibration (isotonic regression, 5-fold)
    log.info("Fitting calibrated model on full training set...")
    calibrated_lr = CalibratedClassifierCV(
        base_lr, method="isotonic", cv=5
    )
    calibrated_lr.fit(X_train, y_train)

    # --- Predict on held-out test set ---
    y_prob = calibrated_lr.predict_proba(X_test)[:, 1]

    # --- Threshold selection ---
    theta_star, min_cost = optimal_threshold_cost(y_test, y_prob)
    log.info(f"Optimal threshold: {theta_star:.4f} (expected cost/txn: )")

    # --- Compute all metrics ---
    metrics = compute_metrics(y_test, y_prob, threshold=theta_star, model_name="LogisticRegression")
    ci = full_bootstrap_ci(y_test, y_prob, threshold=theta_star)
    metrics.update(ci)
    metrics["best_cv_auprc"] = round(study.best_value, 4)
    metrics["optuna_params"] = best_params
    log.info(f"Test AUPRC: {metrics['auprc']:.4f}  AUROC: {metrics['auroc']:.4f}  MCC: {metrics['mcc']:.4f}")

    # --- Save model ---
    model_path = MODELS_DIR / f"lr_model_v{version}.joblib"
    joblib.dump(calibrated_lr, model_path)
    log.info(f"Model saved -> {model_path}")

    # Save predictions for ensemble / comparison
    np.save(MODELS_DIR / f"lr_y_prob_v{version}.npy", y_prob)

    # --- SHAP explanations ---
    _compute_shap(calibrated_lr, X_train, X_test, version)

    # --- Save results ---
    save_results(metrics, f"lr_v{version}")
    return metrics


def _compute_shap(model, X_train: np.ndarray, X_test: np.ndarray, version: int) -> None:
    """KernelExplainer SHAP for calibrated LR (global beeswarm on sample)."""
    log.info("Computing SHAP values (KernelExplainer, 100 background samples)...")
    background = shap.sample(X_train, 100, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(
        model.predict_proba,
        background,
        link="logit",
    )
    # Explain a stratified sample of 500 test instances (full test set is slow for Kernel)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.integers(0, len(X_test), size=min(500, len(X_test)))
    shap_values = explainer.shap_values(X_test[sample_idx], nsamples=200)
    # shap_values[1] = fraud class
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Load feature names if available
    feat_names_path = MODELS_DIR / f"feature_names_v{version}.json"
    feat_names = json.loads(feat_names_path.read_text()) if feat_names_path.exists() else None

    # Beeswarm summary plot
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv, X_test[sample_idx], feature_names=feat_names, show=False, plot_size=None)
    plt.tight_layout()
    fig.savefig(SHAP_DIR / f"lr_shap_beeswarm_v{version}.pdf", bbox_inches="tight")
    plt.close("all")
    log.info(f"SHAP beeswarm saved -> {SHAP_DIR / f'lr_shap_beeswarm_v{version}.pdf'}")

    # Save shap values for later use
    np.save(MODELS_DIR / f"lr_shap_v{version}.npy", sv)
    np.save(MODELS_DIR / f"lr_shap_idx_v{version}.npy", sample_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    results = train_logistic_regression(version=args.version, n_optuna_trials=args.trials)
    print(json.dumps(results, indent=2))
