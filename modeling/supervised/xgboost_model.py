"""
modeling/supervised/xgboost_model.py
=====================================
XGBoost classifier with Optuna Bayesian hyperparameter optimisation.

Scientific protocol:
  - 100-trial Optuna TPE search, objective = 5-fold CV AUPRC
  - Early stopping: 50 rounds on validation AUPRC per fold
  - scale_pos_weight = neg/pos for cost-sensitive training
  - SHAP TreeExplainer (exact, not approximation)
  - Threshold via Expected Cost minimisation at theta*
  - Bootstrap CIs (n=1000) for AUPRC and AUROC

Outputs (in models/):
  xgb_model_v1.joblib
  xgb_results_v1.json
  report/figures/shap/xgb_*
  report/figures/models/xgb_feature_importance_v1.pdf
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import shap
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modeling.supervised.model_selection import (
    RANDOM_STATE,
    compute_metrics,
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

plt.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.dpi": 300})

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
SHAP_DIR = ROOT / "report" / "figures" / "shap"
MODEL_FIG_DIR = ROOT / "report" / "figures" / "models"
SHAP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5


def _auprc_cv_fold(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    """Single CV evaluation for Optuna objective (AUPRC)."""
    scale_pos_weight = float((y == 0).sum()) / float((y == 1).sum())
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            device="cpu",
            random_state=RANDOM_STATE,
            verbosity=0,
            early_stopping_rounds=50,
        )
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, y_prob))
    return float(np.mean(scores))


def objective(trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective: maximise 5-fold CV AUPRC."""
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 2000),
        "max_depth":          trial.suggest_int("max_depth", 3, 10),
        "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
    }
    return _auprc_cv_fold(params, X, y)


def train_xgboost(
    version: int = 1,
    n_optuna_trials: int = 100,
) -> dict:
    """Full XGBoost training + evaluation pipeline."""
    log.info("=== XGBoost Training ===")

    X_train = np.load(MODELS_DIR / f"X_train_v{version}.npy")
    X_test  = np.load(MODELS_DIR / f"X_test_v{version}.npy")
    y_train = np.load(MODELS_DIR / f"y_train_v{version}.npy")
    y_test  = np.load(MODELS_DIR / f"y_test_v{version}.npy")
    log.info(f"Loaded arrays: train={X_train.shape}, test={X_test.shape}")

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    log.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

    # --- Optuna hyperopt ---
    log.info(f"Starting Optuna hyperopt ({n_optuna_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_optuna_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    log.info(f"Best CV AUPRC: {study.best_value:.4f}")
    log.info(f"Best params: {json.dumps(best_params, indent=2)}")

    # --- Final model on full train set ---
    log.info("Fitting final XGBoost on full training set...")
    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cpu",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    # No early stopping on final fit — use n_estimators from hyperopt
    final_model.fit(X_train, y_train, verbose=False)

    # --- Predictions ---
    y_prob = final_model.predict_proba(X_test)[:, 1]
    np.save(MODELS_DIR / f"xgb_y_prob_v{version}.npy", y_prob)

    # --- Threshold ---
    theta_star, min_cost = optimal_threshold_cost(y_test, y_prob)
    log.info(f"Optimal threshold: {theta_star:.4f} (cost/txn: )")

    # --- Metrics ---
    metrics = compute_metrics(y_test, y_prob, threshold=theta_star, model_name="XGBoost")
    ci = full_bootstrap_ci(y_test, y_prob, threshold=theta_star)
    metrics.update(ci)
    metrics["best_cv_auprc"] = round(study.best_value, 4)
    metrics["optuna_params"] = best_params
    log.info(f"Test AUPRC: {metrics['auprc']:.4f}  AUROC: {metrics['auroc']:.4f}  MCC: {metrics['mcc']:.4f}")

    # --- Save model ---
    model_path = MODELS_DIR / f"xgb_model_v{version}.joblib"
    joblib.dump(final_model, model_path)
    log.info(f"Model saved -> {model_path}")

    # --- Feature importance (all three types) ---
    _plot_feature_importance(final_model, version)

    # --- SHAP (TreeExplainer - exact) ---
    _compute_shap(final_model, X_train, X_test, version)

    save_results(metrics, f"xgb_v{version}")
    return metrics


def _plot_feature_importance(model: XGBClassifier, version: int, top_n: int = 25) -> None:
    """Plot gain / cover / frequency importance for top N features."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    importance_types = ["gain", "cover", "weight"]
    titles = ["Gain", "Cover", "Frequency (Weight)"]

    for ax, imp_type, title in zip(axes, importance_types, titles):
        scores = model.get_booster().get_score(importance_type=imp_type)
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])
        ax.barh(list(scores.keys())[::-1], list(scores.values())[::-1], color="#2c7bb6")
        ax.set_title(f"XGBoost — {title}")
        ax.set_xlabel(title)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"XGBoost Feature Importance (Top {top_n})", fontsize=13)
    plt.tight_layout()
    path = MODEL_FIG_DIR / f"xgb_feature_importance_v{version}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Feature importance saved -> {path}")


def _compute_shap(
    model: XGBClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    version: int,
) -> None:
    """TreeExplainer SHAP (exact values, not approximation)."""
    log.info("Computing SHAP values (TreeExplainer, exact)...")
    explainer = shap.TreeExplainer(model)
    # Use all test samples for TreeExplainer (fast for trees)
    shap_values = explainer.shap_values(X_test)
    np.save(MODELS_DIR / f"xgb_shap_v{version}.npy", shap_values)

    feat_names_path = MODELS_DIR / f"feature_names_v{version}.json"
    feat_names = json.loads(feat_names_path.read_text()) if feat_names_path.exists() else None

    # Beeswarm summary
    fig, _ = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feat_names, show=False, plot_size=None)
    plt.tight_layout()
    path = SHAP_DIR / f"xgb_shap_beeswarm_v{version}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close("all")
    log.info(f"SHAP beeswarm saved -> {path}")

    # Bar summary (mean |SHAP|)
    fig, _ = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_test, feature_names=feat_names,
        plot_type="bar", show=False, plot_size=None
    )
    plt.tight_layout()
    path_bar = SHAP_DIR / f"xgb_shap_bar_v{version}.pdf"
    fig.savefig(path_bar, bbox_inches="tight")
    plt.close("all")
    log.info(f"SHAP bar saved -> {path_bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    results = train_xgboost(version=args.version, n_optuna_trials=args.trials)
    print(json.dumps(results, indent=2))
