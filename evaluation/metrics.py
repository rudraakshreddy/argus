"""
evaluation/metrics.py
======================
Comprehensive metric computation for all four fraud detection models.

Metrics computed (never raw accuracy on imbalanced data):
  Primary:
    - AUPRC  (Average Precision = area under Precision-Recall curve)
  Secondary:
    - AUROC  (Area Under ROC curve)
  At optimal threshold theta*:
    - Precision, Recall, F1
    - Matthews Correlation Coefficient (MCC)
    - Brier Score (calibration quality)
    - Expected Cost per transaction (USD)
  Statistical:
    - Bootstrap 95% CI for AUPRC and AUROC (n=1000 resamples)
    - Wilcoxon signed-rank test between best two models

All results are saved to:
  models/all_metrics.json         -- machine-readable
  report/figures/models/          -- LaTeX-formatted table (via model_comparison.py)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling.supervised.model_selection import (
    BOOTSTRAP_N,
    C_FP,
    C_FN,
    RANDOM_STATE,
    bootstrap_ci,
    full_bootstrap_ci,
    optimal_threshold_cost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

MODEL_CONFIGS = [
    {"name": "Logistic Regression", "prob_file": "lr_y_prob_v1.npy",  "supervised": True},
    {"name": "XGBoost",             "prob_file": "xgb_y_prob_v1.npy", "supervised": True},
    {"name": "Isolation Forest",    "prob_file": "if_y_prob_v1.npy",  "supervised": False},
    {"name": "Autoencoder",         "prob_file": "ae_y_prob_v1.npy",  "supervised": False},
]


def compute_all_metrics(version: int = 1) -> dict:
    """
    Load predictions for all models and compute the full metric suite.

    Returns
    -------
    dict: model_name -> full metrics dict including bootstrap CIs
    """
    log.info("=== Computing all model metrics ===")
    y_test = np.load(MODELS_DIR / f"y_test_v{version}.npy")
    log.info(f"Test set: {len(y_test):,} samples | fraud rate: {y_test.mean():.4f}")

    all_metrics = {}

    for cfg in MODEL_CONFIGS:
        prob_path = MODELS_DIR / cfg["prob_file"]
        if not prob_path.exists():
            log.warning(f"Skipping {cfg['name']}: {prob_path} not found")
            continue

        log.info(f"\n--- {cfg['name']} ---")
        y_prob = np.load(prob_path)

        # Optimal threshold
        theta_star, min_cost = optimal_threshold_cost(y_test, y_prob)
        y_pred = (y_prob >= theta_star).astype(int)

        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tp = np.sum((y_pred == 1) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))

        # Point estimates
        metrics = {
            "model":       cfg["name"],
            "supervised":  cfg["supervised"],
            "threshold":   round(float(theta_star), 4),
            "auprc":       round(float(average_precision_score(y_test, y_prob)), 4),
            "auroc":       round(float(roc_auc_score(y_test, y_prob)), 4),
            "precision":   round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":      round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1":          round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "mcc":         round(float(matthews_corrcoef(y_test, y_pred)), 4),
            "brier":       round(float(brier_score_loss(y_test, y_prob)), 4),
            "exp_cost_per_txn": round(float(min_cost), 4),
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
            "n_flagged":        int(y_pred.sum()),
            "n_fraud_total":    int(y_test.sum()),
            "n_fraud_caught":   int(tp),
            "dollar_saved":     round(float(tp * C_FN - fp * C_FP), 2),
        }

        # Bootstrap CIs
        log.info(f"  Computing bootstrap CIs (n={BOOTSTRAP_N})...")
        ci = full_bootstrap_ci(y_test, y_prob, threshold=theta_star, n=BOOTSTRAP_N)
        metrics.update(ci)

        # Log key metrics
        log.info(
            f"  AUPRC={metrics['auprc']:.4f} [{metrics['auprc_ci_lo']:.4f}-{metrics['auprc_ci_hi']:.4f}]  "
            f"AUROC={metrics['auroc']:.4f}  MCC={metrics['mcc']:.4f}  "
            f"F1={metrics['f1']:.4f}  Brier={metrics['brier']:.4f}"
        )
        log.info(
            f"  Threshold={theta_star:.4f}  Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}"
        )
        log.info(
            f"  TP={tp} FP={fp} TN={tn} FN={fn}  "
            f"Dollar saved at theta*: "
        )

        all_metrics[cfg["name"]] = metrics

    # --- Wilcoxon signed-rank test: best supervised vs best unsupervised ---
    _run_wilcoxon_test(all_metrics, y_test)

    # Save
    out_path = MODELS_DIR / "all_metrics.json"
    out_path.write_text(json.dumps(all_metrics, indent=2))
    log.info(f"\nAll metrics saved -> {out_path}")

    return all_metrics


def _run_wilcoxon_test(all_metrics: dict, y_test: np.ndarray) -> None:
    """
    Wilcoxon signed-rank test comparing best supervised vs best unsupervised model.

    Test: H0: distributions of pointwise scores are equal.
    Two-sided, alpha=0.05.
    """
    supervised_models = [
        m for m, cfg in zip(MODEL_CONFIGS, MODEL_CONFIGS) if cfg["supervised"]
        # Re-select by checking metrics dict
    ]
    sup_names = [cfg["name"] for cfg in MODEL_CONFIGS if cfg["supervised"] and cfg["name"] in all_metrics]
    unsup_names = [cfg["name"] for cfg in MODEL_CONFIGS if not cfg["supervised"] and cfg["name"] in all_metrics]

    if not sup_names or not unsup_names:
        log.warning("Cannot run Wilcoxon test: need at least one supervised and one unsupervised model")
        return

    # Pick best of each by AUPRC
    best_sup   = max(sup_names,   key=lambda n: all_metrics[n]["auprc"])
    best_unsup = max(unsup_names, key=lambda n: all_metrics[n]["auprc"])

    prob_sup   = np.load(MODELS_DIR / f"{'lr' if 'Logistic' in best_sup else 'xgb'}_y_prob_v1.npy")
    prob_unsup = np.load(MODELS_DIR / f"{'ae' if 'Auto' in best_unsup else 'if'}_y_prob_v1.npy")

    # Pointwise difference in squared error (Brier-like per-sample)
    # Wilcoxon on per-sample squared probability errors
    sq_err_sup   = (prob_sup   - y_test) ** 2
    sq_err_unsup = (prob_unsup - y_test) ** 2

    stat, p_value = wilcoxon(sq_err_sup, sq_err_unsup, alternative="two-sided")
    result = {
        "test": "Wilcoxon signed-rank",
        "model_A": best_sup,
        "model_B": best_unsup,
        "H0": "Distributions of per-sample squared errors are equal",
        "statistic": round(float(stat), 4),
        "p_value":   round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
        "interpretation": (
            f"{'Reject' if p_value < 0.05 else 'Fail to reject'} H0 at alpha=0.05. "
            f"{best_sup} {'significantly' if p_value < 0.05 else 'not significantly'} "
            f"outperforms {best_unsup} (p={p_value:.4f})."
        ),
    }
    log.info("\n--- Wilcoxon Signed-Rank Test ---")
    log.info(f"  {best_sup} vs {best_unsup}")
    log.info(f"  statistic={stat:.4f}  p={p_value:.6f}")
    log.info(f"  {result['interpretation']}")

    # Save to metrics dir
    wilcoxon_path = MODELS_DIR / "wilcoxon_test.json"
    wilcoxon_path.write_text(json.dumps(result, indent=2))
    log.info(f"  Wilcoxon results saved -> {wilcoxon_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    results = compute_all_metrics(version=args.version)
