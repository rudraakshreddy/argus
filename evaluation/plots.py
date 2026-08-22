"""
evaluation/plots.py
====================
All publication-quality evaluation plots for the report.

Figures generated (all saved as PDF to report/figures/models/):
  01_roc_curves.pdf           - ROC curves for all four models
  02_pr_curves.pdf            - Precision-Recall curves (primary for imbalanced data)
  03_confusion_matrices.pdf   - Confusion matrices at theta* for all models
  04_calibration_curves.pdf   - Reliability diagrams for supervised models
  05_threshold_sweep.pdf      - Precision/Recall/F1 vs threshold sweep
  06_shap_beeswarm_xgb.pdf    - Copied from shap/ dir (already generated)

All plots use:
  - 300 DPI, PDF vector format (for LaTeX pdflatex)
  - Serif fonts (consistent with Elsevier elsarticle style)
  - Colourblind-safe palette where possible
  - Informative axis labels, titles, and legends
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling.supervised.model_selection import optimal_threshold_cost

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "report" / "figures" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    {"name": "Logistic Regression", "prob_file": "lr_y_prob_v1.npy",  "color": "#1a9641", "ls": "-",  "sup": True},
    {"name": "XGBoost",             "prob_file": "xgb_y_prob_v1.npy", "color": "#d7191c", "ls": "-",  "sup": True},
    {"name": "Isolation Forest",    "prob_file": "if_y_prob_v1.npy",  "color": "#fdae61", "ls": "--", "sup": False},
    {"name": "Autoencoder",         "prob_file": "ae_y_prob_v1.npy",  "color": "#2c7bb6", "ls": "--", "sup": False},
]


def _load_probs(version: int) -> tuple[np.ndarray, list[dict]]:
    """Load y_test and all available model probability arrays."""
    y_test = np.load(MODELS_DIR / f"y_test_v{version}.npy")
    loaded = []
    for cfg in MODEL_CONFIGS:
        p = MODELS_DIR / cfg["prob_file"]
        if p.exists():
            loaded.append({**cfg, "y_prob": np.load(p)})
        else:
            log.warning(f"Skipping {cfg['name']}: not found")
    return y_test, loaded


def plot_roc_curves(version: int = 1) -> None:
    """Fig 01: ROC curves for all models with AUC annotations."""
    y_test, models = _load_probs(version)
    fig, ax = plt.subplots(figsize=(7, 6))

    for m in models:
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=m["color"], linestyle=m["ls"],
                linewidth=2.0, label=f"{m['name']} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)

    path = FIG_DIR / "01_roc_curves.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_pr_curves(version: int = 1) -> None:
    """
    Fig 02: Precision-Recall curves.
    PR curves are the primary evaluation for heavily imbalanced datasets.
    The no-skill baseline is the fraud prevalence rate (not 0.5).
    """
    y_test, models = _load_probs(version)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"No-skill baseline (prevalence={baseline:.4f})")

    for m in models:
        prec, rec, _ = precision_recall_curve(y_test, m["y_prob"])
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=m["color"], linestyle=m["ls"],
                linewidth=2.0, label=f"{m['name']} (AUPRC={ap:.3f})")
        # Mark theta* point
        theta_star, _ = optimal_threshold_cost(y_test, m["y_prob"])
        y_pred_star = (m["y_prob"] >= theta_star).astype(int)
        from sklearn.metrics import precision_score as ps, recall_score as rs
        p_star = ps(y_test, y_pred_star, zero_division=0)
        r_star = rs(y_test, y_pred_star, zero_division=0)
        ax.scatter(r_star, p_star, color=m["color"], s=80, zorder=5, marker="*")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models\n(★ marks operating point at θ*)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)

    path = FIG_DIR / "02_pr_curves.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_confusion_matrices(version: int = 1) -> None:
    """Fig 03: Confusion matrices at theta* for all models (2x2 grid)."""
    y_test, models = _load_probs(version)
    n = len(models)
    cols = 2; rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4.5))
    axes = axes.flatten()

    for ax, m in zip(axes, models):
        theta_star, _ = optimal_threshold_cost(y_test, m["y_prob"])
        y_pred = (m["y_prob"] >= theta_star).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{m['name']}\n(θ*={theta_star:.3f})", fontsize=10)

    # Hide empty axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices at Optimal Cost Threshold θ*", fontsize=13, y=1.01)
    path = FIG_DIR / "03_confusion_matrices.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_calibration_curves(version: int = 1) -> None:
    """
    Fig 04: Reliability diagrams for supervised models.
    A well-calibrated model has predicted probabilities that match empirical
    fraud rates. Calibration is essential for threshold-based decisions.
    """
    y_test, models = _load_probs(version)
    sup_models = [m for m in models if m["sup"]]

    fig, axes = plt.subplots(1, len(sup_models), figsize=(5 * len(sup_models), 5))
    if len(sup_models) == 1:
        axes = [axes]

    for ax, m in zip(axes, sup_models):
        prob_true, prob_pred = calibration_curve(y_test, m["y_prob"], n_bins=15, strategy="quantile")
        ax.plot(prob_pred, prob_true, color=m["color"], linewidth=2.0, marker="o", markersize=4,
                label="Model calibration")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (Empirical)")
        ax.set_title(f"{m['name']}\nReliability Diagram")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Calibration Curves (Reliability Diagrams)", fontsize=13)
    path = FIG_DIR / "04_calibration_curves.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_threshold_sweep(version: int = 1) -> None:
    """
    Fig 05: Precision / Recall / F1 vs threshold for XGBoost.
    Illustrates why theta=0.5 is not the correct operating point.
    """
    y_test, models = _load_probs(version)
    xgb_model = next((m for m in models if "XGBoost" in m["name"]), None)
    if xgb_model is None:
        log.warning("XGBoost predictions not found, skipping threshold sweep")
        return

    y_prob = xgb_model["y_prob"]
    thresholds = np.linspace(0.001, 0.999, 300)
    precisions, recalls, f1s = [], [], []

    from sklearn.metrics import precision_score as ps, recall_score as rs, f1_score as f1s_fn
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(ps(y_test, y_pred, zero_division=0))
        recalls.append(rs(y_test, y_pred, zero_division=0))
        f1s.append(f1s_fn(y_test, y_pred, zero_division=0))

    theta_star, _ = optimal_threshold_cost(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, color="#1a9641", linewidth=1.8, label="Precision")
    ax.plot(thresholds, recalls,    color="#d7191c", linewidth=1.8, label="Recall")
    ax.plot(thresholds, f1s,        color="#2c7bb6", linewidth=1.8, label="F1")
    ax.axvline(0.5,        color="gray",   linestyle=":", linewidth=1.2, label="θ=0.5 (naive)")
    ax.axvline(theta_star, color="black",  linestyle="--", linewidth=1.5, label=f"θ*={theta_star:.3f} (cost-optimal)")

    ax.set_xlabel("Decision Threshold (θ)")
    ax.set_ylabel("Score")
    ax.set_title("XGBoost: Precision / Recall / F1 vs Decision Threshold")
    ax.legend(loc="center right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    path = FIG_DIR / "05_threshold_sweep.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def generate_all_plots(version: int = 1) -> None:
    """Generate all evaluation plots in sequence."""
    log.info("Generating all evaluation plots...")
    plot_roc_curves(version)
    plot_pr_curves(version)
    plot_confusion_matrices(version)
    plot_calibration_curves(version)
    plot_threshold_sweep(version)
    log.info(f"All plots saved to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    generate_all_plots(version=args.version)
