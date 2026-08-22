"""
modeling/cost_analysis.py
==========================
Dollar-cost curve analysis and optimal decision threshold selection.

This is the academic centrepiece of the project.

Theory
------
For a binary classifier with decision threshold theta, the expected cost per
transaction is:

    E[Cost](theta) = FP(theta) * c_fp + FN(theta) * c_fn

where:
  FP(theta) = number of false positives at threshold theta
  FN(theta) = number of false negatives at threshold theta
  c_fp      = cost of one false positive (analyst review time + customer friction)
  c_fn      = cost of one false negative (median fraud loss)

The optimal threshold is:
    theta* = argmin_theta E[Cost](theta)

This is NOT the same as the F1-maximising threshold or the arbitrary 0.5 cutoff.

Cost parameters (documented in report Section 4.5):
  c_fp = .00   (30 minutes analyst review at /hr industry average)
  c_fn = .00  (median IEEE-CIS fraud transaction amount, computed from EDA)

Sensitivity analysis:
  We sweep the c_fp/c_fn ratio over 3 orders of magnitude to show how
  theta* changes as the relative cost of false alarms vs. missed fraud shifts.
  This demonstrates the model's decision-theoretic flexibility.

Outputs:
  report/figures/models/cost_curve_all_models.pdf
  report/figures/models/cost_sensitivity.pdf
  models/cost_analysis_results.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODEL_FIG_DIR = ROOT / "report" / "figures" / "models"
MODEL_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cost parameters
C_FP_BASE = 12.0    # USD — false positive cost
C_FN_BASE = 850.0   # USD — false negative cost

MODEL_CONFIGS = {
    "Logistic Regression": {
        "prob_file": "lr_y_prob_v1.npy",
        "color": "#1a9641",
        "linestyle": "-",
    },
    "XGBoost": {
        "prob_file": "xgb_y_prob_v1.npy",
        "color": "#d7191c",
        "linestyle": "-",
    },
    "Isolation Forest": {
        "prob_file": "if_y_prob_v1.npy",
        "color": "#fdae61",
        "linestyle": "--",
    },
    "Autoencoder": {
        "prob_file": "ae_y_prob_v1.npy",
        "color": "#2c7bb6",
        "linestyle": "--",
    },
}

N_THRESHOLDS = 500


def expected_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fp: float = C_FP_BASE,
    c_fn: float = C_FN_BASE,
    n_thresholds: int = N_THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute expected cost curve over all thresholds.

    Returns
    -------
    thresholds  : np.ndarray shape (n,)
    costs       : np.ndarray shape (n,) - expected cost per transaction
    theta_star  : float - threshold minimising expected cost
    min_cost    : float - minimum expected cost per transaction
    """
    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    costs = np.zeros(n_thresholds)
    n = len(y_true)

    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        costs[i] = (fp * c_fp + fn * c_fn) / n

    best_idx = np.argmin(costs)
    return thresholds, costs, float(thresholds[best_idx]), float(costs[best_idx])


def run_cost_analysis(version: int = 1) -> dict:
    """
    Generate all cost curve plots and compute optimal thresholds for all models.
    """
    log.info("=== Cost Analysis ===")

    y_test = np.load(MODELS_DIR / f"y_test_v{version}.npy")
    results = {}

    # --- Fig 1: Expected Cost Curve — all models on one axes ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, cfg in MODEL_CONFIGS.items():
        prob_path = MODELS_DIR / cfg["prob_file"]
        if not prob_path.exists():
            log.warning(f"Skipping {model_name}: {prob_path} not found")
            continue

        y_prob = np.load(prob_path)
        thresholds, costs, theta_star, min_cost = expected_cost_curve(
            y_test, y_prob, c_fp=C_FP_BASE, c_fn=C_FN_BASE
        )

        ax.plot(
            thresholds, costs,
            label=f"{model_name}",
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.0,
        )
        ax.axvline(
            theta_star,
            color=cfg["color"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
        )
        ax.annotate(
            f"  θ*={theta_star:.2f}",
            xy=(theta_star, min_cost),
            fontsize=8,
            color=cfg["color"],
            va="bottom",
        )

        results[model_name] = {
            "theta_star": round(theta_star, 4),
            "min_cost_per_txn": round(min_cost, 4),
            "c_fp": C_FP_BASE,
            "c_fn": C_FN_BASE,
        }
        log.info(f"  {model_name}: theta*={theta_star:.4f}, cost/txn=")

    ax.set_xlabel("Decision Threshold (θ)")
    ax.set_ylabel("Expected Cost per Transaction (USD)")
    ax.set_title(
        f"Expected Cost Curve  |  {{FP}}$=,  {{FN}}$="
    )
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)

    # Shade the minimum-cost region
    ax.axhspan(0, min(r["min_cost_per_txn"] for r in results.values()) * 1.5,
               alpha=0.04, color="green", label="_nolegend_")

    path1 = MODEL_FIG_DIR / "cost_curve_all_models.pdf"
    fig.savefig(path1, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Cost curve saved -> {path1}")

    # --- Fig 2: Sensitivity Analysis — theta* vs c_fp/c_fn ratio ---
    _plot_sensitivity_analysis(y_test, version)

    # --- Fig 3: Cost components breakdown at theta* for best model ---
    _plot_cost_breakdown(y_test, version)

    # Save results
    out_path = MODELS_DIR / "cost_analysis_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Cost analysis results saved -> {out_path}")

    return results


def _plot_sensitivity_analysis(y_test: np.ndarray, version: int) -> None:
    """
    Fig 2: How does theta* shift as the c_fp/c_fn ratio changes?

    Sweeps c_fp/c_fn from 0.001 to 10 on a log scale.
    Shows theta* for XGBoost (best supervised model) and Autoencoder.
    """
    log.info("Computing sensitivity analysis...")
    ratios = np.logspace(-3, 1, 200)  # c_fp/c_fn from 0.001 to 10

    model_subset = {
        k: v for k, v in MODEL_CONFIGS.items()
        if k in ("XGBoost", "Autoencoder", "Logistic Regression")
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, cfg in model_subset.items():
        prob_path = MODELS_DIR / cfg["prob_file"]
        if not prob_path.exists():
            continue
        y_prob = np.load(prob_path)

        theta_stars = []
        for ratio in ratios:
            c_fp = ratio * C_FN_BASE  # keep c_fn fixed at 
            _, _, theta_star, _ = expected_cost_curve(
                y_test, y_prob, c_fp=c_fp, c_fn=C_FN_BASE
            )
            theta_stars.append(theta_star)

        ax.semilogx(
            ratios, theta_stars,
            label=model_name,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.0,
        )

    ax.axvline(C_FP_BASE / C_FN_BASE, color="black", linestyle=":", linewidth=1.5,
               label=f"Baseline ratio ({C_FP_BASE}/{C_FN_BASE:.0f}={C_FP_BASE/C_FN_BASE:.3f})")
    ax.set_xlabel("{FP} / c_{FN}$ ratio (log scale)")
    ax.set_ylabel("Optimal Threshold θ*")
    ax.set_title("Sensitivity Analysis: θ* vs. Cost Ratio")
    ax.legend()
    ax.set_ylim(0, 1)

    path = MODEL_FIG_DIR / "cost_sensitivity.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Sensitivity analysis saved -> {path}")


def _plot_cost_breakdown(y_test: np.ndarray, version: int) -> None:
    """
    Fig 3: Stacked bar chart of FP cost vs FN cost at theta* for each model.
    Illustrates the precision-recall trade-off in dollar terms.
    """
    log.info("Computing cost breakdown plot...")
    model_names, fp_costs, fn_costs = [], [], []

    for model_name, cfg in MODEL_CONFIGS.items():
        prob_path = MODELS_DIR / cfg["prob_file"]
        if not prob_path.exists():
            continue
        y_prob = np.load(prob_path)
        _, _, theta_star, _ = expected_cost_curve(y_test, y_prob)
        y_pred = (y_prob >= theta_star).astype(int)
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        n = len(y_test)
        model_names.append(model_name)
        fp_costs.append(fp * C_FP_BASE / n)
        fn_costs.append(fn * C_FN_BASE / n)

    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_fp = ax.bar(x, fp_costs, label=f"FP Cost ({{FP}}$=)", color="#fdae61")
    bars_fn = ax.bar(x, fn_costs, bottom=fp_costs,
                     label=f"FN Cost ({{FN}}$=)", color="#d7191c")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Expected Cost per Transaction (USD)")
    ax.set_title("Cost Breakdown at θ* — FP vs FN Component")
    ax.legend()

    # Annotate total cost
    for i, (fp_c, fn_c) in enumerate(zip(fp_costs, fn_costs)):
        total = fp_c + fn_c
        ax.text(i, total + 0.002, f"", ha="center", va="bottom", fontsize=8.5)

    path = MODEL_FIG_DIR / "cost_breakdown.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Cost breakdown saved -> {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    results = run_cost_analysis(version=args.version)
    print(json.dumps(results, indent=2))
