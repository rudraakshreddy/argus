"""
processing/eda.py
=================
Full exploratory data analysis of the IEEE-CIS dataset.
All plots are saved as PDF (vector) to report/figures/eda/ for LaTeX inclusion.

Runs standalone or after init_db + load_ieee_cis.

Usage:
    python processing/eda.py [--db-path PATH]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- safe on server / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "report" / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Publication-quality matplotlib style ----
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})
PALETTE = {"Legitimate": "#2c7bb6", "Fraud": "#d7191c"}


def load_data(db_path: Path) -> pd.DataFrame:
    """Load merged transactions + labels from SQLite."""
    log.info(f"Loading data from {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT t.*, f.isFraud
            FROM transactions t
            JOIN fraud_labels f USING (TransactionID)
            """,
            conn,
        )
    log.info(f"Loaded {len(df):,} rows x {df.shape[1]} cols")
    return df


def plot_class_balance(df: pd.DataFrame) -> None:
    """Fig 1: Class imbalance bar chart with percentage labels."""
    counts = df["isFraud"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    total = counts.sum()
    pcts = counts / total * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[PALETTE[k] for k in counts.index],
        width=0.5,
        edgecolor="white",
    )
    for bar, pct in zip(bars, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Class Distribution (IEEE-CIS Dataset)")
    ax.set_ylabel("Transaction Count")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.set_xlabel("")
    path = FIG_DIR / "01_class_balance.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_amount_distribution(df: pd.DataFrame) -> None:
    """Fig 2: Transaction amount KDE by class on log scale."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in df.groupby("isFraud"):
        name = "Fraud" if label == 1 else "Legitimate"
        log_amt = np.log1p(group["TransactionAmt"].dropna())
        sns.kdeplot(log_amt, ax=ax, label=name, color=PALETTE[name], fill=True, alpha=0.3)
    ax.set_xlabel("log(1 + TransactionAmt)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Amount Distribution by Class")
    ax.legend()
    path = FIG_DIR / "02_amount_distribution.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_temporal_fraud_rate(df: pd.DataFrame) -> None:
    """Fig 3: Hour-of-day x day-of-week fraud rate heatmap."""
    # TransactionDT is seconds from a reference date; mod 86400 = time of day
    df = df.copy()
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["dow"] = (df["TransactionDT"] // 86400) % 7  # 0=Mon reference
    pivot = df.groupby(["dow", "hour"])["isFraud"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        fmt=".3f",
        linewidths=0.3,
        cbar_kws={"label": "Fraud Rate"},
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week (0=Mon)")
    ax.set_title("Hourly Fraud Rate by Day of Week")
    path = FIG_DIR / "03_temporal_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_missing_values(df: pd.DataFrame, top_n: int = 30) -> None:
    """Fig 4: Top-N features by missing rate."""
    null_rate = df.isnull().mean().sort_values(ascending=False).head(top_n)
    null_rate = null_rate[null_rate > 0]

    fig, ax = plt.subplots(figsize=(8, max(4, len(null_rate) * 0.25)))
    colors = ["#d7191c" if v > 0.5 else "#fdae61" if v > 0.2 else "#2c7bb6" for v in null_rate.values]
    ax.barh(null_rate.index[::-1], null_rate.values[::-1], color=colors[::-1])
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="50% threshold")
    ax.axvline(0.2, color="orange", linestyle="--", linewidth=1, label="20% threshold")
    ax.set_xlabel("Missing Rate")
    ax.set_title(f"Top {len(null_rate)} Features by Missing Value Rate")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend()
    path = FIG_DIR / "04_missing_values.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Fig 5: Correlation heatmap of numeric C/D features, conditioned on fraud class."""
    cd_cols = [c for c in df.columns if c.startswith(("C", "D")) and c[1:].isdigit()]
    cd_cols = cd_cols[:20]  # top 20 for readability

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label, title in zip(axes, [0, 1], ["Legitimate", "Fraud"]):
        subset = df[df["isFraud"] == label][cd_cols].dropna()
        corr = subset.corr()
        sns.heatmap(
            corr,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=True,
            yticklabels=True,
            linewidths=0.1,
        )
        ax.set_title(f"Correlation — {title}")
        ax.tick_params(labelsize=7)
    path = FIG_DIR / "05_correlation_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


def plot_top_discriminating_features(df: pd.DataFrame, top_n: int = 10) -> None:
    """Fig 6: ANOVA F-statistic — top discriminating numeric features."""
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=["TransactionID", "TransactionDT", "isFraud"], errors="ignore"
    )
    fraud_mask = df["isFraud"] == 1
    results = []
    for col in numeric.columns:
        valid = numeric[col].notna()
        fraud_vals = numeric.loc[fraud_mask & valid, col]
        legit_vals = numeric.loc[~fraud_mask & valid, col]
        if len(fraud_vals) < 10 or len(legit_vals) < 10:
            continue
        f_stat, _ = stats.f_oneway(fraud_vals, legit_vals)
        results.append({"feature": col, "f_stat": f_stat})

    results_df = pd.DataFrame(results).sort_values("f_stat", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(results_df["feature"][::-1], results_df["f_stat"][::-1], color="#2c7bb6")
    ax.set_xlabel("ANOVA F-Statistic (higher = more discriminating)")
    ax.set_title(f"Top {top_n} Discriminating Features")
    path = FIG_DIR / "06_top_features_anova.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")
    return results_df["feature"].tolist()


def print_summary_stats(df: pd.DataFrame) -> dict:
    """Print and return key EDA summary statistics for the report."""
    stats_dict = {
        "n_transactions": len(df),
        "n_fraud": int(df["isFraud"].sum()),
        "fraud_rate": float(df["isFraud"].mean()),
        "n_columns": df.shape[1],
        "n_numeric": df.select_dtypes(include=[np.number]).shape[1],
        "n_categorical": df.select_dtypes(include=["object"]).shape[1],
        "overall_null_rate": float(df.isnull().mean().mean()),
        "median_txn_amt": float(df["TransactionAmt"].median()),
        "median_fraud_amt": float(df.loc[df["isFraud"] == 1, "TransactionAmt"].median()),
        "median_legit_amt": float(df.loc[df["isFraud"] == 0, "TransactionAmt"].median()),
    }
    log.info("=" * 55)
    log.info("EDA SUMMARY")
    for k, v in stats_dict.items():
        if isinstance(v, float):
            log.info(f"  {k:<30} {v:.4f}")
        else:
            log.info(f"  {k:<30} {v:,}")
    log.info("=" * 55)
    return stats_dict


def run_eda(db_path: Path) -> None:
    df = load_data(db_path)
    print_summary_stats(df)
    plot_class_balance(df)
    plot_amount_distribution(df)
    plot_temporal_fraud_rate(df)
    plot_missing_values(df)
    plot_correlation_heatmap(df)
    plot_top_discriminating_features(df)
    log.info(f"All EDA figures saved to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        type=Path,
        default=ROOT / "db" / "fraud.db",
    )
    args = parser.parse_args()
    run_eda(args.db_path)
