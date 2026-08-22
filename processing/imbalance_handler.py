"""
processing/imbalance_handler.py
================================
Rigorous comparison of five class-imbalance handling strategies.
Each is evaluated via 5-fold stratified cross-validation, using AUPRC
as the sole selection metric (never raw accuracy).

Strategies compared:
  1. None (raw imbalanced data baseline)
  2. Random undersampling of majority class
  3. SMOTE (Synthetic Minority Oversampling Technique)
  4. Borderline-SMOTE (focuses on boundary region)
  5. Class-weight adjustment (cost-sensitive, no resampling)

Output:
  - CV AUPRC table printed + saved to data/processed/imbalance_comparison.csv
  - Optimal strategy name saved to data/processed/best_imbalance_strategy.txt

Usage:
    python processing/imbalance_handler.py [--features PATH]
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "processed"

N_SPLITS = 5
RANDOM_STATE = 42


def load_features(features_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load engineered feature matrix."""
    df = pd.read_parquet(features_path)
    # Drop non-feature columns
    drop_cols = ["TransactionID", "isFraud", "loaded_at", "source"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    y = df["isFraud"].astype(int)
    X = df.drop(columns=drop_cols + ["isFraud"] if "isFraud" in df.columns else drop_cols)
    # Keep only numeric columns for baseline LR comparison
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    log.info(f"Feature matrix: {X.shape}, Fraud rate: {y.mean():.4f}")
    return X, y


def make_strategies(random_state: int = RANDOM_STATE) -> dict:
    """Define all five imbalance strategies as imblearn Pipeline steps."""
    lr = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    lr_balanced = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    return {
        "1_None (Baseline)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", lr),
        ]),
        "2_Random Undersample": ImbPipeline([
            ("under", RandomUnderSampler(random_state=random_state)),
            ("scaler", StandardScaler()),
            ("clf", lr),
        ]),
        "3_SMOTE": ImbPipeline([
            ("over", SMOTE(k_neighbors=5, random_state=random_state)),
            ("scaler", StandardScaler()),
            ("clf", lr),
        ]),
        "4_Borderline-SMOTE": ImbPipeline([
            ("over", BorderlineSMOTE(k_neighbors=5, random_state=random_state)),
            ("scaler", StandardScaler()),
            ("clf", lr),
        ]),
        "5_Class-Weight (cost-sensitive)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", lr_balanced),
        ]),
    }


def evaluate_strategies(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """5-fold stratified CV AUPRC for each strategy."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strategies = make_strategies(random_state)
    results = []

    for name, pipeline in strategies.items():
        fold_auprcs = []
        log.info(f"Evaluating: {name}")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            auprc = average_precision_score(y_val, y_prob)
            fold_auprcs.append(auprc)
            log.info(f"  Fold {fold_idx+1}/{n_splits}: AUPRC = {auprc:.4f}")

        results.append({
            "Strategy": name,
            "Mean AUPRC": np.mean(fold_auprcs),
            "Std AUPRC": np.std(fold_auprcs),
            "Min AUPRC": np.min(fold_auprcs),
            "Max AUPRC": np.max(fold_auprcs),
        })

    return pd.DataFrame(results).sort_values("Mean AUPRC", ascending=False)


def run_comparison(features_path: Path) -> str:
    """Run full comparison; return name of best strategy."""
    X, y = load_features(features_path)
    results_df = evaluate_strategies(X, y)

    log.info("\n" + "=" * 65)
    log.info("IMBALANCE STRATEGY COMPARISON RESULTS (5-Fold CV AUPRC)")
    log.info("=" * 65)
    log.info("\n" + results_df.to_string(index=False))
    log.info("=" * 65)

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "imbalance_comparison.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")

    best = results_df.iloc[0]["Strategy"]
    best_path = OUT_DIR / "best_imbalance_strategy.txt"
    best_path.write_text(best)
    log.info(f"Best strategy: {best} -> saved to {best_path}")

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "features.parquet",
    )
    args = parser.parse_args()
    run_comparison(args.features)
