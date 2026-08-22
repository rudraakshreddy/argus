"""
processing/pipeline.py
======================
Assembles the full sklearn-compatible feature Pipeline from custom transformers.

The Pipeline covers:
  1. LogAmountTransformer      - log1p(amount) + card-level z-score
  2. CyclicalTimeTransformer   - sin/cos(hour, DOW) + is_weekend
  3. MissingFlagTransformer    - binary indicators for cols > 5% null
  4. FrequencyEncoder          - card1, card4, ProductCD
  5. LeakFreeTargetEncoder     - email domains, card4 (k-fold cross-fit)
  6. VFeaturePCA               - V1-V339 -> 30 PCA components
  7. ColumnDropper             - remove raw ID / non-feature columns
  8. NumericOnlySelector       - select numeric columns, fix column order
  9. SimpleImputer             - median imputation for remaining NaN
 10. StandardScaler            - zero-mean, unit-variance final scaling

Saved to: models/feature_pipeline_v{version}.joblib

Usage:
    python processing/pipeline.py [--db-path PATH] [--version 1]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.feature_engineering import (
    CyclicalTimeTransformer,
    FrequencyEncoder,
    LeakFreeTargetEncoder,
    LogAmountTransformer,
    MissingFlagTransformer,
    VFeaturePCA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FREQ_COLS = ["card1", "card4", "ProductCD"]
TARGET_ENC_COLS = ["P_emaildomain", "R_emaildomain", "card4"]

_DROP_COLS = [
    "TransactionID", "TransactionDT", "isFraud", "loaded_at", "source",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "ProductCD",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
] + [f"id_{i:02d}" for i in range(12, 39)]


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop non-feature columns safely (ignores missing columns)."""
    def __init__(self, drop_cols: list = None):
        self.drop_cols = drop_cols or _DROP_COLS

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_to_drop_ = [c for c in self.drop_cols if c in X.columns]
        log.info(f"ColumnDropper: will drop {len(self.cols_to_drop_)} columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


class NumericOnlySelector(BaseEstimator, TransformerMixin):
    """Keep only numeric columns; fix column order for consistent transform."""
    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        log.info(f"NumericOnlySelector: keeping {len(self.numeric_cols_)} numeric columns")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_num = X.reindex(columns=self.numeric_cols_)
        return X_num.values


def build_feature_pipeline() -> Pipeline:
    """Construct the full feature engineering Pipeline (unfitted)."""
    return Pipeline([
        ("log_amount",     LogAmountTransformer()),
        ("cyclical_time",  CyclicalTimeTransformer()),
        ("missing_flags",  MissingFlagTransformer(threshold=0.05)),
        ("freq_encoder",   FrequencyEncoder(cols=FREQ_COLS)),
        ("target_encoder", LeakFreeTargetEncoder(cols=TARGET_ENC_COLS)),
        ("v_pca",          VFeaturePCA(n_components=30)),
        ("col_dropper",    ColumnDropper()),
        ("numeric_sel",    NumericOnlySelector()),
        ("imputer",        SimpleImputer(strategy="median")),
        ("scaler",         StandardScaler()),
    ])


def fit_and_save_pipeline(
    db_path: Path,
    version: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Fit the full feature pipeline on training data and save to disk."""
    log.info(f"Loading data from {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT t.*, f.isFraud FROM transactions t JOIN fraud_labels f USING(TransactionID)",
            conn,
        )
    log.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    log.info(
        f"Train: {len(X_train):,} (fraud={y_train.mean():.4f}) | "
        f"Test: {len(X_test):,} (fraud={y_test.mean():.4f})"
    )

    pipeline = build_feature_pipeline()
    log.info("Fitting feature pipeline on training data only...")
    X_tr = pipeline.fit_transform(X_train, y_train)
    X_te = pipeline.transform(X_test)
    log.info(f"Train shape: {X_tr.shape} | Test shape: {X_te.shape}")

    # Save artifacts
    joblib.dump(pipeline, MODELS_DIR / f"feature_pipeline_v{version}.joblib")
    np.save(MODELS_DIR / f"X_train_v{version}.npy", X_tr)
    np.save(MODELS_DIR / f"X_test_v{version}.npy",  X_te)
    np.save(MODELS_DIR / f"y_train_v{version}.npy", y_train.values)
    np.save(MODELS_DIR / f"y_test_v{version}.npy",  y_test.values)

    feature_names = pipeline.named_steps["numeric_sel"].numeric_cols_
    (MODELS_DIR / f"feature_names_v{version}.json").write_text(
        json.dumps(feature_names, indent=2)
    )
    log.info(f"All artifacts saved to {MODELS_DIR}")
    return pipeline, X_tr, X_te, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=ROOT / "db" / "fraud.db")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    fit_and_save_pipeline(args.db_path, args.version)
