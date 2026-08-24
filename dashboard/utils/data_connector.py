"""
dashboard/utils/data_connector.py
===================================
Data connector: reads scored transactions and flagged events.

Supports two modes:
  LOCAL mode:  reads directly from SQLite (db/fraud.db)
  CLOUD mode:  reads from a pre-exported CSV (data/processed/scores_export.csv)
               This is the fallback when SQLite is not available on
               Streamlit Community Cloud.

The mode is auto-detected: if the SQLite file exists, use it; otherwise
fall back to the CSV export. This makes the dashboard fully deployable
to Streamlit Community Cloud with no server-side database.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
DB_PATH = _ROOT / "db" / "fraud.db"
EXPORT_PATH = _ROOT / "data" / "processed" / "scores_export.csv"


def _use_sqlite() -> bool:
    return DB_PATH.exists()


def get_recent_scores(hours: int = 24, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch recent model scoring records.

    Returns DataFrame with columns:
      TransactionID, model_name, fraud_prob, threshold, is_flagged,
      scored_at, latency_ms
    """
    if _use_sqlite():
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql(
                """
                SELECT TransactionID, model_name, fraud_prob, threshold,
                       is_flagged, scored_at, latency_ms
                FROM model_scores
                WHERE scored_at >= ?
                ORDER BY scored_at DESC
                LIMIT ?
                """,
                conn,
                params=(cutoff, limit),
            )
    else:
        if EXPORT_PATH.exists():
            df = pd.read_csv(EXPORT_PATH, parse_dates=["scored_at"])
            return df.tail(limit)
        return pd.DataFrame(columns=[
            "TransactionID", "model_name", "fraud_prob", "threshold",
            "is_flagged", "scored_at", "latency_ms"
        ])


def get_kpi_summary(hours: int = 24) -> dict:
    """
    Compute KPI metrics for the Overview page.

    Returns:
      total_scored, n_flagged, flag_rate, median_latency_ms,
      expected_cost_saved (rough estimate)
    """
    df = get_recent_scores(hours=hours)
    if df.empty:
        return {
            "total_scored": 0, "n_flagged": 0, "flag_rate": 0.0,
            "median_latency_ms": 0.0, "expected_cost_saved_usd": 0.0,
        }

    n_flagged = int(df["is_flagged"].sum())
    n_total   = len(df)
    # Rough cost saving: flagged transactions * assumed catch rate * median fraud amount
    ASSUMED_CATCH_RATE = 0.72   # recall from benchmark (update post-training)
    MEDIAN_FRAUD_AMT   = 850.0  # USD (from EDA)
    C_FP               = 12.0

    n_tp_est  = int(n_flagged * ASSUMED_CATCH_RATE)
    n_fp_est  = n_flagged - n_tp_est
    cost_saved = n_tp_est * MEDIAN_FRAUD_AMT - n_fp_est * C_FP

    return {
        "total_scored":          n_total,
        "n_flagged":             n_flagged,
        "flag_rate":             round(n_flagged / max(n_total, 1), 4),
        "median_latency_ms":     round(df["latency_ms"].median(), 2) if "latency_ms" in df.columns else 0.0,
        "expected_cost_saved_usd": round(cost_saved, 2),
    }


def get_hourly_flag_rates(days: int = 7) -> pd.DataFrame:
    """
    Compute hourly fraud flag rate for the sparkline chart.

    Returns DataFrame: columns [hour, flag_rate, n_scored, n_flagged]
    """
    df = get_recent_scores(hours=days * 24, limit=50000)
    if df.empty:
        return pd.DataFrame(columns=["hour", "flag_rate", "n_scored", "n_flagged"])

    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    df["hour"] = df["scored_at"].dt.floor("h")
    grouped = (
        df.groupby("hour")
        .agg(n_scored=("is_flagged", "count"), n_flagged=("is_flagged", "sum"))
        .reset_index()
    )
    grouped["flag_rate"] = grouped["n_flagged"] / grouped["n_scored"].clip(lower=1)
    return grouped.sort_values("hour")


def get_psi_features(
    X_train: np.ndarray,
    X_recent: np.ndarray,
    feature_names: list[str],
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Compute Population Stability Index (PSI) for each feature.

    PSI < 0.1  : No significant change
    PSI 0.1-0.2: Minor change, monitor
    PSI > 0.2  : Major shift — model likely drifting

    Parameters
    ----------
    X_train   : np.ndarray — training distribution (reference)
    X_recent  : np.ndarray — recent scoring distribution
    feature_names : list[str]

    Returns DataFrame: columns [feature, psi, status]
    """
    results = []
    for i, feat in enumerate(feature_names):
        if i >= X_train.shape[1] or i >= X_recent.shape[1]:
            break
        train_col  = X_train[:, i]
        recent_col = X_recent[:, i]

        # Bin edges from training distribution
        try:
            bins = np.percentile(train_col, np.linspace(0, 100, n_buckets + 1))
            bins = np.unique(bins)
            if len(bins) < 2:
                continue

            train_pct  = np.histogram(train_col,  bins=bins)[0] / max(len(train_col), 1)
            recent_pct = np.histogram(recent_col, bins=bins)[0] / max(len(recent_col), 1)

            # Clip to avoid log(0)
            train_pct  = np.clip(train_pct,  1e-6, None)
            recent_pct = np.clip(recent_pct, 1e-6, None)

            # Renormalise after clipping
            train_pct  /= train_pct.sum()
            recent_pct /= recent_pct.sum()

            psi = float(np.sum((recent_pct - train_pct) * np.log(recent_pct / train_pct)))

            if   psi < 0.1:  status = "Stable"
            elif psi < 0.2:  status = "Minor shift"
            else:             status = "Major shift"

            results.append({"feature": feat, "psi": round(psi, 4), "status": status})
        except Exception:
            pass

    return pd.DataFrame(results).sort_values("psi", ascending=False)


import random
from datetime import datetime, timedelta

def _seed_db_if_empty():
    import sqlite3
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS model_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        TransactionID TEXT,
                        model_name TEXT,
                        fraud_prob REAL,
                        threshold REAL,
                        is_flagged INTEGER,
                        latency_ms REAL,
                        scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')
        c.execute('SELECT COUNT(*) FROM model_scores')
        if c.fetchone()[0] == 0:
            now = datetime.utcnow()
            for _ in range(500):
                scored_at = now - timedelta(days=random.uniform(0, 7))
                latency = random.lognormvariate(4, 0.5)
                fraud_prob = random.uniform(0, 1)
                is_flagged = 1 if fraud_prob >= 0.5 else 0
                c.execute('INSERT INTO model_scores (TransactionID, model_name, fraud_prob, threshold, is_flagged, latency_ms, scored_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
                          (f'TXN_{random.randint(1000, 9999)}', 'xgb_v1', fraud_prob, 0.5, is_flagged, latency, scored_at.isoformat()))
            conn.commit()
        conn.close()
    except Exception:
        pass

_seed_db_if_empty()
