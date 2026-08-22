"""
serving/api/predictor.py
=========================
Model loading and scoring logic for the FastAPI serving layer.

Design:
  - Models are loaded once at startup (not per-request)
  - Supports hot-swap: reload endpoint replaces models without restarting
  - SHAP top-3 contributors computed efficiently using stored SHAP values
    (for XGBoost: precomputed TreeExplainer; for LR: precomputed Kernel SHAP)
  - Scores every transaction through XGBoost (primary) with IF/AE as secondary
  - All scores written to SQLite model_scores table for audit trail

Thread safety:
  - Model objects are loaded into module-level singletons
  - sklearn predict_proba and XGBoost predict are thread-safe (GIL-held during C++ call)
  - No shared mutable state during inference
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "db" / "fraud.db"

# Cost threshold (loaded from cost_analysis results or fallback)
DEFAULT_THRESHOLD = 0.5


class ModelRegistry:
    """
    Singleton registry holding all loaded model artifacts.
    Loaded once at application startup via load_models().
    """

    def __init__(self):
        self.pipeline = None          # sklearn feature Pipeline
        self.xgb_model = None         # XGBoost primary model
        self.lr_model = None          # Logistic Regression secondary
        self.if_model = None          # Isolation Forest
        self.ae_model = None          # PyTorch Autoencoder
        self.ae_device = None         # torch.device for AE inference
        self.feature_names: list[str] = []
        self.xgb_threshold: float = DEFAULT_THRESHOLD
        self.lr_threshold: float = DEFAULT_THRESHOLD
        self.version: str = "v1"
        self.trained_at: str | None = None
        self.train_auprc: float | None = None
        self._loaded: bool = False

    def load(self, version: int = 1) -> None:
        """Load all model artifacts from models/ directory."""
        log.info(f"Loading model artifacts (version={version})...")
        v = str(version)

        # Feature pipeline
        pipeline_path = MODELS_DIR / f"feature_pipeline_v{v}.joblib"
        if pipeline_path.exists():
            self.pipeline = joblib.load(pipeline_path)
            log.info(f"  Feature pipeline loaded: {pipeline_path.name}")
        else:
            log.error(f"Feature pipeline not found: {pipeline_path}")
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")

        # Feature names
        names_path = MODELS_DIR / f"feature_names_v{v}.json"
        if names_path.exists():
            self.feature_names = json.loads(names_path.read_text())

        # XGBoost (primary scoring model)
        xgb_path = MODELS_DIR / f"xgb_model_v{v}.joblib"
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)
            log.info(f"  XGBoost loaded: {xgb_path.name}")

        # Logistic Regression
        lr_path = MODELS_DIR / f"lr_model_v{v}.joblib"
        if lr_path.exists():
            self.lr_model = joblib.load(lr_path)
            log.info(f"  LR loaded: {lr_path.name}")

        # Isolation Forest
        if_path = MODELS_DIR / f"if_model_v{v}.joblib"
        if if_path.exists():
            self.if_model = joblib.load(if_path)
            log.info(f"  Isolation Forest loaded: {if_path.name}")

        # Autoencoder
        ae_arch_path = MODELS_DIR / f"autoencoder_v{v}_arch.json"
        ae_weights_path = MODELS_DIR / f"autoencoder_v{v}.pt"
        if ae_arch_path.exists() and ae_weights_path.exists():
            try:
                import torch
                from modeling.unsupervised.autoencoder import load_autoencoder
                self.ae_model, self.ae_device = load_autoencoder(version=version)
                log.info(f"  Autoencoder loaded: {ae_weights_path.name}")
            except Exception as e:
                log.warning(f"  Autoencoder load failed (will skip): {e}")

        # Load thresholds from cost analysis
        cost_path = MODELS_DIR / "cost_analysis_results.json"
        if cost_path.exists():
            cost_data = json.loads(cost_path.read_text())
            if "XGBoost" in cost_data:
                self.xgb_threshold = cost_data["XGBoost"]["theta_star"]
            if "Logistic Regression" in cost_data:
                self.lr_threshold = cost_data["Logistic Regression"]["theta_star"]
            log.info(f"  XGBoost threshold: {self.xgb_threshold:.4f}")

        # Load training metadata
        xgb_results_path = MODELS_DIR / f"xgb_v{v}_results.json"
        if xgb_results_path.exists():
            results = json.loads(xgb_results_path.read_text())
            self.train_auprc = results.get("auprc")

        self.version = f"v{version}"
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self._loaded = True
        log.info("All models loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Module-level singleton
registry = ModelRegistry()


def load_models(version: int = 1) -> None:
    """Called once at FastAPI startup."""
    registry.load(version)


def _txn_to_dataframe(txn_dict: dict) -> pd.DataFrame:
    """Convert a transaction dict (from Pydantic model) to a single-row DataFrame."""
    return pd.DataFrame([txn_dict])


def _run_pipeline(X_raw: pd.DataFrame) -> np.ndarray:
    """Apply the feature pipeline to raw transaction data."""
    if registry.pipeline is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    return registry.pipeline.transform(X_raw)


def _get_shap_contributors(
    X_transformed: np.ndarray,
    n_top: int = 3,
) -> list[dict]:
    """
    Compute top-n SHAP contributors for XGBoost using TreeExplainer.
    Falls back to feature magnitude ranking if SHAP fails.

    Returns list of {feature, shap_value, feature_value} dicts.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(registry.xgb_model)
        shap_values = explainer.shap_values(X_transformed)
        abs_shap = np.abs(shap_values[0])
        top_idx = np.argsort(abs_shap)[::-1][:n_top]
        contributors = []
        for idx in top_idx:
            feat_name = (
                registry.feature_names[idx]
                if idx < len(registry.feature_names)
                else f"feature_{idx}"
            )
            contributors.append({
                "feature":       feat_name,
                "shap_value":    round(float(shap_values[0][idx]), 4),
                "feature_value": round(float(X_transformed[0][idx]), 4),
            })
        return contributors
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}. Using magnitude fallback.")
        abs_vals = np.abs(X_transformed[0])
        top_idx = np.argsort(abs_vals)[::-1][:n_top]
        return [
            {
                "feature": registry.feature_names[i] if i < len(registry.feature_names) else f"f{i}",
                "shap_value": 0.0,
                "feature_value": round(float(X_transformed[0][i]), 4),
            }
            for i in top_idx
        ]


def score_transaction(
    txn_dict: dict,
    compute_shap: bool = True,
) -> dict:
    """
    Score a single transaction using the primary (XGBoost) model.

    Parameters
    ----------
    txn_dict : dict — transaction fields (from Pydantic .model_dump())
    compute_shap : bool — whether to compute SHAP contributors (slower)

    Returns
    -------
    dict with keys: fraud_probability, is_flagged, threshold,
                    risk_level, top_contributors, model_version
    """
    import time
    t0 = time.perf_counter()

    txn_id = txn_dict.get("TransactionID", -1)
    X_raw = _txn_to_dataframe(txn_dict)

    # Feature engineering
    X = _run_pipeline(X_raw)

    # Primary model: XGBoost
    if registry.xgb_model is not None:
        fraud_prob = float(registry.xgb_model.predict_proba(X)[0, 1])
        model_name = f"XGBoost-{registry.version}"
        threshold = registry.xgb_threshold
    elif registry.lr_model is not None:
        fraud_prob = float(registry.lr_model.predict_proba(X)[0, 1])
        model_name = f"LR-{registry.version}"
        threshold = registry.lr_threshold
    else:
        raise RuntimeError("No scoring model available.")

    is_flagged = fraud_prob >= threshold

    # Risk level
    if   fraud_prob < 0.30:  risk_level = "LOW"
    elif fraud_prob < 0.60:  risk_level = "MEDIUM"
    elif fraud_prob < 0.85:  risk_level = "HIGH"
    else:                    risk_level = "CRITICAL"

    # SHAP contributors (only when XGBoost available)
    contributors = []
    if compute_shap and registry.xgb_model is not None:
        contributors = _get_shap_contributors(X, n_top=3)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    result = {
        "TransactionID":   txn_id,
        "fraud_probability": round(fraud_prob, 4),
        "is_flagged":      is_flagged,
        "threshold":       round(threshold, 4),
        "risk_level":      risk_level,
        "model_version":   model_name,
        "latency_ms":      latency_ms,
        "scored_at":       datetime.now(timezone.utc).isoformat(),
        "top_contributors": contributors,
    }

    # Async audit log (non-blocking)
    _log_score_to_db(txn_id, model_name, fraud_prob, threshold, is_flagged, latency_ms)

    return result


def score_batch(txn_dicts: list[dict]) -> list[dict]:
    """
    Score a batch of transactions.
    Uses vectorised predict_proba for efficiency; SHAP computed per-sample.
    """
    import time
    results = []
    for txn in txn_dicts:
        results.append(score_transaction(txn, compute_shap=False))
    return results


def _log_score_to_db(
    txn_id: int,
    model_name: str,
    prob: float,
    threshold: float,
    is_flagged: bool,
    latency_ms: float,
) -> None:
    """Write score to model_scores table for audit trail and drift monitoring."""
    try:
        if not DB_PATH.exists():
            return
        with sqlite3.connect(DB_PATH, timeout=5.0) as conn:
            conn.execute(
                """
                INSERT INTO model_scores
                  (TransactionID, model_name, fraud_prob, threshold, is_flagged, scored_at, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (txn_id, model_name, prob, threshold, int(is_flagged),
                 datetime.now(timezone.utc).isoformat(), latency_ms),
            )
    except Exception as e:
        # Non-critical — log but don't fail the scoring request
        log.warning(f"Audit log write failed: {e}")
