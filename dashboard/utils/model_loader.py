"""
dashboard/utils/model_loader.py
================================
Self-contained model artifact loader for the Streamlit dashboard.

Critical design constraint:
  The dashboard MUST work on Streamlit Community Cloud without any
  dependency on the FastAPI server being alive. It loads model artifacts
  directly from the models/ directory.

  On Streamlit Cloud, model artifacts are committed to the repo (or
  stored in a public cloud bucket). The dashboard loads them directly.

  This means all prediction logic in the dashboard is independent of
  the FastAPI serving layer — they share the same model files but do
  not communicate over HTTP for core functionality.

Caching:
  @st.cache_resource loads models once per session (not per render).
  This is the correct Streamlit pattern for large model objects.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

log = logging.getLogger(__name__)

# Root is the project directory — works both locally and on Streamlit Cloud
# when the full repo is deployed
_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = _ROOT / "models"


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_models_dir() -> Path:
    return MODELS_DIR


def load_pipeline(version: int = 1):
    """Load the sklearn feature pipeline (cached by Streamlit)."""
    path = MODELS_DIR / f"feature_pipeline_v{version}.joblib"
    if not path.exists():
        log.warning(f"Pipeline not found: {path}")
        return None
    return joblib.load(path)


def load_xgb_model(version: int = 1):
    """Load the primary XGBoost model."""
    path = MODELS_DIR / f"xgb_model_v{version}.joblib"
    if not path.exists():
        log.warning(f"XGBoost model not found: {path}")
        return None
    return joblib.load(path)


def load_feature_names(version: int = 1) -> list[str]:
    """Load feature names list."""
    path = MODELS_DIR / f"feature_names_v{version}.json"
    result = _load_json(path)
    return result if isinstance(result, list) else []


def load_all_metrics() -> dict:
    """Load the full benchmark metrics dictionary."""
    path = MODELS_DIR / "all_metrics.json"
    result = _load_json(path)
    return result if isinstance(result, dict) else {}


def load_benchmark_summary() -> list[dict]:
    """Load the benchmark summary for the comparison table."""
    path = MODELS_DIR / "benchmark_summary.json"
    result = _load_json(path)
    return result if isinstance(result, list) else []


def load_xgb_threshold(version: int = 1) -> float:
    """Load the cost-optimal threshold for XGBoost."""
    path = MODELS_DIR / "cost_analysis_results.json"
    data = _load_json(path)
    if data and "XGBoost" in data:
        return data["XGBoost"]["theta_star"]
    # Fallback: read from results JSON
    results_path = MODELS_DIR / f"xgb_v{version}_results.json"
    data = _load_json(results_path)
    if data:
        return data.get("threshold", 0.5)
    return 0.5


def load_y_test_and_probs(version: int = 1) -> dict[str, np.ndarray]:
    """Load test labels and all model probability arrays."""
    data = {}
    y_test_path = MODELS_DIR / f"y_test_v{version}.npy"
    if y_test_path.exists():
        data["y_test"] = np.load(y_test_path)

    model_keys = {
        "LogisticRegression": f"lr_y_prob_v{version}.npy",
        "XGBoost":            f"xgb_y_prob_v{version}.npy",
        "IsolationForest":    f"if_y_prob_v{version}.npy",
        "Autoencoder":        f"ae_y_prob_v{version}.npy",
    }
    for model_name, fname in model_keys.items():
        p = MODELS_DIR / fname
        if p.exists():
            data[model_name] = np.load(p)

    return data


def load_shap_values(model: str = "xgb", version: int = 1) -> np.ndarray | None:
    """Load precomputed SHAP values."""
    path = MODELS_DIR / f"{model}_shap_v{version}.npy"
    if not path.exists():
        return None
    return np.load(path)


def load_wilcoxon_result() -> dict:
    """Load Wilcoxon test result."""
    path = MODELS_DIR / "wilcoxon_test.json"
    result = _load_json(path)
    return result if isinstance(result, dict) else {}
