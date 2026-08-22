"""
serving/api/main.py
====================
FastAPI application — the central serving layer.

Endpoints:
  POST /score          - Score a single transaction
  POST /score/batch    - Score 1-1000 transactions
  GET  /health         - Health check (liveness probe)
  GET  /metrics        - Prometheus-format metrics
  GET  /model/info     - Model metadata

CORS, structured logging, and request ID injection are configured
via middleware. All scoring is delegated to predictor.py.

Run locally:
    uvicorn serving.api.main:app --host 0.0.0.0 --port 8000 --reload

Production (Docker):
    gunicorn serving.api.main:app -k uvicorn.workers.UvicornWorker \
             -w 2 --bind 0.0.0.0:8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from serving.api.middleware import (
    AccessLogMiddleware,
    MetricsMiddleware,
    RequestIDMiddleware,
    metrics,
)
from serving.api.predictor import load_models, registry, score_batch, score_transaction
from serving.api.schema import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ModelInfoResponse,
    ScoreResponse,
    TransactionFeatures,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("api.main")

_START_TIME = time.perf_counter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    log.info("Starting up: loading model artifacts...")
    try:
        load_models(version=1)
        log.info("Models loaded. API ready.")
    except FileNotFoundError as e:
        log.error(f"Model loading failed: {e}")
        log.warning("API starting without models — /score endpoints will return 503")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Fraud & Anomaly Risk-Scoring API",
    description=(
        "Real-time transaction fraud scoring using XGBoost (primary) with "
        "Isolation Forest and Autoencoder (secondary anomaly detectors). "
        "Threshold selected via Expected Cost minimisation (not 0.5)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# --- Middleware (order matters: outermost wraps first) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/score",
    response_model=ScoreResponse,
    summary="Score a single transaction",
    description=(
        "Accepts a single transaction and returns a fraud probability score, "
        "flag decision, risk level, and top-3 SHAP contributors."
    ),
    tags=["Scoring"],
)
async def score_single(transaction: TransactionFeatures) -> ScoreResponse:
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Retry shortly.")
    try:
        txn_dict = transaction.model_dump()
        result = score_transaction(txn_dict, compute_shap=True)
        metrics.record_score(result["is_flagged"])
        return ScoreResponse(**result)
    except Exception as e:
        log.exception(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post(
    "/score/batch",
    response_model=BatchScoreResponse,
    summary="Score a batch of transactions (up to 1,000)",
    description=(
        "Accepts 1-1,000 transactions and returns scores for all. "
        "SHAP contributors are not computed for batch requests (latency). "
        "Use the single /score endpoint for SHAP explanations."
    ),
    tags=["Scoring"],
)
async def score_batch_endpoint(request: BatchScoreRequest) -> BatchScoreResponse:
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    t0 = time.perf_counter()
    try:
        txn_dicts = [t.model_dump() for t in request.transactions]
        results_raw = score_batch(txn_dicts)
        results = [ScoreResponse(**r) for r in results_raw]
        total_latency = round((time.perf_counter() - t0) * 1000, 2)
        n_flagged = sum(1 for r in results if r.is_flagged)

        for r in results:
            metrics.record_score(r.is_flagged)

        return BatchScoreResponse(
            results=results,
            batch_size=len(results),
            total_latency_ms=total_latency,
            n_flagged=n_flagged,
            flag_rate=round(n_flagged / len(results), 4),
        )
    except Exception as e:
        log.exception(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check (liveness probe)",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if registry.is_loaded else "degraded",
        model_version=registry.version,
        uptime_seconds=round(time.perf_counter() - _START_TIME, 1),
        threshold=registry.xgb_threshold,
    )


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus-format metrics",
    tags=["Operations"],
)
async def prometheus_metrics() -> str:
    return metrics.to_prometheus()


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    tags=["Operations"],
)
async def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name="XGBoost",
        model_version=registry.version,
        trained_at=registry.trained_at,
        threshold=registry.xgb_threshold,
        train_auprc=registry.train_auprc,
        n_features=len(registry.feature_names) if registry.feature_names else None,
    )


if __name__ == "__main__":
    uvicorn.run(
        "serving.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
