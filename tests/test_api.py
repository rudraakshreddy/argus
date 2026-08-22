"""
tests/test_api.py
==================
Integration tests for the FastAPI scoring endpoints.

These tests use FastAPI's TestClient (httpx-based, no live server needed).
All model loading is mocked so tests run without trained artifacts.

Test coverage:
  - GET /health         returns 200 with correct schema
  - GET /model/info     returns 200 with correct schema
  - GET /metrics        returns 200 Prometheus text
  - POST /score         validates request schema (422 on bad input)
  - POST /score/batch   validates batch size limits
  - POST /score         returns correct response keys when model is loaded
  - Request ID header   injected in every response
  - Latency field       present in score response and > 0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Mock model registry so tests run without trained artifacts
# ---------------------------------------------------------------------------

def _make_mock_registry():
    """Create a mock ModelRegistry with a loaded XGBoost model."""
    reg = MagicMock()
    reg.is_loaded = True
    reg.version = "v1"
    reg.xgb_threshold = 0.42
    reg.lr_threshold = 0.38
    reg.trained_at = "2026-01-01T00:00:00Z"
    reg.train_auprc = 0.872
    reg.feature_names = [f"feat_{i}" for i in range(20)]

    # Mock model returns fixed probability
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
    reg.xgb_model = mock_model
    reg.lr_model = None
    reg.if_model = None
    reg.ae_model = None
    return reg


def _make_mock_pipeline():
    """Mock sklearn Pipeline that returns a fixed feature array."""
    pipeline = MagicMock()
    pipeline.transform.return_value = np.random.randn(1, 20)
    return pipeline


VALID_TXN = {
    "TransactionID":  123456,
    "TransactionDT":  86400,
    "TransactionAmt": 150.00,
    "ProductCD":      "W",
    "card1":          1001,
}


@pytest.fixture
def client():
    """TestClient with all model loading mocked."""
    mock_registry = _make_mock_registry()
    mock_pipeline = _make_mock_pipeline()

    with patch("serving.api.predictor.registry", mock_registry), \
         patch("serving.api.predictor.registry.pipeline", mock_pipeline):

        mock_registry.pipeline = mock_pipeline

        # Patch _run_pipeline to return fixed array
        with patch("serving.api.predictor._run_pipeline", return_value=np.random.randn(1, 20)), \
             patch("serving.api.predictor._log_score_to_db", return_value=None), \
             patch("serving.api.predictor._get_shap_contributors", return_value=[]):

            from serving.api.main import app
            with TestClient(app, raise_server_exceptions=True) as c:
                yield c


# ---------------------------------------------------------------------------
# Health & Operations Endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_version" in data
        assert "uptime_seconds" in data
        assert "threshold" in data

    def test_uptime_positive(self, client):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0.0

    def test_request_id_in_headers(self, client):
        response = client.get("/health")
        assert "x-request-id" in response.headers


class TestMetricsEndpoint:

    def test_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_is_prometheus_format(self, client):
        text = client.get("/metrics").text
        assert "http_requests_total" in text
        assert "fraud_scores_total" in text


class TestModelInfoEndpoint:

    def test_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/model/info").json()
        assert "model_name"    in data
        assert "model_version" in data
        assert "threshold"     in data


# ---------------------------------------------------------------------------
# Single Score Endpoint
# ---------------------------------------------------------------------------

class TestScoreEndpoint:

    def test_valid_request_returns_200(self, client):
        response = client.post("/score", json=VALID_TXN)
        assert response.status_code == 200

    def test_response_contains_required_fields(self, client):
        data = client.post("/score", json=VALID_TXN).json()
        required = [
            "TransactionID", "fraud_probability", "is_flagged",
            "threshold", "risk_level", "model_version",
            "latency_ms", "scored_at",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_fraud_probability_in_unit_interval(self, client):
        data = client.post("/score", json=VALID_TXN).json()
        prob = data["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_latency_positive(self, client):
        data = client.post("/score", json=VALID_TXN).json()
        assert data["latency_ms"] > 0

    def test_risk_level_valid_enum(self, client):
        data = client.post("/score", json=VALID_TXN).json()
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_is_flagged_is_boolean(self, client):
        data = client.post("/score", json=VALID_TXN).json()
        assert isinstance(data["is_flagged"], bool)

    def test_echo_transaction_id(self, client):
        data = client.post("/score", json=VALID_TXN).json()
        assert data["TransactionID"] == VALID_TXN["TransactionID"]

    def test_missing_required_field_returns_422(self, client):
        """TransactionAmt is required — omitting it must return 422."""
        bad_txn = {k: v for k, v in VALID_TXN.items() if k != "TransactionAmt"}
        response = client.post("/score", json=bad_txn)
        assert response.status_code == 422

    def test_negative_amount_returns_422(self, client):
        bad_txn = {**VALID_TXN, "TransactionAmt": -50.0}
        response = client.post("/score", json=bad_txn)
        assert response.status_code == 422

    def test_request_id_in_response_headers(self, client):
        response = client.post("/score", json=VALID_TXN)
        assert "x-request-id" in response.headers


# ---------------------------------------------------------------------------
# Batch Score Endpoint
# ---------------------------------------------------------------------------

class TestBatchScoreEndpoint:

    def test_single_item_batch(self, client):
        payload = {"transactions": [VALID_TXN]}
        response = client.post("/score/batch", json=payload)
        assert response.status_code == 200

    def test_batch_response_schema(self, client):
        payload = {"transactions": [VALID_TXN, VALID_TXN]}
        data = client.post("/score/batch", json=payload).json()
        assert "results"          in data
        assert "batch_size"       in data
        assert "total_latency_ms" in data
        assert "n_flagged"        in data
        assert "flag_rate"        in data

    def test_batch_size_matches_input(self, client):
        txns = [{**VALID_TXN, "TransactionID": i} for i in range(5)]
        data = client.post("/score/batch", json={"transactions": txns}).json()
        assert data["batch_size"] == 5
        assert len(data["results"]) == 5

    def test_empty_batch_returns_422(self, client):
        response = client.post("/score/batch", json={"transactions": []})
        assert response.status_code == 422

    def test_oversized_batch_returns_422(self, client):
        txns = [{**VALID_TXN, "TransactionID": i} for i in range(1001)]
        response = client.post("/score/batch", json={"transactions": txns})
        assert response.status_code == 422

    def test_flag_rate_consistent(self, client):
        txns = [{**VALID_TXN, "TransactionID": i} for i in range(10)]
        data = client.post("/score/batch", json={"transactions": txns}).json()
        expected_rate = data["n_flagged"] / data["batch_size"]
        assert abs(data["flag_rate"] - expected_rate) < 1e-6
