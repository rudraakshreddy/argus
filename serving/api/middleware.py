"""
serving/api/middleware.py
==========================
FastAPI middleware for:
  1. Request ID injection     — every request gets a unique UUID in headers
  2. Structured access logging — method, path, status, latency logged to stdout
  3. Prometheus-format metrics — /metrics endpoint (counters + latency histograms)
  4. Error handling            — structured JSON error responses

All latency measurements use time.perf_counter() (monotonic, high-resolution).
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

log = logging.getLogger("api.access")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique X-Request-ID header into every request and response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Structured access log in JSON-like format.
    Logs: request_id, method, path, status_code, latency_ms.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        request_id = getattr(request.state, "request_id", "unknown")
        log.info(
            f'request_id="{request_id}" method="{request.method}" '
            f'path="{request.url.path}" status={response.status_code} '
            f'latency_ms={latency_ms}'
        )
        return response


class MetricsCollector:
    """
    In-memory Prometheus-format metrics collector.

    Tracks:
      - http_requests_total{method, path, status}
      - http_request_duration_ms{path} — p50, p95, p99
      - fraud_scores_total
      - fraud_flagged_total
    """

    def __init__(self):
        self.request_counts: dict[tuple, int] = defaultdict(int)
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.fraud_scores_total: int = 0
        self.fraud_flagged_total: int = 0
        self.start_time: float = time.perf_counter()

    def record_request(
        self,
        method: str,
        path: str,
        status: int,
        latency_ms: float,
    ) -> None:
        key = (method, path, str(status))
        self.request_counts[key] += 1
        self.latencies[path].append(latency_ms)
        # Keep only last 10,000 latency samples per path (memory cap)
        if len(self.latencies[path]) > 10_000:
            self.latencies[path] = self.latencies[path][-5_000:]

    def record_score(self, is_flagged: bool) -> None:
        self.fraud_scores_total += 1
        if is_flagged:
            self.fraud_flagged_total += 1

    def percentile(self, path: str, p: float) -> float:
        data = self.latencies.get(path, [])
        if not data:
            return 0.0
        import numpy as np
        return float(np.percentile(data, p))

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus text exposition format."""
        import numpy as np
        lines = [
            "# HELP http_requests_total Total HTTP requests by method/path/status",
            "# TYPE http_requests_total counter",
        ]
        for (method, path, status), count in self.request_counts.items():
            lines.append(
                f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
            )

        lines += [
            "",
            "# HELP http_request_duration_ms HTTP request latency in milliseconds",
            "# TYPE http_request_duration_ms summary",
        ]
        for path, lats in self.latencies.items():
            if lats:
                for q, pct_val in [(0.50, np.percentile(lats, 50)),
                                   (0.95, np.percentile(lats, 95)),
                                   (0.99, np.percentile(lats, 99))]:
                    lines.append(
                        f'http_request_duration_ms{{path="{path}",quantile="{q}"}} {pct_val:.2f}'
                    )
                lines.append(
                    f'http_request_duration_ms_count{{path="{path}"}} {len(lats)}'
                )

        lines += [
            "",
            "# HELP fraud_scores_total Total transactions scored",
            "# TYPE fraud_scores_total counter",
            f"fraud_scores_total {self.fraud_scores_total}",
            "",
            "# HELP fraud_flagged_total Total transactions flagged as fraud",
            "# TYPE fraud_flagged_total counter",
            f"fraud_flagged_total {self.fraud_flagged_total}",
            "",
            "# HELP api_uptime_seconds API server uptime in seconds",
            "# TYPE api_uptime_seconds gauge",
            f"api_uptime_seconds {time.perf_counter() - self.start_time:.1f}",
        ]
        return "\n".join(lines)


# Module-level singleton metrics collector
metrics = MetricsCollector()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request metrics into the MetricsCollector singleton."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        metrics.record_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
        )
        return response
