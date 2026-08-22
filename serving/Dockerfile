# syntax=docker/dockerfile:1.7
# ============================================================
# Stage 1: Builder — installs all Python dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (for packages that compile C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# Stage 2: Runtime — minimal image, non-root user
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="fraud-risk-engine"
LABEL description="Fraud & Anomaly Risk-Scoring API — FastAPI + XGBoost"

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

WORKDIR /app

# Copy application source
COPY serving/     ./serving/
COPY modeling/    ./modeling/
COPY processing/  ./processing/
COPY db/          ./db/

# Copy trained model artifacts (or mount as volume in docker-compose)
# In production, mount: -v /path/to/models:/app/models
COPY models/ ./models/

# DB directory (SQLite file mounted via volume in production)
RUN mkdir -p /app/db && chown appuser:appgroup /app/db

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check (Docker will mark container unhealthy if this fails)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Production server: gunicorn + uvicorn workers
CMD ["python", "-m", "gunicorn", \
     "serving.api.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
