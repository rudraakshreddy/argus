# ==============================================================================
# ARGUS — Makefile
# Adaptive Real-time Grading & Unsupervised Scoring
# ==============================================================================
#
# Usage:
#   make run-pipeline   Full ETL -> Feature eng. -> Train all -> Evaluate
#   make train          Train all four models (assumes pipeline already run)
#   make evaluate       Evaluation, plots, LaTeX table (assumes training done)
#   make report         Compile LaTeX PDF
#   make up             Start all Docker services (API + Dashboard + Airflow)
#   make down           Stop all Docker services
#   make test           Run pytest suite
#   make clean          Remove generated artifacts (models, figures)
#   make help           Show this message

PYTHON   := python
PROJ_DIR := $(CURDIR)
DATA_DIR := $(PROJ_DIR)/data/raw
DB_PATH  := $(PROJ_DIR)/db/fraud.db
REPORT   := $(PROJ_DIR)/report/main.tex
VERSION  := 1

.PHONY: all run-pipeline ingest features imbalance train train-lr train-xgb \
        train-if train-ae cost evaluate report up down restart test clean help

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
all: help

# ---------------------------------------------------------------------------
# Full pipeline (end-to-end, sequential)
# ---------------------------------------------------------------------------
run-pipeline: ingest features imbalance train cost evaluate
	@echo "=========================================================="
	@echo " ARGUS pipeline complete. Results in models/ and report/"
	@echo "=========================================================="

# ---------------------------------------------------------------------------
# Individual pipeline stages
# ---------------------------------------------------------------------------

ingest:
	@echo "[1/7] Initialising database and loading IEEE-CIS data..."
	$(PYTHON) db/init_db.py
	$(PYTHON) ingestion/load_ieee_cis.py --db $(DB_PATH)

eda:
	@echo "[EDA] Generating EDA figures..."
	$(PYTHON) processing/eda.py --db $(DB_PATH)

features:
	@echo "[2/7] Running feature engineering pipeline..."
	$(PYTHON) processing/eda.py --db $(DB_PATH)
	$(PYTHON) processing/pipeline.py --db $(DB_PATH) --version $(VERSION)

imbalance:
	@echo "[3/7] Comparing imbalance handling strategies..."
	$(PYTHON) processing/imbalance_handler.py --version $(VERSION)

train: train-lr train-xgb train-if train-ae
	@echo "[4/7] All models trained."

train-lr:
	@echo "  [LR] Training Logistic Regression..."
	$(PYTHON) modeling/supervised/logistic_regression.py --version $(VERSION) --trials 50

train-xgb:
	@echo "  [XGB] Training XGBoost (100-trial Optuna)..."
	$(PYTHON) modeling/supervised/xgboost_model.py --version $(VERSION) --trials 100

train-if:
	@echo "  [IF] Training Isolation Forest..."
	$(PYTHON) modeling/unsupervised/isolation_forest.py --version $(VERSION)

train-ae:
	@echo "  [AE] Training Autoencoder (PyTorch)..."
	$(PYTHON) modeling/unsupervised/autoencoder.py --version $(VERSION)

cost:
	@echo "[5/7] Running cost-curve analysis..."
	$(PYTHON) modeling/cost_analysis.py --version $(VERSION)

evaluate:
	@echo "[6/7] Evaluating all models..."
	$(PYTHON) evaluation/metrics.py --version $(VERSION)
	$(PYTHON) evaluation/plots.py --version $(VERSION)
	$(PYTHON) evaluation/model_comparison.py --version $(VERSION)

# ---------------------------------------------------------------------------
# LaTeX report
# ---------------------------------------------------------------------------
report:
	@echo "[7/7] Compiling LaTeX report..."
	cd report && pdflatex -interaction=nonstopmode main.tex
	cd report && biber main
	cd report && pdflatex -interaction=nonstopmode main.tex
	cd report && pdflatex -interaction=nonstopmode main.tex
	@echo "Report compiled: report/main.pdf"

# ---------------------------------------------------------------------------
# Docker services
# ---------------------------------------------------------------------------
up:
	@echo "Starting ARGUS services (API + Dashboard + Airflow)..."
	docker compose -f serving/docker-compose.yml up -d --build
	@echo "API:       http://localhost:8000"
	@echo "API Docs:  http://localhost:8000/docs"
	@echo "Dashboard: http://localhost:8501"
	@echo "Airflow:   http://localhost:8080  (admin/admin)"

down:
	@echo "Stopping ARGUS services..."
	docker compose -f serving/docker-compose.yml down

restart:
	docker compose -f serving/docker-compose.yml restart

logs-api:
	docker compose -f serving/docker-compose.yml logs -f api

logs-dashboard:
	docker compose -f serving/docker-compose.yml logs -f dashboard

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:
	@echo "Running pytest suite..."
	$(PYTHON) -m pytest tests/ -v --tb=short --color=yes
	@echo "All tests passed."

test-pipeline:
	$(PYTHON) -m pytest tests/test_pipeline.py -v

test-metrics:
	$(PYTHON) -m pytest tests/test_metrics.py -v

test-api:
	$(PYTHON) -m pytest tests/test_api.py -v

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	@echo "Cleaning generated artifacts..."
	-del /Q models\*.joblib models\*.pt models\*.npy 2>nul || true
	-del /Q report\figures\models\*.pdf report\figures\eda\*.pdf 2>nul || true
	-del /Q report\figures\shap\*.pdf 2>nul || true
	-del /Q report\*.aux report\*.bbl report\*.bcf report\*.blg 2>nul || true
	-del /Q report\*.log report\*.out report\*.run.xml 2>nul || true
	@echo "Clean complete."

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "ARGUS Makefile"
	@echo "=============="
	@echo "  make run-pipeline   Full end-to-end pipeline"
	@echo "  make ingest         Load IEEE-CIS data into SQLite"
	@echo "  make features       Run feature engineering"
	@echo "  make train          Train all four models"
	@echo "  make evaluate       Metrics, plots, benchmark table"
	@echo "  make report         Compile LaTeX PDF"
	@echo "  make up             Start Docker services"
	@echo "  make down           Stop Docker services"
	@echo "  make test           Run pytest suite"
	@echo "  make clean          Remove generated files"
	@echo ""
