# ARGUS

**Adaptive Real-time Grading & Unsupervised Scoring — cost-sensitive fraud detection under temporal validation.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-live-FF4B4B.svg)](https://argus-system.streamlit.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)

**[Live dashboard](https://argus-system.streamlit.app)** · **[Full technical paper (PDF)](report/argus_report.pdf)**

---

## What this is

A fraud detection system and study on the **IEEE-CIS** dataset: **590,540 card-not-present transactions**,
433 predictors, 182 days, **3.50 % fraud rate**.

Three properties of this problem determine everything about how it is modelled and evaluated:

- **Severe imbalance** — predicting "legitimate" everywhere scores 96.5 % accuracy and is worthless.
- **Cost asymmetry** — a missed fraud costs ≈ \$850; a false alarm costs ≈ \$12. A ratio near **1:71**
  makes the default 0.5 threshold indefensible.
- **Non-stationarity** — fraud is adversarial and drifts, so the evaluation must respect time.

## The design decisions that matter

**1. Temporal validation, not a shuffled split.**

Transactions are time-ordered, and a deployed model trains on the past to score the future. ARGUS trains on
the earliest 80 % of transaction time and tests on the latest 20 %. Holding features and model fixed, the
study also runs the shuffled alternative to quantify how much optimism it introduces — a substantial gap
that a random split would silently hide.

**2. The threshold is chosen by minimising money, not by maximising F1.**

```
E[Cost](θ) = ( c_FP · FP(θ) + c_FN · FN(θ) ) / N        θ* = argmin_θ E[Cost](θ)
```

with `c_FP = $12` (≈30 min analyst review) and `c_FN = $850` (median fraudulent amount). Each model is
evaluated at **its own** cost-optimal threshold, since comparing models at a shared fixed threshold
confounds discrimination with calibration. The cost ratio is swept over three orders of magnitude so the
choice is a stated sensitivity rather than an arbitrary constant.

**3. Anomaly scores are searched on the quantile scale.**

Autoencoder reconstruction error is heavy-tailed — extreme outliers exceed the bulk of the distribution by
many orders of magnitude. Min–max normalising such a score compresses almost the whole distribution toward
zero, so a uniform threshold grid over [0,1] has essentially **no resolution** where the decision boundary
actually lies. AUROC and AUPRC are unaffected (they are rank-based), so the failure is silent: the ranking
metrics look fine while the deployed operating point is badly wrong. ARGUS therefore converts anomaly
scores to their empirical CDF before searching the threshold — strictly monotone, so ranking metrics are
unchanged, while the search grid is uniform in probability mass.

## Models compared

| Model | Type | Role |
|---|---|---|
| Logistic regression | Supervised | Interpretable reference with balanced class weights |
| **XGBoost** | Supervised | 600 depth-8 trees, `scale_pos_weight`, AUCPR objective |
| Isolation forest | Unsupervised | Isolation path length; uses no labels |
| Autoencoder | Unsupervised | `d→128→64→32→64→128→d`, trained on legitimate traffic only |

The two unsupervised detectors are included to test a specific claim — that they are competitive when
labels are scarce. Since labels *are* available here, the comparison quantifies the cost of not using them.

## Feature pipeline

Every transform is fitted on training data only and is a pipeline step, so the discipline is structural
rather than a matter of care:

- **Log amount**, additionally standardised against the originating card's own history
- **Cyclical time encoding** (sin/cos of hour-of-day and day-of-week) so hour 23 and hour 0 are adjacent
- **Missingness indicators** for columns above 5 % missing — absence is informative here, not incidental
- **Frequency encoding** for high-cardinality identifiers
- **Cross-fitted target encoding** — out-of-fold, so a row's own label never enters its own encoding
- **PCA on the 339 anonymised V-features** → 30 components (75.7 % of variance)
- Median imputation and standardisation → **408 numeric features**

## Repository layout

```
├── ingestion/          IEEE-CIS loading and synthetic generation
├── processing/         EDA, feature engineering, imbalance handling, pipeline
├── modeling/
│   ├── supervised/     Logistic regression, XGBoost, model selection
│   ├── unsupervised/   Isolation forest, autoencoder
│   └── cost_analysis.py   Cost curves and threshold selection
├── evaluation/         Metrics, model comparison, plots
├── serving/api/        FastAPI scoring service (Docker)
├── dashboard/          Streamlit app (overview, performance, drift)
├── automation/dags/    Airflow ingest and retrain DAGs
├── tests/              API, metric and pipeline tests
└── report/             Technical paper (LaTeX source + PDF)
```

## Quick start

```bash
git clone https://github.com/rudraakshreddy/argus.git
cd argus
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

make pipeline        # ingest → features → train → evaluate
make serve           # API on :8000, dashboard on :8501
```

Scoring a transaction:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 249.99, "ProductCD": "W", "card1": 13553, "card4": "visa"}'
```

The IEEE-CIS source files are not redistributed here; obtain them from the original competition and place
them in `data/raw/`.

## Scope

**In scope.** Binary classification of a transaction at authorisation time on a single public benchmark;
offline batch evaluation under a temporal partition; a cost-sensitive decision layer with explicit FP/FN
costs; post-hoc explanation via SHAP; calibration and week-by-week drift tracking.

**Out of scope.** Real-time serving guarantees (latency and throughput are not benchmarked); adversarial
adaptation (attackers are fixed historical behaviour); **graph and network features** (device/IP/card
linkage, known to carry strong signal, are not constructed); sequential or account-level state (each
transaction is scored independently); online-learning and retraining-cadence optimisation; fairness and
disparate-impact assessment (the anonymised fields would not support it); and estimation of the cost
parameters themselves, which are treated as given inputs and swept for sensitivity.

## Limitations

- **One dataset, 182 days.** Drift beyond this window is not observed, and the fraud mix is specific to
  this merchant population.
- **Anonymised features.** 339 of the predictors are undocumented, so feature-level interpretation is
  limited to the engineered and named columns.
- **Cost parameters are estimates.** `c_FP` and `c_FN` are approximations, not an institution's own ledger;
  results are reported with a sensitivity sweep rather than as a single number.
- **No graph features**, which is the most consequential omission relative to production fraud systems.
- **Label latency ignored.** In deployment, fraud labels arrive weeks later via chargebacks; this study
  assumes labels are available at training time.

## Citation

```bibtex
@techreport{reddy2026argus,
  title  = {ARGUS: Cost-Sensitive Fraud Detection under Temporal Validation},
  author = {Yeddula Rudraaksh Reddy and S. N. Chakri},
  year   = {2026},
  type   = {Technical Report},
  url    = {https://github.com/rudraakshreddy/argus}
}
```

## Authors

- **Yeddula Rudraaksh Reddy** — primary and corresponding author ·
  [yeddularudraaksh@gmail.com](mailto:yeddularudraaksh@gmail.com) ·
  [LinkedIn](https://www.linkedin.com/in/rudraakshreddy) · [GitHub](https://github.com/rudraakshreddy)
- **S. N. Chakri** — [snchakrim@gmail.com](mailto:snchakrim@gmail.com) ·
  [LinkedIn](https://www.linkedin.com/in/snchakri) · [GitHub](https://github.com/snchakri) ·
  [snchakri.com](https://snchakri.com)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The IEEE-CIS dataset is the property of its publishers and is not redistributed by this repository.
