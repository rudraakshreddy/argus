-- ============================================================
-- Fraud & Anomaly Risk-Scoring Engine -- SQLite Schema
-- ============================================================
-- Design principles:
--   1. Normalised: labels separated from raw transactions
--   2. Audit trail: every model score persisted (model_scores)
--   3. Indexed for both batch training queries and real-time API
-- ============================================================

PRAGMA journal_mode = WAL;       -- Write-Ahead Logging for concurrent reads
PRAGMA foreign_keys = ON;

-- -------------------------------------------------------
-- Core transaction table (IEEE-CIS schema)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
    TransactionID       INTEGER PRIMARY KEY,
    TransactionDT       INTEGER,        -- seconds offset from reference date
    TransactionAmt      REAL,
    ProductCD           TEXT,
    card1               INTEGER,
    card2               REAL,
    card3               REAL,
    card4               TEXT,
    card5               REAL,
    card6               TEXT,
    addr1               REAL,
    addr2               REAL,
    dist1               REAL,
    dist2               REAL,
    P_emaildomain       TEXT,
    R_emaildomain       TEXT,
    -- C features (counting features, obfuscated by Vesta)
    C1 REAL, C2 REAL, C3 REAL, C4 REAL, C5 REAL,
    C6 REAL, C7 REAL, C8 REAL, C9 REAL, C10 REAL,
    C11 REAL, C12 REAL, C13 REAL, C14 REAL,
    -- D features (timedelta features)
    D1 REAL, D2 REAL, D3 REAL, D4 REAL, D5 REAL,
    D6 REAL, D7 REAL, D8 REAL, D9 REAL, D10 REAL,
    D11 REAL, D12 REAL, D13 REAL, D14 REAL, D15 REAL,
    -- M features (match features)
    M1 TEXT, M2 TEXT, M3 TEXT, M4 TEXT, M5 TEXT,
    M6 TEXT, M7 TEXT, M8 TEXT, M9 TEXT,
    -- Identity features (joined from train_identity.csv)
    id_01 REAL, id_02 REAL, id_03 REAL, id_04 REAL, id_05 REAL,
    id_06 REAL, id_07 REAL, id_08 REAL, id_09 REAL, id_10 REAL,
    id_11 REAL, id_12 TEXT, id_13 REAL, id_14 REAL, id_15 TEXT,
    id_16 TEXT, id_17 REAL, id_18 REAL, id_19 REAL, id_20 REAL,
    id_21 REAL, id_22 REAL, id_23 TEXT, id_24 REAL, id_25 REAL,
    id_26 REAL, id_27 TEXT, id_28 TEXT, id_29 TEXT, id_30 TEXT,
    id_31 TEXT, id_32 REAL, id_33 TEXT, id_34 TEXT, id_35 TEXT,
    id_36 TEXT, id_37 TEXT, id_38 TEXT,
    DeviceType          TEXT,
    DeviceInfo          TEXT,
    -- Metadata
    loaded_at           TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------
-- Fraud ground-truth labels (separate for clean joins)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS fraud_labels (
    TransactionID       INTEGER PRIMARY KEY
                            REFERENCES transactions(TransactionID)
                            ON DELETE CASCADE,
    isFraud             INTEGER NOT NULL CHECK(isFraud IN (0, 1)),
    label_source        TEXT NOT NULL DEFAULT 'ieee_cis',
    labelled_at         TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------
-- Engineered feature cache (written by feature_engineering.py)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS engineered_features (
    TransactionID       INTEGER PRIMARY KEY
                            REFERENCES transactions(TransactionID)
                            ON DELETE CASCADE,
    log_amt             REAL,
    amt_zscore          REAL,
    hour_sin            REAL,
    hour_cos            REAL,
    dow_sin             REAL,
    dow_cos             REAL,
    is_weekend          INTEGER,
    card1_freq          REAL,
    P_email_fraud_rate  REAL,
    R_email_fraud_rate  REAL,
    -- V-feature PCA components (top 30)
    pca_v01 REAL, pca_v02 REAL, pca_v03 REAL, pca_v04 REAL, pca_v05 REAL,
    pca_v06 REAL, pca_v07 REAL, pca_v08 REAL, pca_v09 REAL, pca_v10 REAL,
    pca_v11 REAL, pca_v12 REAL, pca_v13 REAL, pca_v14 REAL, pca_v15 REAL,
    pca_v16 REAL, pca_v17 REAL, pca_v18 REAL, pca_v19 REAL, pca_v20 REAL,
    pca_v21 REAL, pca_v22 REAL, pca_v23 REAL, pca_v24 REAL, pca_v25 REAL,
    pca_v26 REAL, pca_v27 REAL, pca_v28 REAL, pca_v29 REAL, pca_v30 REAL,
    feature_version     TEXT NOT NULL DEFAULT '1.0',
    created_at          TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------
-- Model scoring audit trail (written by FastAPI + dashboard)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_scores (
    score_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    TransactionID       INTEGER,            -- nullable: supports external scoring
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    fraud_prob          REAL NOT NULL,
    threshold           REAL NOT NULL,
    is_flagged          INTEGER NOT NULL CHECK(is_flagged IN (0, 1)),
    -- SHAP top-3 contributors (JSON string)
    top3_contributors   TEXT,
    latency_ms          REAL,
    scored_at           TEXT DEFAULT (datetime('now'))
);

-- -------------------------------------------------------
-- Model registry (tracks trained model versions)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    registry_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    artifact_path       TEXT NOT NULL,
    train_auprc         REAL,
    val_auprc           REAL,
    test_auprc          REAL,
    threshold           REAL,
    is_production       INTEGER NOT NULL DEFAULT 0
                            CHECK(is_production IN (0, 1)),
    trained_at          TEXT DEFAULT (datetime('now')),
    notes               TEXT
);

-- -------------------------------------------------------
-- Indexes
-- -------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_fraud_label    ON fraud_labels(isFraud);
CREATE INDEX IF NOT EXISTS idx_txn_dt         ON transactions(TransactionDT);
CREATE INDEX IF NOT EXISTS idx_score_model    ON model_scores(model_name, scored_at);
CREATE INDEX IF NOT EXISTS idx_score_flagged  ON model_scores(is_flagged, scored_at);
CREATE INDEX IF NOT EXISTS idx_registry_prod  ON model_registry(model_name, is_production);
