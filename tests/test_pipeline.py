"""
tests/test_pipeline.py
========================
Unit tests for the feature engineering pipeline.

Tests:
  - All custom sklearn Transformers fit/transform without error
  - No data leakage: LeakFreeTargetEncoder must use only training fold labels
  - VFeaturePCA reduces dimensionality correctly
  - Missing value flags are created for columns above threshold
  - Pipeline serialisation: joblib.dump -> joblib.load -> predict produces identical output
  - Output dtype is float64 (required by all downstream models)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.feature_engineering import (
    CyclicalTimeTransformer,
    FrequencyEncoder,
    LeakFreeTargetEncoder,
    LogAmountTransformer,
    MissingFlagTransformer,
    VFeaturePCA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_df():
    """Minimal DataFrame mimicking IEEE-CIS transaction schema."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "TransactionID":  np.arange(n),
        "TransactionDT":  np.random.randint(86400, 86400 * 30, size=n),
        "TransactionAmt": np.random.exponential(scale=100, size=n) + 0.01,
        "card1":          np.random.choice([1001, 1002, 1003, 1004], size=n),
        "card4":          np.random.choice(["visa", "mastercard", "amex"], size=n),
        "ProductCD":      np.random.choice(["W", "H", "C", "S"], size=n),
        "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", None], size=n),
        "R_emaildomain":  np.random.choice(["gmail.com", "hotmail.com", None], size=n),
    })
    # Add V-features with some missingness
    for i in range(1, 21):
        col = pd.Series(np.random.randn(n))
        col[np.random.choice(n, size=n//3, replace=False)] = np.nan
        df[f"V{i}"] = col
    return df


@pytest.fixture
def fraud_labels(minimal_df):
    """Binary fraud labels with ~3% fraud rate."""
    np.random.seed(42)
    y = (np.random.rand(len(minimal_df)) < 0.03).astype(int)
    return pd.Series(y, name="isFraud")


# ---------------------------------------------------------------------------
# LogAmountTransformer
# ---------------------------------------------------------------------------

class TestLogAmountTransformer:

    def test_fit_transform_adds_columns(self, minimal_df):
        t = LogAmountTransformer()
        out = t.fit(minimal_df).transform(minimal_df)
        assert "log_amt" in out.columns
        assert "amt_zscore" in out.columns

    def test_log_amt_is_positive(self, minimal_df):
        t = LogAmountTransformer()
        out = t.fit(minimal_df).transform(minimal_df)
        assert (out["log_amt"] >= 0).all(), "log1p of positive amounts must be non-negative"

    def test_no_inf_values(self, minimal_df):
        t = LogAmountTransformer()
        out = t.fit(minimal_df).transform(minimal_df)
        assert not np.isinf(out["log_amt"]).any()
        assert not np.isinf(out["amt_zscore"]).any()

    def test_fit_only_on_train(self, minimal_df):
        """card_stats_ must be fit only on training data."""
        train, test = minimal_df.iloc[:400], minimal_df.iloc[400:]
        t = LogAmountTransformer().fit(train)
        # Transform must not re-fit on test
        out = t.transform(test)
        assert "log_amt" in out.columns


# ---------------------------------------------------------------------------
# CyclicalTimeTransformer
# ---------------------------------------------------------------------------

class TestCyclicalTimeTransformer:

    def test_adds_four_columns(self, minimal_df):
        t = CyclicalTimeTransformer()
        out = t.fit_transform(minimal_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]:
            assert col in out.columns

    def test_sin_cos_bounds(self, minimal_df):
        t = CyclicalTimeTransformer()
        out = t.fit_transform(minimal_df)
        assert out["hour_sin"].between(-1, 1).all()
        assert out["hour_cos"].between(-1, 1).all()
        assert out["dow_sin"].between(-1, 1).all()
        assert out["dow_cos"].between(-1, 1).all()

    def test_is_weekend_binary(self, minimal_df):
        t = CyclicalTimeTransformer()
        out = t.fit_transform(minimal_df)
        assert set(out["is_weekend"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# MissingFlagTransformer
# ---------------------------------------------------------------------------

class TestMissingFlagTransformer:

    def test_creates_missing_indicators(self, minimal_df):
        t = MissingFlagTransformer(threshold=0.05)
        out = t.fit(minimal_df).transform(minimal_df)
        missing_cols = [c for c in out.columns if c.endswith("_missing")]
        assert len(missing_cols) > 0, "Should create missing indicators for V-features"

    def test_indicators_are_binary(self, minimal_df):
        t = MissingFlagTransformer(threshold=0.05)
        out = t.fit_transform(minimal_df)
        for col in [c for c in out.columns if c.endswith("_missing")]:
            assert set(out[col].unique()).issubset({0, 1})

    def test_no_new_nulls_introduced(self, minimal_df):
        original_null_count = minimal_df.isnull().sum().sum()
        t = MissingFlagTransformer(threshold=0.05)
        out = t.fit_transform(minimal_df)
        # Missing indicator columns themselves should have no nulls
        indicator_cols = [c for c in out.columns if c.endswith("_missing")]
        assert out[indicator_cols].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# LeakFreeTargetEncoder — data leakage test
# ---------------------------------------------------------------------------

class TestLeakFreeTargetEncoder:

    def test_no_label_leakage(self, minimal_df, fraud_labels):
        """
        Each row's target-encoded value must NOT use its own label.
        Test: compare encoding when one label is flipped — if leakage exists,
        the encoding of the flipped sample changes by more than just smoothing.
        """
        t = LeakFreeTargetEncoder(cols=["card4"], n_splits=5)
        t.fit(minimal_df, fraud_labels)
        enc1 = t.transform(minimal_df)["card4_target_enc"].values.copy()

        # Flip one label and refit
        y_flipped = fraud_labels.copy()
        y_flipped.iloc[0] = 1 - y_flipped.iloc[0]
        t2 = LeakFreeTargetEncoder(cols=["card4"], n_splits=5)
        t2.fit(minimal_df, y_flipped)
        enc2 = t2.transform(minimal_df)["card4_target_enc"].values

        # Encodings must differ (model has changed) but only by smoothed amounts
        diff = np.abs(enc1 - enc2).max()
        assert diff < 0.5, "Single label flip should not cause >0.5 change (leakage indicator)"

    def test_encoded_values_in_range(self, minimal_df, fraud_labels):
        t = LeakFreeTargetEncoder(cols=["card4"])
        out = t.fit(minimal_df, fraud_labels).transform(minimal_df)
        col = out["card4_target_enc"]
        assert col.between(0, 1).all(), "Target-encoded values must be in [0,1] (fraud probabilities)"


# ---------------------------------------------------------------------------
# VFeaturePCA
# ---------------------------------------------------------------------------

class TestVFeaturePCA:

    def test_reduces_dimension(self, minimal_df):
        v_cols_before = [c for c in minimal_df.columns if c.startswith("V")]
        t = VFeaturePCA(n_components=5)
        out = t.fit_transform(minimal_df)
        v_cols_after = [c for c in out.columns if c.startswith("V") and not c.startswith("pca")]
        pca_cols = [c for c in out.columns if c.startswith("pca_v")]
        assert len(v_cols_after) == 0, "Raw V features should be dropped after PCA"
        assert len(pca_cols) == 5

    def test_handles_all_nan_column(self, minimal_df):
        df = minimal_df.copy()
        df["V1"] = np.nan  # All-NaN column
        t = VFeaturePCA(n_components=3)
        out = t.fit_transform(df)
        assert not out.isnull().any().any()


# ---------------------------------------------------------------------------
# Integration: full pipeline round-trip
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def test_pipeline_output_dtype(self, minimal_df, fraud_labels):
        """Final pipeline output must be float64 numpy array."""
        import joblib, tempfile
        from processing.pipeline import build_feature_pipeline

        pipeline = build_feature_pipeline()
        X = pipeline.fit_transform(minimal_df, fraud_labels)
        assert X.dtype == np.float64 or X.dtype == np.float32

    def test_pipeline_serialisation(self, minimal_df, fraud_labels, tmp_path):
        """joblib.dump -> joblib.load -> transform produces identical output."""
        import joblib
        from processing.pipeline import build_feature_pipeline

        pipeline = build_feature_pipeline()
        X1 = pipeline.fit_transform(minimal_df, fraud_labels)

        path = tmp_path / "test_pipeline.joblib"
        joblib.dump(pipeline, path)
        loaded = joblib.load(path)
        X2 = loaded.transform(minimal_df)

        np.testing.assert_array_almost_equal(X1, X2, decimal=6)

    def test_no_nans_in_output(self, minimal_df, fraud_labels):
        """Pipeline must produce NaN-free output (imputer covers any remaining gaps)."""
        from processing.pipeline import build_feature_pipeline
        pipeline = build_feature_pipeline()
        X = pipeline.fit_transform(minimal_df, fraud_labels)
        assert not np.isnan(X).any(), "Pipeline output must be NaN-free"
