"""
serving/api/schema.py
======================
Pydantic v2 request and response schemas for the FastAPI scoring API.

Design principles:
  - Every field is documented with a description
  - Input fields match the IEEE-CIS raw feature schema
  - Response includes fraud_probability, is_flagged, latency_ms,
    model_version, and top-3 SHAP contributors for explainability
  - Batch endpoint accepts up to 1,000 transactions
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TransactionFeatures(BaseModel):
    """
    Raw transaction features matching the IEEE-CIS schema.
    All fields mirror the feature engineering pipeline inputs.
    Optional fields are nullable (match real-world data incompleteness).
    """
    # Core transaction fields
    TransactionID:  int   = Field(..., description="Unique transaction identifier")
    TransactionDT:  int   = Field(..., description="Seconds from reference date (timedelta)")
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD:      str   = Field(..., description="Product code (W/H/C/S/R)")

    # Card features
    card1: int            = Field(..., description="Card anonymised feature 1")
    card2: float | None   = Field(None, description="Card anonymised feature 2")
    card3: float | None   = Field(None, description="Card anonymised feature 3")
    card4: str | None     = Field(None, description="Card type (visa/mastercard/etc)")
    card5: float | None   = Field(None, description="Card anonymised feature 5")
    card6: str | None     = Field(None, description="Card category (debit/credit)")

    # Address features
    addr1: float | None   = Field(None, description="Billing address feature 1")
    addr2: float | None   = Field(None, description="Billing country code")

    # Distance features
    dist1: float | None   = Field(None, description="Distance (transaction vs billing)")
    dist2: float | None   = Field(None, description="Distance (transaction vs mailing)")

    # Email domain features
    P_emaildomain: str | None = Field(None, description="Purchaser email domain")
    R_emaildomain: str | None = Field(None, description="Recipient email domain")

    # Counting features C1-C14
    C1:  float | None = None; C2:  float | None = None; C3:  float | None = None
    C4:  float | None = None; C5:  float | None = None; C6:  float | None = None
    C7:  float | None = None; C8:  float | None = None; C9:  float | None = None
    C10: float | None = None; C11: float | None = None; C12: float | None = None
    C13: float | None = None; C14: float | None = None

    # Timedelta features D1-D15
    D1:  float | None = None; D2:  float | None = None; D3:  float | None = None
    D4:  float | None = None; D5:  float | None = None; D6:  float | None = None
    D7:  float | None = None; D8:  float | None = None; D9:  float | None = None
    D10: float | None = None; D11: float | None = None; D12: float | None = None
    D13: float | None = None; D14: float | None = None; D15: float | None = None

    # Match features M1-M9
    M1:  str | None = None; M2:  str | None = None; M3:  str | None = None
    M4:  str | None = None; M5:  str | None = None; M6:  str | None = None
    M7:  str | None = None; M8:  str | None = None; M9:  str | None = None

    # Vesta engineered V-features (V1-V339, most optional/nullable)
    # Include first 20 explicitly; remainder handled by feature pipeline
    V1:  float | None = None; V2:  float | None = None; V3:  float | None = None
    V4:  float | None = None; V5:  float | None = None; V6:  float | None = None
    V7:  float | None = None; V8:  float | None = None; V9:  float | None = None
    V10: float | None = None; V11: float | None = None; V12: float | None = None
    V13: float | None = None; V14: float | None = None; V15: float | None = None
    V16: float | None = None; V17: float | None = None; V18: float | None = None
    V19: float | None = None; V20: float | None = None

    # Identity features (from identity table join)
    DeviceType: str | None   = None
    DeviceInfo: str | None   = None
    id_01: float | None = None; id_02: float | None = None
    id_03: float | None = None; id_04: float | None = None
    id_05: float | None = None; id_06: float | None = None
    id_09: float | None = None; id_10: float | None = None
    id_11: float | None = None

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("TransactionAmt must be positive")
        return round(v, 2)

    model_config = {"extra": "allow"}  # Allow V21-V339 as extra fields


class BatchScoreRequest(BaseModel):
    """Batch scoring request: 1 to 1,000 transactions."""
    transactions: Annotated[
        list[TransactionFeatures],
        Field(min_length=1, max_length=1000, description="List of transactions to score"),
    ]


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SHAPContributor(BaseModel):
    """Top SHAP contributor for a single prediction."""
    feature:   str   = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value (positive = fraud signal)")
    feature_value: float | None = Field(None, description="Actual feature value for this transaction")


class ScoreResponse(BaseModel):
    """Response for a single transaction scoring."""
    TransactionID:       int     = Field(..., description="Echo of input TransactionID")
    fraud_probability:   float   = Field(..., ge=0, le=1, description="P(fraud) in [0, 1]")
    is_flagged:          bool    = Field(..., description="True if fraud_probability >= threshold")
    threshold:           float   = Field(..., description="Decision threshold in use (theta*)")
    risk_level:          str     = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    model_version:       str     = Field(..., description="Model identifier and version")
    latency_ms:          float   = Field(..., description="End-to-end scoring latency in ms")
    scored_at:           datetime = Field(..., description="UTC timestamp of scoring")
    top_contributors:    list[SHAPContributor] = Field(
        default_factory=list,
        description="Top 3 SHAP contributors driving this score",
    )

    @property
    def risk_level_computed(self) -> str:
        p = self.fraud_probability
        if p < 0.3:   return "LOW"
        if p < 0.6:   return "MEDIUM"
        if p < 0.85:  return "HIGH"
        return "CRITICAL"


class BatchScoreResponse(BaseModel):
    """Response for batch scoring."""
    results:         list[ScoreResponse] = Field(..., description="Score for each transaction")
    batch_size:      int                 = Field(..., description="Number of transactions scored")
    total_latency_ms: float              = Field(..., description="Total batch processing time")
    n_flagged:       int                 = Field(..., description="Number flagged as fraud")
    flag_rate:       float               = Field(..., description="Fraction flagged (n_flagged/batch_size)")


class HealthResponse(BaseModel):
    status:        str   = "ok"
    model_version: str
    uptime_seconds: float
    threshold:     float


class ModelInfoResponse(BaseModel):
    model_name:    str
    model_version: str
    trained_at:    str | None
    threshold:     float
    train_auprc:   float | None
    n_features:    int | None
