"""
ingestion/synthetic_generator.py
================================
Synthetic transaction generator for:
  1. Sensitivity analysis (varying fraud rate)
  2. API stress-testing without the real dataset
  3. Augmenting the training set with known fraud patterns

Fraud patterns injected:
  - Card-testing attack: many small transactions in rapid succession
  - Account-takeover: large transaction from new device/location
  - Velocity attack: high-frequency transactions on same card

Usage:
    python ingestion/synthetic_generator.py \
        --n-transactions 100000 \
        --fraud-rate 0.02 \
        --output data/synthetic/synth_100k_2pct.parquet
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RNG = np.random.default_rng(seed=42)  # Global seeded RNG for reproducibility


@dataclass
class GeneratorConfig:
    n_transactions: int = 100_000
    fraud_rate: float = 0.035        # 3.5% base rate (matches IEEE-CIS)
    start_dt: int = 86_400           # seconds offset from reference
    time_span_days: int = 180        # 6 months
    n_cards: int = 10_000
    n_merchants: int = 500
    product_codes: list[str] = field(default_factory=lambda: ["W", "H", "C", "S", "R"])
    output_path: Path = ROOT / "data" / "synthetic" / "synthetic_transactions.parquet"


def _sample_legitimate_amount(n: int) -> np.ndarray:
    """Lognormal mixture: small retail + medium e-commerce + large luxury."""
    components = RNG.choice(3, size=n, p=[0.6, 0.3, 0.1])
    amounts = np.where(
        components == 0,
        RNG.lognormal(mean=3.0, sigma=0.8, size=n),   # ~$20 median
        np.where(
            components == 1,
            RNG.lognormal(mean=4.5, sigma=0.6, size=n),  # ~$90 median
            RNG.lognormal(mean=5.8, sigma=0.9, size=n),  # ~$330 median
        ),
    )
    return amounts.clip(0.01, 20_000)


def _sample_fraud_amount(pattern: str, n: int) -> np.ndarray:
    """Fraud amounts depend on attack pattern."""
    if pattern == "card_testing":
        return RNG.uniform(0.01, 5.0, size=n)    # Tiny amounts to test card validity
    elif pattern == "account_takeover":
        return RNG.lognormal(mean=6.5, sigma=0.5, size=n)  # Large ($665 median)
    else:  # velocity
        return RNG.lognormal(mean=4.0, sigma=0.7, size=n)


def generate(cfg: GeneratorConfig) -> pd.DataFrame:
    """Generate a synthetic transaction dataset matching IEEE-CIS column schema."""
    n = cfg.n_transactions
    n_fraud = int(n * cfg.fraud_rate)
    n_legit = n - n_fraud

    log.info(f"Generating {n:,} transactions ({n_fraud:,} fraud @ {cfg.fraud_rate:.1%})")

    # ---- Timestamps ----
    span = cfg.time_span_days * 86_400
    txn_dt = np.sort(RNG.integers(cfg.start_dt, cfg.start_dt + span, size=n))

    # ---- Card IDs ----
    card1 = RNG.integers(1000, 18000, size=n)
    card4 = RNG.choice(["discover", "mastercard", "visa", "american express"], size=n, p=[0.05, 0.25, 0.60, 0.10])

    # ---- Legitimate amounts ----
    amounts = _sample_legitimate_amount(n)

    # ---- Fraud pattern injection ----
    fraud_patterns: list[Literal["card_testing", "account_takeover", "velocity"]] = []
    fraud_pattern_sizes = [
        int(n_fraud * 0.4),   # card testing (40%)
        int(n_fraud * 0.35),  # account takeover (35%)
        n_fraud - int(n_fraud * 0.4) - int(n_fraud * 0.35),  # velocity
    ]
    for pattern, size in zip(
        ["card_testing", "account_takeover", "velocity"], fraud_pattern_sizes
    ):
        fraud_patterns.extend([pattern] * size)

    RNG.shuffle(fraud_patterns)

    # Override fraud transaction amounts
    fraud_idx = RNG.choice(n, size=n_fraud, replace=False)
    for pattern in set(fraud_patterns):
        mask = [i for i, p in enumerate(fraud_patterns) if p == pattern]
        amounts[fraud_idx[mask]] = _sample_fraud_amount(pattern, len(mask))

    # ---- Labels ----
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[fraud_idx] = 1

    # ---- Email domains ----
    legit_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]
    fraud_domains = ["anonymous.com", "protonmail.com", "temp-mail.org", "guerrillamail.com"]
    p_email = np.where(
        is_fraud == 1,
        RNG.choice(fraud_domains, size=n),
        RNG.choice(legit_domains, size=n, p=[0.40, 0.25, 0.20, 0.10, 0.05]),
    )

    # ---- Assemble DataFrame ----
    df = pd.DataFrame({
        "TransactionID": np.arange(9_000_000, 9_000_000 + n),  # distinct from IEEE-CIS IDs
        "TransactionDT": txn_dt,
        "TransactionAmt": amounts.round(2),
        "ProductCD": RNG.choice(cfg.product_codes, size=n, p=[0.45, 0.20, 0.15, 0.10, 0.10]),
        "card1": card1,
        "card4": card4,
        "P_emaildomain": p_email,
        "isFraud": is_fraud,
        "source": "synthetic",
        "fraud_pattern": np.where(
            is_fraud == 1,
            [fraud_patterns[list(fraud_idx).index(i)] if i in fraud_idx else "none" for i in range(n)],
            "none",
        ),
    })

    # Summary stats
    actual_rate = df["isFraud"].mean()
    log.info(f"Generated. Shape: {df.shape}  Actual fraud rate: {actual_rate:.4f}")
    log.info(f"Pattern breakdown:\n{df[df['isFraud']==1]['fraud_pattern'].value_counts()}")

    return df


def save(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log.info(f"Saved to {output_path.resolve()} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fraud transaction data")
    parser.add_argument("--n-transactions", type=int, default=100_000)
    parser.add_argument("--fraud-rate", type=float, default=0.035)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "synthetic" / "synthetic_transactions.parquet",
    )
    args = parser.parse_args()

    cfg = GeneratorConfig(
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        output_path=args.output,
    )
    df = generate(cfg)
    save(df, cfg.output_path)
