"""Generate the bundled sample datasets.

Usage::

    python scripts/make_sample_data.py

The datasets are intentionally *messy* (missing values, duplicates, mixed
casing, outliers, class imbalance) so every stage of the workflow has something
to demonstrate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow running as a plain script
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402


def customer_churn(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Binary classification with missing values, duplicates and mixed casing."""
    rng = np.random.default_rng(seed)
    plan = rng.choice(["basic", "standard", "premium"], size=rows, p=[0.45, 0.35, 0.20])
    tenure = np.clip(rng.gamma(2.2, 9.0, rows), 1, 72).round()
    monthly = np.clip(rng.normal(70, 25, rows), 15, 180).round(2)
    support_calls = rng.poisson(1.4, rows)
    late_payments = rng.poisson(0.8, rows)
    age = np.clip(rng.normal(41, 12, rows), 18, 85).round()
    usage_hours = np.clip(rng.normal(18, 7, rows), 0.5, 60).round(2)

    logit = (
        1.2 * (plan == "basic")
        + 0.6 * (plan == "standard")
        - 0.035 * tenure
        + 0.012 * monthly
        + 0.42 * support_calls
        + 0.55 * late_payments
        - 0.05 * usage_hours
        - 1.1
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = (rng.random(rows) < probability).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:05d}" for index in range(1, rows + 1)],
            "plan_type": plan,
            "tenure_months": tenure,
            "monthly_charge": monthly,
            "support_calls": support_calls,
            "late_payments": late_payments,
            "age": age,
            "usage_hours": usage_hours,
            "signup_date": pd.date_range("2021-01-01", periods=rows, freq="5h")[:rows].strftime("%Y-%m-%d"),
            "contract": rng.choice(["monthly", "1 year", "2 year"], size=rows, p=[0.5, 0.3, 0.2]),
            "churn": churn,
        }
    )
    # realistic messiness
    for column, share in (("monthly_charge", 0.04), ("age", 0.05), ("usage_hours", 0.07), ("contract", 0.03)):
        mask = rng.random(rows) < share
        frame.loc[mask, column] = np.nan
    frame.loc[rng.random(rows) < 0.03, "plan_type"] = " Premium "
    frame.loc[rng.random(rows) < 0.02, "age"] = -1  # invalid
    frame.loc[rng.random(rows) < 0.01, "monthly_charge"] = frame["monthly_charge"] * 6  # outliers
    frame = pd.concat([frame, frame.sample(12, random_state=seed)], ignore_index=True)  # duplicates
    return frame


def house_prices(rows: int = 900, seed: int = 7) -> pd.DataFrame:
    """Regression with skewed targets and a leaky-looking column."""
    rng = np.random.default_rng(seed)
    area = rng.normal(140, 45, rows).clip(35, 400)
    rooms = rng.integers(1, 7, rows)
    age = rng.integers(0, 80, rows)
    distance = rng.gamma(2.0, 2.5, rows).round(2)
    quality = rng.integers(1, 11, rows)
    price = (
        1800 * area
        + 22000 * rooms
        - 900 * age
        - 15000 * distance
        + 38000 * quality
        + rng.normal(0, 45000, rows)
    ).clip(30000, None)
    frame = pd.DataFrame(
        {
            "property_id": [f"P{index:05d}" for index in range(1, rows + 1)],
            "area_sqm": area.round(1),
            "rooms": rooms,
            "building_age": age,
            "distance_center_km": distance,
            "quality_score": quality,
            "city": rng.choice(["Istanbul", "Ankara", "Izmir", "Bursa"], rows, p=[0.45, 0.25, 0.2, 0.1]),
            "heating": rng.choice(["gas", "electric", "none"], rows, p=[0.6, 0.3, 0.1]),
            "sale_date": pd.date_range("2022-01-01", periods=rows, freq="8h").strftime("%Y-%m-%d"),
            "price": price.round(0),
        }
    )
    for column, share in (("area_sqm", 0.03), ("quality_score", 0.06)):
        frame.loc[rng.random(rows) < share, column] = np.nan
    frame.loc[rng.random(rows) < 0.02, "rooms"] = 0
    return frame


def retail_sales(days: int = 900, seed: int = 11) -> pd.DataFrame:
    """Daily sales with trend, weekly/annual seasonality and promotions."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    trend = np.linspace(1000, 1650, days)
    weekly = 180 * np.sin(2 * np.pi * np.arange(days) / 7)
    yearly = 260 * np.sin(2 * np.pi * np.arange(days) / 365.25 - 1.2)
    promo = (rng.random(days) < 0.08).astype(int) * rng.integers(150, 600, days)
    noise = rng.normal(0, 95, days)
    sales = (trend + weekly + yearly + promo + noise).clip(200, None)
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "store": rng.choice(["S1", "S2", "S3"], days),
            "sales": sales.round(2),
            "promotion": promo.astype(int),
            "temperature": (18 + 9 * np.sin(2 * np.pi * np.arange(days) / 365.25) + rng.normal(0, 3, days)).round(1),
            "holiday": (pd.Series(dates).dt.dayofweek >= 5).astype(int).values,
        }
    )
    frame.loc[rng.random(days) < 0.01, "sales"] = np.nan
    return frame


def main() -> None:
    settings = get_settings()
    settings.ensure_directories()
    target = Path(settings.samples_dir)
    target.mkdir(parents=True, exist_ok=True)
    datasets = {
        "customer_churn.csv": customer_churn(),
        "house_prices.csv": house_prices(),
        "retail_sales.csv": retail_sales(),
    }
    for name, frame in datasets.items():
        path = target / name
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({frame.shape[0]:,} rows x {frame.shape[1]} columns)")

    knowledge = Path(settings.knowledge_dir)
    knowledge.mkdir(parents=True, exist_ok=True)
    notes = knowledge / "business_rules.md"
    if not notes.exists():
        notes.write_text(
            "# Business rules\n\n"
            "- A customer is considered *churned* when they stop using the service for more than 60 days.\n"
            "- The retention team acts on the top 20% highest-risk customers first.\n"
            "- Monthly charge is billed in local currency; discounts are already applied.\n"
            "- Support calls are counted per resolved ticket, not per contact.\n"
            "- Data older than the last 24 months is excluded from retraining.\n",
            encoding="utf-8",
        )
        print(f"wrote {notes}")


if __name__ == "__main__":
    main()
