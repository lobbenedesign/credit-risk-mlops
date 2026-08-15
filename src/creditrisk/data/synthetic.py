"""Synthetic credit application generator.

No real credit bureau or banking data is used anywhere in this project —
same discipline as the synthetic generators in the other two repos of this
portfolio. The generation process is designed to produce a realistic and
genuinely useful fairness test case, not a strawman:

- True default risk depends on `existing_debt_ratio`, `income_to_loan_ratio`
  and `employment_years`, plus a `true_creditworthiness` latent factor — and
  is IDENTICAL in distribution across `protected_group`. Group membership
  has zero causal effect on true default risk, by construction.
- `credit_history_score` (the bureau-style score a real scorecard would use)
  is partly driven by that same latent creditworthiness (it IS predictive,
  not pure noise) but also carries a systematic offset for group "B" —
  modeling the well-documented "thin file" effect where newer entrants to a
  credit system (younger, immigrants, ...) score lower on traditional
  bureau metrics for reasons unrelated to their actual repayment behavior.

This is what makes the fairness report in docs/adr/0003 a genuine finding
rather than a foregone conclusion: a model trained on this data has a real,
specific reason to produce disparate outcomes across groups even though the
ground truth default rate is equal — the exact "proxy discrimination"
scenario credit-scoring fairness regulation exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "income_monthly",
    "employment_years",
    "existing_debt_ratio",
    "credit_history_score",
    "loan_amount_requested",
]

GROUP_B_CREDIT_HISTORY_PENALTY = 70.0  # points, on a 300-850 scale — the "thin file" offset


@dataclass(frozen=True, slots=True)
class SyntheticCreditDataset:
    applications: pd.DataFrame  # one row per application, includes `defaulted` ground truth
    feature_columns: list[str]


def _clip(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(values, lo, hi)


def generate_synthetic_applications(
    n: int = 3000,
    months: int = 24,
    seed: int = 42,
    start: date | None = None,
) -> SyntheticCreditDataset:
    rng = np.random.default_rng(seed)
    start = start or date.today() - timedelta(days=30 * months)

    protected_group = rng.choice(["A", "B"], size=n, p=[0.7, 0.3])
    is_group_b = protected_group == "B"

    age = _clip(rng.normal(40, 12, size=n), 18, 75)
    employment_years = _clip((age - 18) * rng.uniform(0.1, 0.6, size=n), 0, 40)
    income_monthly = _clip(rng.lognormal(mean=7.5, sigma=0.4, size=n), 900, 12000)

    true_creditworthiness = rng.normal(0, 1, size=n)  # latent, identical distribution both groups
    existing_debt_ratio = _clip(
        0.35 - 0.10 * true_creditworthiness + rng.normal(0, 0.10, size=n), 0.0, 1.2
    )
    loan_amount_requested = _clip(
        income_monthly * rng.uniform(2, 10, size=n) * (1 - 0.15 * true_creditworthiness), 1000, 100_000
    )

    credit_history_score = _clip(
        600
        + 80 * true_creditworthiness
        + rng.normal(0, 40, size=n)
        - GROUP_B_CREDIT_HISTORY_PENALTY * is_group_b,
        300,
        850,
    )

    income_to_loan_ratio = income_monthly * 12 / loan_amount_requested

    # true default probability: depends only on debt ratio, income/loan
    # ratio, employment and the latent creditworthiness factor — NEVER on
    # protected_group and never directly on credit_history_score (which is
    # itself only a noisy, group-biased proxy for the same latent factor)
    z = (
        -3.0  # intercept calibrated so the base default rate lands around 20%,
        # in the range a real (if somewhat risk-heavy) consumer credit book
        # would show — see docs/adr/0003 for the calibration numbers.
        + 2.2 * (existing_debt_ratio - 0.35) / 0.20
        - 1.5 * (income_to_loan_ratio - np.mean(income_to_loan_ratio)) / np.std(income_to_loan_ratio)
        - 0.8 * (employment_years - np.mean(employment_years)) / np.std(employment_years)
        - 1.6 * true_creditworthiness
        + rng.normal(0, 0.6, size=n)
    )
    default_probability = 1 / (1 + np.exp(-z))
    defaulted = rng.uniform(size=n) < default_probability

    application_date = [
        start + timedelta(days=int(d)) for d in rng.uniform(0, months * 30, size=n)
    ]

    df = pd.DataFrame(
        {
            "application_id": [f"app-{i:05d}" for i in range(n)],
            "application_date": application_date,
            "protected_group": protected_group,
            "age": age.round(0).astype(int),
            "income_monthly": income_monthly.round(2),
            "employment_years": employment_years.round(1),
            "existing_debt_ratio": existing_debt_ratio.round(3),
            "credit_history_score": credit_history_score.round(0).astype(int),
            "loan_amount_requested": loan_amount_requested.round(2),
            "defaulted": defaulted,
        }
    )
    df = df.sort_values("application_date").reset_index(drop=True)

    return SyntheticCreditDataset(applications=df, feature_columns=FEATURE_COLUMNS)
