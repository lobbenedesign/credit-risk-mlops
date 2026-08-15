"""Feature preprocessing shared by both models.

A single `StandardScaler` fit on the training split only — reused by both
the scorecard and the black-box comparator, so an AUC difference between
them reflects the model, not two different preprocessing pipelines. Fitting
on the full dataset (train + test) would leak test-set statistics into
training, the same class of bug an out-of-time split exists to prevent
(docs/adr/0001) — fitting only on `train_df` here is the other half of that
discipline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from creditrisk.data.synthetic import FEATURE_COLUMNS
from creditrisk.models import CreditApplication


def application_to_frame(application: CreditApplication) -> pd.DataFrame:
    """Single-row DataFrame with exactly the model's feature columns, for
    scoring one application through the same preprocessing/model path used
    for batch training and evaluation."""
    return pd.DataFrame(
        [
            {
                "age": application.age,
                "income_monthly": application.income_monthly,
                "employment_years": application.employment_years,
                "existing_debt_ratio": application.existing_debt_ratio,
                "credit_history_score": application.credit_history_score,
                "loan_amount_requested": application.loan_amount_requested,
            }
        ]
    )


class FeaturePreprocessor:
    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> FeaturePreprocessor:
        self._scaler.fit(train_df[FEATURE_COLUMNS])
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeaturePreprocessor.fit(train_df) must be called before transform()")
        scaled = self._scaler.transform(df[FEATURE_COLUMNS])
        return pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=df.index)

    @property
    def feature_columns(self) -> list[str]:
        return list(FEATURE_COLUMNS)
