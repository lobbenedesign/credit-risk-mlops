"""The interpretable model actually used for decisioning.

A standardized-coefficient logistic regression. Reason codes are computed as
`coefficient × standardized_feature_value` for the specific application
being scored — an exact decomposition of the model's own linear score, not
a post-hoc approximation of a different, more complex model. See
docs/adr/0002 for why this is preferred over SHAP-on-a-black-box for the
decision-making model itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from creditrisk.model.features import FeaturePreprocessor, application_to_frame
from creditrisk.models import CreditApplication, ReasonCode

VERSION = "scorecard-logreg-v1"
DEFAULT_TOP_N_REASON_CODES = 3


class ScorecardModel:
    def __init__(self, preprocessor: FeaturePreprocessor) -> None:
        self._preprocessor = preprocessor
        self._clf = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> ScorecardModel:
        X = self._preprocessor.transform(train_df)
        y = train_df["defaulted"].astype(int)
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        X = self._preprocessor.transform(df)
        return self._clf.predict_proba(X)[:, 1]

    def predict_proba_one(self, application: CreditApplication) -> float:
        return float(self.predict_proba(application_to_frame(application))[0])

    def reason_codes(
        self, application: CreditApplication, top_n: int = DEFAULT_TOP_N_REASON_CODES
    ) -> list[ReasonCode]:
        """Top-`top_n` features by |contribution| to this specific
        application's score, positive contribution meaning it pushed the
        predicted default probability up."""
        self._require_fitted()
        X = self._preprocessor.transform(application_to_frame(application))
        standardized_values = X.iloc[0].to_numpy()
        coefficients = self._clf.coef_[0]
        contributions = standardized_values * coefficients

        order = np.argsort(-np.abs(contributions))[:top_n]
        feature_names = list(X.columns)
        return [
            ReasonCode(
                feature_name=feature_names[i],
                contribution=round(float(contributions[i]), 4),
                direction="increases_risk" if contributions[i] > 0 else "decreases_risk",
            )
            for i in order
        ]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ScorecardModel.fit(train_df) must be called before predicting")
