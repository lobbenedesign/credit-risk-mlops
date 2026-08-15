"""The black-box comparator — never used for actual decisions.

A random forest, trained on the same data through the same
`FeaturePreprocessor` as `ScorecardModel`, exists solely to answer "how much
AUC would we give up by using an interpretable model instead of a more
flexible one?" (see docs/adr/0002 and README §Numeri misurati for the
answer). It deliberately has no `reason_codes` method: a permutation-
importance or SHAP explanation of a random forest is an approximation of
what the model did, not a report of it — good enough for a data-science
review of feature importance, not for an AI Act Art. 13 explanation handed
to a rejected applicant. `ScorecardModel` is what makes decisions; this
class only ever appears in the AUC comparison in `train.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from creditrisk.model.features import FeaturePreprocessor

VERSION = "blackbox-randomforest-v1"


class BlackBoxModel:
    def __init__(self, preprocessor: FeaturePreprocessor, random_state: int = 42) -> None:
        self._preprocessor = preprocessor
        self._clf = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=random_state
        )
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> BlackBoxModel:
        X = self._preprocessor.transform(train_df)
        y = train_df["defaulted"].astype(int)
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BlackBoxModel.fit(train_df) must be called before predicting")
        X = self._preprocessor.transform(df)
        return self._clf.predict_proba(X)[:, 1]
