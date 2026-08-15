"""Training + AUC comparison entrypoint used by scripts/demo.py and tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

from creditrisk.model.blackbox import VERSION as BLACKBOX_VERSION
from creditrisk.model.blackbox import BlackBoxModel
from creditrisk.model.features import FeaturePreprocessor
from creditrisk.model.scorecard import VERSION as SCORECARD_VERSION
from creditrisk.model.scorecard import ScorecardModel
from creditrisk.model.split import OutOfTimeSplit, split_out_of_time


@dataclass(frozen=True, slots=True)
class TrainingResult:
    split: OutOfTimeSplit
    preprocessor: FeaturePreprocessor
    scorecard: ScorecardModel
    blackbox: BlackBoxModel
    scorecard_auc: float
    blackbox_auc: float

    @property
    def auc_gap(self) -> float:
        """How much AUC the black-box model gains over the interpretable
        one — the number that makes "we chose explainability" a measured
        trade-off instead of an assertion."""
        return self.blackbox_auc - self.scorecard_auc


def train_and_evaluate(df: pd.DataFrame, test_fraction: float = 0.25) -> TrainingResult:
    split = split_out_of_time(df, test_fraction=test_fraction)

    preprocessor = FeaturePreprocessor().fit(split.train_df)

    scorecard = ScorecardModel(preprocessor).fit(split.train_df)
    blackbox = BlackBoxModel(preprocessor).fit(split.train_df)

    y_test = split.test_df["defaulted"].astype(int)
    scorecard_auc = roc_auc_score(y_test, scorecard.predict_proba(split.test_df))
    blackbox_auc = roc_auc_score(y_test, blackbox.predict_proba(split.test_df))

    return TrainingResult(
        split=split,
        preprocessor=preprocessor,
        scorecard=scorecard,
        blackbox=blackbox,
        scorecard_auc=float(scorecard_auc),
        blackbox_auc=float(blackbox_auc),
    )


__all__ = ["TrainingResult", "train_and_evaluate", "SCORECARD_VERSION", "BLACKBOX_VERSION"]
