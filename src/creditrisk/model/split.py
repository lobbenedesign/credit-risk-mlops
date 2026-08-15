"""Out-of-time train/test split.

A random row-level split would let the model "see the future" relative to
some training examples — two applications from the same week, one in train
and one in test, is not what a scorecard faces in production, where every
scored application is strictly *after* the data it was trained on. An
out-of-time split (train on the earlier period, test on the later one) is
the standard backtesting discipline for exactly this reason, not an
arbitrary choice — see docs/adr/0001.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class OutOfTimeSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cutoff_date: object


def split_out_of_time(df: pd.DataFrame, test_fraction: float = 0.25) -> OutOfTimeSplit:
    """Sorts by `application_date` and takes the most recent `test_fraction`
    of rows as the test set — the cutoff date is derived from the data, not
    hardcoded, so this works for any date range passed in."""
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    sorted_df = df.sort_values("application_date").reset_index(drop=True)
    split_index = int(len(sorted_df) * (1 - test_fraction))
    cutoff_date = sorted_df.iloc[split_index]["application_date"]

    train_df = sorted_df.iloc[:split_index].reset_index(drop=True)
    test_df = sorted_df.iloc[split_index:].reset_index(drop=True)
    return OutOfTimeSplit(train_df=train_df, test_df=test_df, cutoff_date=cutoff_date)
