"""Fairness metrics across `protected_group`, computed directly rather than
via the `fairlearn` package.

Two metrics, not one, for the same reason `docrag`'s eval harness in the RAG
repo of this portfolio keeps retrieval accuracy and refusal rate separate:
they answer different questions and collapsing them hides which one a given
model actually fails.

- **Demographic parity difference**: does the model approve both groups at
  the same rate, full stop? Simple, but blind to whether the difference is
  "justified" by different true risk — which here, by construction of the
  synthetic data (docs/adr/0003), it is not: true default risk is identical
  across groups, so any approval-rate gap is a genuine fairness signal, not
  a reflection of real risk difference.
- **Equal opportunity difference**: among applicants who would NOT actually
  default (the ones a fair model should approve), is the approval rate the
  same across groups? This is the metric that isolates *missed qualified
  applicants* specifically, rather than the overall approval-rate gap,
  which can also be driven by a genuine difference in the pool's risk mix.

No external fairness library is used: the two metrics are simple rate
comparisons, easy to audit by reading the four lines of pandas below —
consistent with this portfolio's preference for auditable code over an
opaque dependency where the underlying computation is not itself complex
(same reasoning as skipping `fairlearn` and `shap` here for a two-group,
two-metric report).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisk.models import FairnessReport, GroupMetrics


def compute_fairness_report(
    model_version: str,
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> FairnessReport:
    df = test_df.copy()
    df["probability_of_default"] = probabilities
    df["approved"] = df["probability_of_default"] < threshold

    groups: list[GroupMetrics] = []
    for group_name, group_df in df.groupby("protected_group", sort=True):
        non_defaulters = group_df[~group_df["defaulted"]]
        tpr = float(non_defaulters["approved"].mean()) if len(non_defaulters) else float("nan")
        groups.append(
            GroupMetrics(
                group=str(group_name),
                n=len(group_df),
                approval_rate=float(group_df["approved"].mean()),
                true_positive_rate=tpr,
                n_non_defaulters=len(non_defaulters),
            )
        )

    approval_rates = [g.approval_rate for g in groups]
    tprs = [g.true_positive_rate for g in groups]

    return FairnessReport(
        model_version=model_version,
        groups=groups,
        demographic_parity_difference=round(max(approval_rates) - min(approval_rates), 4),
        equal_opportunity_difference=round(max(tprs) - min(tprs), 4),
        metadata={"threshold": threshold, "n_total": len(df)},
    )
