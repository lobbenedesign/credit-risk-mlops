import numpy as np
import pandas as pd

from creditrisk.fairness.fairness_report import compute_fairness_report


def _df(groups, defaulted):
    return pd.DataFrame({"protected_group": groups, "defaulted": defaulted})


class TestComputeFairnessReport:
    def test_equal_approval_rates_yield_zero_demographic_parity_difference(self):
        df = _df(["A", "A", "B", "B"], [False, False, False, False])
        probs = np.array([0.1, 0.1, 0.1, 0.1])  # everyone approved, same rate both groups
        report = compute_fairness_report("v1", df, probs, threshold=0.5)
        assert report.demographic_parity_difference == 0.0

    def test_detects_a_real_approval_rate_gap(self):
        df = _df(["A", "A", "B", "B"], [False, False, False, False])
        probs = np.array([0.1, 0.1, 0.9, 0.9])  # group A approved, group B denied
        report = compute_fairness_report("v1", df, probs, threshold=0.5)
        assert report.demographic_parity_difference == 1.0

    def test_equal_opportunity_only_considers_non_defaulters(self):
        # group B's only non-defaulter is denied; its defaulter being denied
        # too should not affect the metric (denying an actual defaulter is
        # the model doing its job, not a fairness problem)
        df = _df(["A", "A", "B", "B"], [False, False, False, True])
        probs = np.array([0.1, 0.1, 0.9, 0.9])
        report = compute_fairness_report("v1", df, probs, threshold=0.5)
        groups = {g.group: g for g in report.groups}
        assert groups["B"].n_non_defaulters == 1
        assert groups["B"].true_positive_rate == 0.0

    def test_groups_are_reported_with_correct_counts(self):
        df = _df(["A", "A", "A", "B"], [False, True, False, False])
        probs = np.array([0.1, 0.9, 0.2, 0.3])
        report = compute_fairness_report("v1", df, probs, threshold=0.5)
        groups = {g.group: g for g in report.groups}
        assert groups["A"].n == 3
        assert groups["B"].n == 1

    def test_metadata_carries_threshold_and_total_n(self):
        df = _df(["A", "B"], [False, False])
        probs = np.array([0.1, 0.2])
        report = compute_fairness_report("v1", df, probs, threshold=0.4)
        assert report.metadata["threshold"] == 0.4
        assert report.metadata["n_total"] == 2

    def test_real_dataset_shows_the_designed_disparate_impact(self):
        # integration-level regression guard for the finding documented in
        # README §Numeri misurati: this is not a toy case, it is the actual
        # trained scorecard on the actual synthetic dataset
        from creditrisk.data.synthetic import generate_synthetic_applications
        from creditrisk.model.train import train_and_evaluate

        ds = generate_synthetic_applications(n=3000, months=24, seed=42)
        result = train_and_evaluate(ds.applications)
        probs = result.scorecard.predict_proba(result.split.test_df)
        report = compute_fairness_report("scorecard", result.split.test_df, probs, threshold=0.5)

        assert report.demographic_parity_difference > 0.05
        groups = {g.group: g for g in report.groups}
        assert groups["A"].approval_rate > groups["B"].approval_rate
