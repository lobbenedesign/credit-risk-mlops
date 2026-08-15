from datetime import date

import pytest

from creditrisk.data.synthetic import generate_synthetic_applications
from creditrisk.model.train import train_and_evaluate
from creditrisk.models import CreditApplication


def _trained():
    ds = generate_synthetic_applications(n=2000, months=24, seed=42)
    return train_and_evaluate(ds.applications)


class TestTrainAndEvaluate:
    def test_both_models_achieve_reasonable_discrimination(self):
        # regression guard: this dataset has strong, designed signal — an
        # AUC collapse here means training or the data generator broke
        result = _trained()
        assert result.scorecard_auc > 0.80
        assert result.blackbox_auc > 0.80

    def test_the_interpretable_model_is_not_meaningfully_worse(self):
        # the actual finding documented in the README: choosing the
        # explainable model costs (almost) nothing in this problem
        result = _trained()
        assert result.auc_gap < 0.03

    def test_out_of_time_split_is_used_not_a_random_one(self):
        result = _trained()
        train_max = result.split.train_df["application_date"].max()
        test_min = result.split.test_df["application_date"].min()
        assert train_max <= test_min


class TestScorecardModel:
    def test_predict_proba_one_returns_a_probability(self):
        result = _trained()
        application = CreditApplication(
            application_id="t1", application_date=date.today(), protected_group="A",
            age=40, income_monthly=2000, employment_years=10, existing_debt_ratio=0.2,
            credit_history_score=700, loan_amount_requested=10000,
        )
        p = result.scorecard.predict_proba_one(application)
        assert 0.0 <= p <= 1.0

    def test_a_low_risk_and_a_high_risk_profile_are_scored_very_differently(self):
        result = _trained()
        low_risk = CreditApplication(
            application_id="low", application_date=date.today(), protected_group="A",
            age=45, income_monthly=4000, employment_years=15, existing_debt_ratio=0.10,
            credit_history_score=780, loan_amount_requested=8000,
        )
        high_risk = CreditApplication(
            application_id="high", application_date=date.today(), protected_group="A",
            age=25, income_monthly=1100, employment_years=0.5, existing_debt_ratio=0.85,
            credit_history_score=380, loan_amount_requested=40000,
        )
        p_low = result.scorecard.predict_proba_one(low_risk)
        p_high = result.scorecard.predict_proba_one(high_risk)
        assert p_high > p_low + 0.3

    def test_reason_codes_are_returned_sorted_by_absolute_contribution(self):
        result = _trained()
        application = CreditApplication(
            application_id="t2", application_date=date.today(), protected_group="B",
            age=29, income_monthly=1400, employment_years=1.5, existing_debt_ratio=0.55,
            credit_history_score=490, loan_amount_requested=18000,
        )
        codes = result.scorecard.reason_codes(application, top_n=3)
        assert len(codes) == 3
        magnitudes = [abs(c.contribution) for c in codes]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_reason_code_direction_matches_the_sign_of_its_contribution(self):
        result = _trained()
        application = CreditApplication(
            application_id="t3", application_date=date.today(), protected_group="A",
            age=40, income_monthly=2000, employment_years=10, existing_debt_ratio=0.2,
            credit_history_score=700, loan_amount_requested=10000,
        )
        for code in result.scorecard.reason_codes(application):
            if code.contribution > 0:
                assert code.direction == "increases_risk"
            else:
                assert code.direction == "decreases_risk"

    def test_predicting_before_fit_raises(self):
        from creditrisk.model.features import FeaturePreprocessor
        from creditrisk.model.scorecard import ScorecardModel

        model = ScorecardModel(FeaturePreprocessor())
        application = CreditApplication(
            application_id="t4", application_date=date.today(), protected_group="A",
            age=40, income_monthly=2000, employment_years=10, existing_debt_ratio=0.2,
            credit_history_score=700, loan_amount_requested=10000,
        )
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_proba_one(application)
