import math
from dataclasses import replace

import pytest

from creditrisk.data.synthetic import generate_synthetic_applications
from creditrisk.explain.dossier import DossierGenerationError, generate_dossier
from creditrisk.fairness.fairness_report import compute_fairness_report
from creditrisk.model.train import train_and_evaluate


def _result_and_fairness():
    ds = generate_synthetic_applications(n=1000, months=24, seed=42)
    result = train_and_evaluate(ds.applications)
    fairness = compute_fairness_report(
        "scorecard-logreg-v1", result.split.test_df,
        result.scorecard.predict_proba(result.split.test_df), threshold=0.5,
    )
    return result, fairness


class TestGenerateDossier:
    def test_produces_a_dossier_from_complete_artifacts(self):
        result, fairness = _result_and_fairness()
        dossier = generate_dossier("scorecard-logreg-v1", result, fairness)
        assert dossier.model_version == "scorecard-logreg-v1"
        assert dossier.fairness is fairness

    def test_markdown_includes_every_required_section(self):
        result, fairness = _result_and_fairness()
        dossier = generate_dossier("scorecard-logreg-v1", result, fairness)
        markdown = dossier.to_markdown()
        for section in ("## Dataset", "## Performance", "## Fairness", "## Limiti noti"):
            assert section in markdown

    def test_known_limitations_mention_the_measured_fairness_gap(self):
        result, fairness = _result_and_fairness()
        dossier = generate_dossier("scorecard-logreg-v1", result, fairness)
        assert any("parità demografica" in limitation for limitation in dossier.known_limitations)

    def test_raises_when_fairness_report_has_no_groups(self):
        result, fairness = _result_and_fairness()
        empty_fairness = replace(fairness, groups=[])
        with pytest.raises(DossierGenerationError, match="no groups"):
            generate_dossier("scorecard-logreg-v1", result, empty_fairness)

    def test_raises_when_demographic_parity_difference_is_nan(self):
        result, fairness = _result_and_fairness()
        broken_fairness = replace(fairness, demographic_parity_difference=float("nan"))
        with pytest.raises(DossierGenerationError, match="NaN"):
            generate_dossier("scorecard-logreg-v1", result, broken_fairness)

    def test_raises_when_auc_is_nan(self):
        result, fairness = _result_and_fairness()
        broken_result = replace(result, scorecard_auc=float("nan"))
        with pytest.raises(DossierGenerationError, match="AUC"):
            generate_dossier("scorecard-logreg-v1", broken_result, fairness)

    def test_raises_when_test_split_is_empty(self):
        result, fairness = _result_and_fairness()
        empty_split = replace(result.split, test_df=result.split.test_df.iloc[0:0])
        broken_result = replace(result, split=empty_split)
        with pytest.raises(DossierGenerationError, match="empty"):
            generate_dossier("scorecard-logreg-v1", broken_result, fairness)

    def test_performance_section_reports_the_auc_gap_correctly(self):
        result, fairness = _result_and_fairness()
        dossier = generate_dossier("scorecard-logreg-v1", result, fairness)
        expected_gap = round(result.blackbox_auc - result.scorecard_auc, 4)
        assert dossier.performance["auc_gap_blackbox_minus_scorecard"] == expected_gap
        assert not math.isnan(dossier.performance["scorecard_auc"])
