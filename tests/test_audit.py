from datetime import UTC, datetime

import pytest

from creditrisk.audit.inference_log import InferenceLog
from creditrisk.audit.oversight import HumanOversightLog, new_decision_id
from creditrisk.models import Decision, ReasonCode, ScoringDecision


def _decision(decision_id: str | None = None, decision: Decision = Decision.DENY) -> ScoringDecision:
    return ScoringDecision(
        decision_id=decision_id or new_decision_id(),
        application_id="app-1",
        model_version="scorecard-logreg-v1",
        probability_of_default=0.8,
        decision_threshold=0.5,
        decision=decision,
        reason_codes=[
            ReasonCode(feature_name="existing_debt_ratio", contribution=1.2, direction="increases_risk")
        ],
        scored_at=datetime.now(UTC),
    )


class TestInferenceLog:
    def test_recorded_decision_is_retrievable_by_id(self):
        log = InferenceLog()
        decision = _decision()
        log.record(decision)
        assert log.get(decision.decision_id) is decision

    def test_unknown_decision_id_returns_none(self):
        log = InferenceLog()
        assert log.get("nonexistent") is None

    def test_recording_the_same_decision_id_twice_raises(self):
        log = InferenceLog()
        decision = _decision(decision_id="dup")
        log.record(decision)
        with pytest.raises(ValueError):
            log.record(_decision(decision_id="dup"))

    def test_len_reflects_number_of_recorded_decisions(self):
        log = InferenceLog()
        log.record(_decision())
        log.record(_decision())
        assert len(log) == 2

    def test_for_application_filters_by_application_id(self):
        log = InferenceLog()
        d1 = _decision()
        log.record(d1)
        assert [d.decision_id for d in log.for_application("app-1")] == [d1.decision_id]
        assert log.for_application("unknown-app") == []


class TestHumanOversightLog:
    def test_override_requires_a_non_empty_reason(self):
        log = InferenceLog()
        decision = _decision()
        log.record(decision)
        oversight = HumanOversightLog(log)
        with pytest.raises(ValueError, match="reason"):
            oversight.override(decision.decision_id, Decision.APPROVE, reason="   ", overridden_by="reviewer")

    def test_override_requires_a_reviewer_identity(self):
        log = InferenceLog()
        decision = _decision()
        log.record(decision)
        oversight = HumanOversightLog(log)
        with pytest.raises(ValueError, match="reviewer"):
            oversight.override(
                decision.decision_id, Decision.APPROVE, reason="valid reason", overridden_by=""
            )

    def test_override_of_unknown_decision_id_raises(self):
        log = InferenceLog()
        oversight = HumanOversightLog(log)
        with pytest.raises(ValueError, match="Unknown decision_id"):
            oversight.override("nonexistent", Decision.APPROVE, reason="x", overridden_by="reviewer")

    def test_successful_override_preserves_the_original_decision(self):
        log = InferenceLog()
        decision = _decision(decision=Decision.DENY)
        log.record(decision)
        oversight = HumanOversightLog(log)

        record = oversight.override(
            decision.decision_id, Decision.APPROVE, reason="Garante verificato.", overridden_by="reviewer"
        )
        assert record.original_decision == Decision.DENY
        assert record.overridden_decision == Decision.APPROVE
        # the original decision in the inference log is untouched
        assert log.get(decision.decision_id).decision == Decision.DENY

    def test_overrides_for_returns_only_that_decisions_overrides(self):
        log = InferenceLog()
        d1, d2 = _decision(), _decision()
        log.record(d1)
        log.record(d2)
        oversight = HumanOversightLog(log)
        oversight.override(d1.decision_id, Decision.APPROVE, reason="r", overridden_by="reviewer")

        assert len(oversight.overrides_for(d1.decision_id)) == 1
        assert oversight.overrides_for(d2.decision_id) == []
        assert len(oversight.all()) == 1
