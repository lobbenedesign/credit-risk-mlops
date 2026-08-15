"""AI Act Art. 14 — human oversight.

A model's decision can always be overridden by a human reviewer, but never
silently: `reason` is mandatory (empty or whitespace-only is rejected before
it reaches the audit trail) and the original decision is preserved, not
replaced — `OverrideRecord` links back to the original `decision_id` rather
than mutating it, so "what did the model decide" and "what actually
happened" both remain answerable from the log, forever. Same append-only
discipline as `InferenceLog`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from creditrisk.audit.inference_log import InferenceLog
from creditrisk.models import Decision, OverrideRecord


class HumanOversightLog:
    def __init__(self, inference_log: InferenceLog) -> None:
        self._inference_log = inference_log
        self._overrides: list[OverrideRecord] = []

    def override(
        self,
        decision_id: str,
        overridden_decision: Decision,
        reason: str,
        overridden_by: str,
    ) -> OverrideRecord:
        if not reason or not reason.strip():
            raise ValueError("A reason is mandatory to override a decision (AI Act Art. 14)")
        if not overridden_by or not overridden_by.strip():
            raise ValueError("The reviewer identity is mandatory to override a decision")

        original = self._inference_log.get(decision_id)
        if original is None:
            raise ValueError(f"Unknown decision_id: {decision_id}")

        record = OverrideRecord(
            decision_id=decision_id,
            original_decision=original.decision,
            overridden_decision=overridden_decision,
            reason=reason.strip(),
            overridden_by=overridden_by.strip(),
            overridden_at=datetime.now(UTC),
        )
        self._overrides.append(record)
        return record

    def overrides_for(self, decision_id: str) -> list[OverrideRecord]:
        return [r for r in self._overrides if r.decision_id == decision_id]

    def all(self) -> list[OverrideRecord]:
        return list(self._overrides)


def new_decision_id() -> str:
    return str(uuid.uuid4())
