"""AI Act Art. 12 — automatic event logging.

Every scored application is recorded here: which model version scored it,
the probability, the threshold, the decision, and the reason codes that
justified it — before the decision is ever acted upon. In-memory here (same
declared trade-off as the other two repos in this portfolio: the interface
is what a Postgres-backed append-only table would also expose), but the
shape — append-only, keyed by a stable `decision_id`, never mutated — is
what a real deployment persists.
"""

from __future__ import annotations

from creditrisk.models import ScoringDecision


class InferenceLog:
    def __init__(self) -> None:
        self._entries: list[ScoringDecision] = []
        self._by_id: dict[str, ScoringDecision] = {}

    def record(self, decision: ScoringDecision) -> None:
        if decision.decision_id in self._by_id:
            raise ValueError(f"decision_id already recorded: {decision.decision_id}")
        self._entries.append(decision)
        self._by_id[decision.decision_id] = decision

    def get(self, decision_id: str) -> ScoringDecision | None:
        return self._by_id.get(decision_id)

    def all(self) -> list[ScoringDecision]:
        return list(self._entries)

    def for_application(self, application_id: str) -> list[ScoringDecision]:
        return [d for d in self._entries if d.application_id == application_id]

    def __len__(self) -> int:
        return len(self._entries)
