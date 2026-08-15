"""Core domain types.

Kept dependency-free (stdlib dataclasses only) so audit/fairness/registry
logic can be unit tested without pulling in scikit-learn or FastAPI — same
discipline as models.py in the other two repos of this portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum


class Decision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class CreditApplication:
    """One credit application. `protected_group` is a synthetic stand-in for
    a legally protected demographic attribute — deliberately anonymized to
    "A"/"B" rather than modeled on a real protected characteristic, since
    this is what the fairness check exercises, not what it claims to
    represent. `defaulted` is the ground-truth label, present only in
    training/backtesting data — never a feature the model sees at scoring
    time (see docs/adr/0003)."""

    application_id: str
    application_date: date
    protected_group: str
    age: int
    income_monthly: float
    employment_years: float
    existing_debt_ratio: float  # existing debt payments / income
    credit_history_score: float  # bureau-style score, 300-850
    loan_amount_requested: float
    defaulted: bool | None = None


@dataclass(frozen=True, slots=True)
class ReasonCode:
    """A single feature's contribution to a decision, in the units of the
    model's own linear score — not a post-hoc approximation. See
    docs/adr/0002 for why this is native to the glass-box model rather than
    a SHAP explanation of a black-box one."""

    feature_name: str
    contribution: float  # signed: positive increases predicted default risk
    direction: str  # "increases_risk" | "decreases_risk"


@dataclass(frozen=True, slots=True)
class ScoringDecision:
    decision_id: str
    application_id: str
    model_version: str
    probability_of_default: float
    decision_threshold: float
    decision: Decision
    reason_codes: list[ReasonCode]
    scored_at: datetime


@dataclass(frozen=True, slots=True)
class OverrideRecord:
    """AI Act Art. 14 human oversight: a decision can be overridden, but
    never silently — `reason` is mandatory and the original decision is
    preserved, not replaced, in the audit trail (docs/adr/0004)."""

    decision_id: str
    original_decision: Decision
    overridden_decision: Decision
    reason: str
    overridden_by: str
    overridden_at: datetime


@dataclass(slots=True)
class GroupMetrics:
    group: str
    n: int
    approval_rate: float
    true_positive_rate: float  # among applicants who did NOT default, share approved
    n_non_defaulters: int


@dataclass(slots=True)
class FairnessReport:
    """Demographic parity and equal opportunity across `protected_group`.
    See docs/adr/0003 for why these two metrics and not others, and for the
    honest caveat about what a two-group synthetic comparison can and
    cannot tell you."""

    model_version: str
    groups: list[GroupMetrics]
    demographic_parity_difference: float  # max - min approval rate across groups
    equal_opportunity_difference: float  # max - min TPR across groups
    metadata: dict = field(default_factory=dict)
