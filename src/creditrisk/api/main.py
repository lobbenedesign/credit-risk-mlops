"""FastAPI service exposing the credit scoring pipeline over HTTP.

`GET /` — analyst console (static HTML/JS, no build step).
`GET /healthz` — liveness plus which model version is in production.
`POST /score` — score one application, returns decision + reason codes;
  every call is recorded in the Art. 12 inference log before it returns.
`POST /override` — Art. 14 human oversight: override a prior decision, with
  a mandatory reason.
`GET /dossier` — the Art. 11 technical dossier, regenerated from the live
  training result and fairness report on every request (never stale).
`GET /fairness` — the fairness report on its own, for a dashboard.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from creditrisk.audit.inference_log import InferenceLog
from creditrisk.audit.oversight import HumanOversightLog, new_decision_id
from creditrisk.data.synthetic import generate_synthetic_applications
from creditrisk.explain.dossier import generate_dossier
from creditrisk.fairness.fairness_report import compute_fairness_report
from creditrisk.model.train import train_and_evaluate
from creditrisk.models import CreditApplication, Decision, ScoringDecision
from creditrisk.registry.model_registry import ModelRegistry, ModelStage

logger = logging.getLogger("creditrisk")

app = FastAPI(title="Credit Risk MLOps", version="0.1.0")

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error"})


DECISION_THRESHOLD = 0.5

_dataset = generate_synthetic_applications()
_training_result = train_and_evaluate(_dataset.applications)
_fairness_report = compute_fairness_report(
    "scorecard-logreg-v1",
    _training_result.split.test_df,
    _training_result.scorecard.predict_proba(_training_result.split.test_df),
    threshold=DECISION_THRESHOLD,
)

_registry = ModelRegistry()
_registry.register("scorecard-logreg-v1", metrics={"auc": _training_result.scorecard_auc})
_registry.promote("scorecard-logreg-v1", ModelStage.SHADOW)
_registry.promote("scorecard-logreg-v1", ModelStage.PROD)

_inference_log = InferenceLog()
_oversight_log = HumanOversightLog(_inference_log)


class ScoreRequest(BaseModel):
    application_id: str
    protected_group: str = Field(description="Tracked for fairness monitoring only — never a model feature")
    age: int
    income_monthly: float
    employment_years: float
    existing_debt_ratio: float
    credit_history_score: float
    loan_amount_requested: float


class ReasonCodeView(BaseModel):
    feature_name: str
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    decision_id: str
    application_id: str
    model_version: str
    probability_of_default: float
    decision: str
    reason_codes: list[ReasonCodeView]


class OverrideRequest(BaseModel):
    overridden_decision: str
    reason: str
    overridden_by: str


class OverrideResponse(BaseModel):
    decision_id: str
    original_decision: str
    overridden_decision: str
    reason: str
    overridden_by: str


@app.get("/healthz")
def healthz() -> dict:
    prod = _registry.current_production()
    return {
        "status": "ok",
        "production_model_version": prod.version if prod else None,
        "production_model_stage": prod.stage.value if prod else None,
        "inference_log_size": len(_inference_log),
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    application = CreditApplication(
        application_id=request.application_id,
        application_date=date.today(),
        protected_group=request.protected_group,
        age=request.age,
        income_monthly=request.income_monthly,
        employment_years=request.employment_years,
        existing_debt_ratio=request.existing_debt_ratio,
        credit_history_score=request.credit_history_score,
        loan_amount_requested=request.loan_amount_requested,
    )

    probability = _training_result.scorecard.predict_proba_one(application)
    decision = Decision.APPROVE if probability < DECISION_THRESHOLD else Decision.DENY
    reason_codes = _training_result.scorecard.reason_codes(application)

    scoring_decision = ScoringDecision(
        decision_id=new_decision_id(),
        application_id=application.application_id,
        model_version="scorecard-logreg-v1",
        probability_of_default=probability,
        decision_threshold=DECISION_THRESHOLD,
        decision=decision,
        reason_codes=reason_codes,
        scored_at=datetime.now(UTC),
    )
    _inference_log.record(scoring_decision)

    return ScoreResponse(
        decision_id=scoring_decision.decision_id,
        application_id=scoring_decision.application_id,
        model_version=scoring_decision.model_version,
        probability_of_default=round(scoring_decision.probability_of_default, 4),
        decision=scoring_decision.decision.value,
        reason_codes=[
            ReasonCodeView(feature_name=r.feature_name, contribution=r.contribution, direction=r.direction)
            for r in reason_codes
        ],
    )


@app.post("/override/{decision_id}", response_model=OverrideResponse)
def override(decision_id: str, request: OverrideRequest) -> OverrideResponse:
    """A missing reason, an unknown reviewer, or an unknown decision_id are
    client errors (400), not server failures — the global 500 handler is
    for genuinely unexpected bugs, not for the mandatory-reason validation
    that IS the point of this endpoint (AI Act Art. 14)."""
    try:
        record = _oversight_log.override(
            decision_id=decision_id,
            overridden_decision=Decision(request.overridden_decision),
            reason=request.reason,
            overridden_by=request.overridden_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OverrideResponse(
        decision_id=record.decision_id,
        original_decision=record.original_decision.value,
        overridden_decision=record.overridden_decision.value,
        reason=record.reason,
        overridden_by=record.overridden_by,
    )


@app.get("/fairness")
def fairness() -> dict:
    return {
        "model_version": _fairness_report.model_version,
        "demographic_parity_difference": _fairness_report.demographic_parity_difference,
        "equal_opportunity_difference": _fairness_report.equal_opportunity_difference,
        "groups": [
            {
                "group": g.group,
                "n": g.n,
                "approval_rate": round(g.approval_rate, 4),
                "true_positive_rate": round(g.true_positive_rate, 4),
            }
            for g in _fairness_report.groups
        ],
    }


@app.get("/dossier", response_class=PlainTextResponse)
def dossier() -> str:
    doc = generate_dossier("scorecard-logreg-v1", _training_result, _fairness_report)
    return doc.to_markdown()
