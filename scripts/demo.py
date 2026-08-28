#!/usr/bin/env python3
"""End-to-end CLI demo — no server needed.

Generates a synthetic credit application dataset, trains both the
interpretable scorecard and the black-box comparator on an out-of-time
split, prints the AUC comparison and fairness report, scores one example
application with reason codes, exercises a human-oversight override, and
prints the AI Act Art. 11 technical dossier.

Run with: `python scripts/demo.py` or `make demo`.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditrisk.audit.inference_log import InferenceLog
from creditrisk.audit.oversight import HumanOversightLog, new_decision_id
from creditrisk.data.synthetic import generate_synthetic_applications
from creditrisk.explain.dossier import generate_dossier
from creditrisk.fairness.fairness_report import compute_fairness_report
from creditrisk.model.train import train_and_evaluate
from creditrisk.models import CreditApplication, Decision, ScoringDecision
from creditrisk.registry.model_registry import ModelRegistry, ModelStage

DECISION_THRESHOLD = 0.5


def main() -> None:
    print("=" * 72)
    print("Credit Risk MLOps — demo end-to-end")
    print("=" * 72)

    dataset = generate_synthetic_applications(n=3000, months=24, seed=42)
    print(f"\nGenerate {len(dataset.applications)} domande di credito sintetiche su 24 mesi.")
    print(f"Tasso di default reale: {dataset.applications['defaulted'].mean():.1%}")

    t0 = time.perf_counter()
    result = train_and_evaluate(dataset.applications)
    train_ms = (time.perf_counter() - t0) * 1000
    print(f"\nAddestramento (out-of-time split, cutoff {result.split.cutoff_date}): {train_ms:.0f} ms")
    print(f"  train={len(result.split.train_df)}  test={len(result.split.test_df)}")
    print(f"  AUC scorecard (glass-box, interpretabile): {result.scorecard_auc:.4f}")
    print(f"  AUC black-box (random forest, confronto):  {result.blackbox_auc:.4f}")
    print(f"  gap (black-box - scorecard): {result.auc_gap:+.4f}  "
          f"({'trascurabile' if abs(result.auc_gap) < 0.01 else 'non trascurabile'})")

    print("\n" + "-" * 72)
    print("Fairness report (attributo protetto: protected_group)")
    print("-" * 72)
    for name, model in [("scorecard", result.scorecard), ("blackbox", result.blackbox)]:
        probs = model.predict_proba(result.split.test_df)
        report = compute_fairness_report(name, result.split.test_df, probs, threshold=DECISION_THRESHOLD)
        print(f"\n  {name}:")
        for g in report.groups:
            print(
                f"    gruppo {g.group}: n={g.n:4d}  approval_rate={g.approval_rate:.3f}  "
                f"TPR={g.true_positive_rate:.3f}"
            )
        print(f"    demographic_parity_difference = {report.demographic_parity_difference:.4f}")
        print(f"    equal_opportunity_difference  = {report.equal_opportunity_difference:.4f}")

    scorecard_fairness = compute_fairness_report(
        "scorecard-logreg-v1", result.split.test_df,
        result.scorecard.predict_proba(result.split.test_df), threshold=DECISION_THRESHOLD,
    )

    print("\n" + "-" * 72)
    print("Model registry: promozione a stadi")
    print("-" * 72)
    registry = ModelRegistry()
    registry.register("scorecard-logreg-v1", metrics={"auc": result.scorecard_auc})
    registry.promote("scorecard-logreg-v1", ModelStage.SHADOW)
    registry.promote("scorecard-logreg-v1", ModelStage.PROD)
    prod = registry.current_production()
    print(f"  in produzione: {prod.version} (stage={prod.stage.value})")
    print(f"  storia stadi: {[(s.value, t.strftime('%H:%M:%S')) for s, t in prod.stage_history]}")

    print("\n" + "-" * 72)
    print("Scoring di una domanda di esempio (profilo a rischio)")
    print("-" * 72)
    inference_log = InferenceLog()
    oversight_log = HumanOversightLog(inference_log)

    application = CreditApplication(
        application_id="demo-app-1", application_date=date.today(), protected_group="B",
        age=29, income_monthly=1400, employment_years=1.5, existing_debt_ratio=0.55,
        credit_history_score=490, loan_amount_requested=18000,
    )
    probability = result.scorecard.predict_proba_one(application)
    decision = Decision.APPROVE if probability < DECISION_THRESHOLD else Decision.DENY
    reason_codes = result.scorecard.reason_codes(application)

    scoring_decision = ScoringDecision(
        decision_id=new_decision_id(), application_id=application.application_id,
        model_version="scorecard-logreg-v1", probability_of_default=probability,
        decision_threshold=DECISION_THRESHOLD, decision=decision, reason_codes=reason_codes,
        scored_at=datetime.now(UTC),
    )
    inference_log.record(scoring_decision)

    print(f"  probabilità di default: {probability:.3f}  ->  decisione: {decision.value}")
    print("  reason codes:")
    for rc in reason_codes:
        print(f"    {rc.feature_name}: {rc.contribution:+.3f} ({rc.direction})")

    print("\n" + "-" * 72)
    print("Human oversight (Art. 14): override della decisione")
    print("-" * 72)
    override = oversight_log.override(
        decision_id=scoring_decision.decision_id,
        overridden_decision=Decision.APPROVE,
        reason="Cliente con garante verificato, reddito familiare complessivo sopra soglia.",
        overridden_by="analista.credito@banca.example",
    )
    print(f"  decisione originale: {override.original_decision.value}")
    print(f"  decisione dopo override: {override.overridden_decision.value}")
    print(f"  motivazione: {override.reason!r}")
    print(f"  da: {override.overridden_by}")

    print("\n" + "-" * 72)
    print("Dossier tecnico (Art. 11) — generato da codice")
    print("-" * 72)
    dossier = generate_dossier("scorecard-logreg-v1", result, scorecard_fairness)
    print(dossier.to_markdown())

    print("\n" + "=" * 72)
    print("Demo completata. Avvia il servizio con: uvicorn creditrisk.api.main:app --reload")
    print("=" * 72)


if __name__ == "__main__":
    main()
