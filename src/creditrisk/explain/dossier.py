"""AI Act Art. 11 — technical documentation generated from code, not written
by hand after the fact.

`generate_dossier` refuses to produce a dossier from incomplete artifacts —
a missing fairness report, a NaN AUC, an empty training split — rather than
silently emitting a document with blank sections. In CI (`scripts/demo.py`,
`.github/workflows/ci.yml`) this means: if a future change breaks fairness
computation or training, dossier generation fails loudly instead of
producing a technically-present-but-hollow compliance document. That
failure mode is the entire point — a documentation pipeline that always
"succeeds" regardless of what it was fed is not actually compliance
tooling, it's a template.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from creditrisk.model.train import TrainingResult
from creditrisk.models import FairnessReport


class DossierGenerationError(ValueError):
    """Raised when a required artifact for the technical dossier is missing
    or invalid — never caught silently, meant to fail a CI step."""


@dataclass(frozen=True, slots=True)
class TechnicalDossier:
    generated_at: datetime
    model_version: str
    dataset_summary: dict
    performance: dict
    fairness: FairnessReport
    known_limitations: list[str]

    def to_markdown(self) -> str:
        lines = [
            f"# Dossier tecnico — {self.model_version}",
            f"\nGenerato automaticamente il {self.generated_at.isoformat()}.",
            "\n## Dataset",
        ]
        for key, value in self.dataset_summary.items():
            lines.append(f"- **{key}**: {value}")

        lines.append("\n## Performance")
        for key, value in self.performance.items():
            lines.append(f"- **{key}**: {value}")

        lines.append("\n## Fairness")
        lines.append(f"- **demographic_parity_difference**: {self.fairness.demographic_parity_difference}")
        lines.append(f"- **equal_opportunity_difference**: {self.fairness.equal_opportunity_difference}")
        for group in self.fairness.groups:
            lines.append(
                f"  - gruppo {group.group}: n={group.n}, approval_rate={group.approval_rate:.3f}, "
                f"TPR={group.true_positive_rate:.3f}"
            )

        lines.append("\n## Limiti noti")
        for limitation in self.known_limitations:
            lines.append(f"- {limitation}")

        return "\n".join(lines)


def generate_dossier(
    model_version: str,
    training_result: TrainingResult,
    fairness_report: FairnessReport,
) -> TechnicalDossier:
    _require_artifacts(training_result, fairness_report)

    dataset_summary = {
        "n_train": len(training_result.split.train_df),
        "n_test": len(training_result.split.test_df),
        "cutoff_date": str(training_result.split.cutoff_date),
        "feature_columns": ", ".join(training_result.preprocessor.feature_columns),
    }
    performance = {
        "scorecard_auc": round(training_result.scorecard_auc, 4),
        "blackbox_auc": round(training_result.blackbox_auc, 4),
        "auc_gap_blackbox_minus_scorecard": round(training_result.auc_gap, 4),
    }
    known_limitations = [
        "Dataset interamente sintetico: nessun dato creditizio reale è stato usato per addestrare "
        "o valutare questo modello.",
        f"Gap di parità demografica rilevato fra i gruppi protetti: "
        f"{fairness_report.demographic_parity_difference:.1%} — vedi sezione Fairness.",
        "Soglia di decisione fissa a 0.5, non ottimizzata per un costo asimmetrico di falsi "
        "positivi/negativi specifico del prodotto.",
        "I reason code sono nativi del modello scorecard (coefficienti standardizzati); il modello "
        "black-box di confronto non produce reason code e non è mai usato per decisioni reali.",
    ]

    return TechnicalDossier(
        generated_at=datetime.now(UTC),
        model_version=model_version,
        dataset_summary=dataset_summary,
        performance=performance,
        fairness=fairness_report,
        known_limitations=known_limitations,
    )


def _require_artifacts(training_result: TrainingResult, fairness_report: FairnessReport) -> None:
    if len(training_result.split.train_df) == 0 or len(training_result.split.test_df) == 0:
        raise DossierGenerationError("training/test split is empty — cannot document an untrained model")
    if math.isnan(training_result.scorecard_auc) or math.isnan(training_result.blackbox_auc):
        raise DossierGenerationError("AUC is NaN — training/evaluation did not complete successfully")
    if not fairness_report.groups:
        raise DossierGenerationError("fairness report has no groups — fairness was not actually computed")
    if math.isnan(fairness_report.demographic_parity_difference):
        raise DossierGenerationError("demographic_parity_difference is NaN — fairness computation failed")
