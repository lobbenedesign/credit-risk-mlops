"""Lightweight staged model registry.

Mirrors the stage-promotion semantics of a real model registry (MLflow
Model Registry's dev → staging → production, or SageMaker Model Registry's
approval workflow) without depending on running one: this project's point
is the promotion *discipline* — a model does not go straight from freshly
trained to serving decisions, and only one version is ever in `prod` at a
time — not the specific tracking-server infrastructure behind it. Swapping
this for a real MLflow-backed registry is a same-interface change, the same
trade-off already made for the in-memory stores in the other repos of this
portfolio (see docs/adr/0004).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class ModelStage(StrEnum):
    DEV = "dev"
    SHADOW = "shadow"
    PROD = "prod"
    ARCHIVED = "archived"


# Stages a model may be promoted *to* from a given current stage, without
# skipping steps. ARCHIVED is reachable from anywhere — a rollback/retire
# must always be possible regardless of how a model got into trouble.
_ALLOWED_TRANSITIONS: dict[ModelStage, set[ModelStage]] = {
    ModelStage.DEV: {ModelStage.SHADOW, ModelStage.ARCHIVED},
    ModelStage.SHADOW: {ModelStage.PROD, ModelStage.ARCHIVED},
    ModelStage.PROD: {ModelStage.ARCHIVED},
    ModelStage.ARCHIVED: set(),
}


@dataclass(slots=True)
class RegisteredModel:
    version: str
    stage: ModelStage
    metrics: dict
    registered_at: datetime
    stage_history: list[tuple[ModelStage, datetime]] = field(default_factory=list)


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, RegisteredModel] = {}

    def register(self, version: str, metrics: dict) -> RegisteredModel:
        if version in self._models:
            raise ValueError(f"model version already registered: {version}")
        now = datetime.now(UTC)
        entry = RegisteredModel(
            version=version, stage=ModelStage.DEV, metrics=dict(metrics),
            registered_at=now, stage_history=[(ModelStage.DEV, now)],
        )
        self._models[version] = entry
        return entry

    def promote(self, version: str, to_stage: ModelStage) -> RegisteredModel:
        model = self._require(version)
        allowed = _ALLOWED_TRANSITIONS[model.stage]
        if to_stage not in allowed:
            raise ValueError(
                f"cannot promote {version} from {model.stage.value} to {to_stage.value}; "
                f"allowed: {sorted(s.value for s in allowed) or 'none (terminal stage)'}"
            )

        if to_stage == ModelStage.PROD:
            current_prod = self.current_production()
            if current_prod is not None and current_prod.version != version:
                self._set_stage(current_prod, ModelStage.ARCHIVED)

        self._set_stage(model, to_stage)
        return model

    def current_production(self) -> RegisteredModel | None:
        for model in self._models.values():
            if model.stage == ModelStage.PROD:
                return model
        return None

    def get(self, version: str) -> RegisteredModel | None:
        return self._models.get(version)

    def all(self) -> list[RegisteredModel]:
        return list(self._models.values())

    def _require(self, version: str) -> RegisteredModel:
        model = self._models.get(version)
        if model is None:
            raise ValueError(f"unknown model version: {version}")
        return model

    @staticmethod
    def _set_stage(model: RegisteredModel, stage: ModelStage) -> None:
        model.stage = stage
        model.stage_history.append((stage, datetime.now(UTC)))
