import pytest

from creditrisk.registry.model_registry import ModelRegistry, ModelStage


class TestModelRegistry:
    def test_new_model_registers_in_dev_stage(self):
        registry = ModelRegistry()
        model = registry.register("v1", metrics={"auc": 0.9})
        assert model.stage == ModelStage.DEV

    def test_registering_the_same_version_twice_raises(self):
        registry = ModelRegistry()
        registry.register("v1", metrics={})
        with pytest.raises(ValueError):
            registry.register("v1", metrics={})

    def test_valid_promotion_path_dev_to_shadow_to_prod(self):
        registry = ModelRegistry()
        registry.register("v1", metrics={})
        registry.promote("v1", ModelStage.SHADOW)
        registry.promote("v1", ModelStage.PROD)
        assert registry.get("v1").stage == ModelStage.PROD

    def test_cannot_skip_shadow_and_go_straight_to_prod(self):
        registry = ModelRegistry()
        registry.register("v1", metrics={})
        with pytest.raises(ValueError):
            registry.promote("v1", ModelStage.PROD)

    def test_prod_is_a_terminal_stage_except_for_archiving(self):
        registry = ModelRegistry()
        registry.register("v1", metrics={})
        registry.promote("v1", ModelStage.SHADOW)
        registry.promote("v1", ModelStage.PROD)
        with pytest.raises(ValueError):
            registry.promote("v1", ModelStage.DEV)
        registry.promote("v1", ModelStage.ARCHIVED)  # this one is allowed
        assert registry.get("v1").stage == ModelStage.ARCHIVED

    def test_archived_stage_accepts_no_further_transitions(self):
        registry = ModelRegistry()
        registry.register("v1", metrics={})
        registry.promote("v1", ModelStage.ARCHIVED)
        with pytest.raises(ValueError):
            registry.promote("v1", ModelStage.DEV)

    def test_promoting_unknown_version_raises(self):
        registry = ModelRegistry()
        with pytest.raises(ValueError, match="unknown model version"):
            registry.promote("nonexistent", ModelStage.SHADOW)

    def test_only_one_model_can_be_in_production_at_a_time(self):
        registry = ModelRegistry()
        registry.register("v1", metrics={})
        registry.promote("v1", ModelStage.SHADOW)
        registry.promote("v1", ModelStage.PROD)

        registry.register("v2", metrics={})
        registry.promote("v2", ModelStage.SHADOW)
        registry.promote("v2", ModelStage.PROD)

        assert registry.get("v1").stage == ModelStage.ARCHIVED
        assert registry.get("v2").stage == ModelStage.PROD
        assert registry.current_production().version == "v2"

    def test_current_production_is_none_when_nothing_is_deployed(self):
        registry = ModelRegistry()
        registry.register("v1", metrics={})
        assert registry.current_production() is None

    def test_stage_history_records_every_transition(self):
        registry = ModelRegistry()
        registry.register("v1", metrics={})
        registry.promote("v1", ModelStage.SHADOW)
        registry.promote("v1", ModelStage.PROD)
        stages = [s for s, _ in registry.get("v1").stage_history]
        assert stages == [ModelStage.DEV, ModelStage.SHADOW, ModelStage.PROD]
