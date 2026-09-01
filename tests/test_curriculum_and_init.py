"""Curriculum coverage, staged unfreezing, and deliberate-init survival.

Three failures motivate this file, all of the same shape: something that looks
trained is not.

1. A component that no stage lists is frozen at *every* stage, silently. That
   had swallowed ~29k parameters, the whole MS-detectability path among them.
2. Stages freeze parameters but never unfroze them, so a model stepped through
   the curriculum stayed pinned at stage 1.
3. `_init_weights` re-initializes every `nn.Embedding` it can reach, so a
   zero-initialized embedding is silently randomized unless it opts out.
"""

import pytest

torch = pytest.importorskip("torch")

from presto.models.presto import Presto  # noqa: E402


@pytest.fixture
def model():
    return Presto(d_model=32, n_layers=2, n_heads=4, latent_topology="expanded")


ALL_STAGES = (
    "STAGE_BINDING_CLASS1",
    "STAGE_BINDING_CLASS2",
    "STAGE_PROCESSING_CLASS1",
    "STAGE_PROCESSING_CLASS2",
    "STAGE_PRESENTATION_MIL",
    "STAGE_IMMUNOGENICITY",
)


class TestEveryParameterIsReachable:
    def test_no_component_is_trained_by_zero_stages(self, model):
        """The invariant, not a list of names: nothing may be unreachable.

        Asserted as a property so a newly added component cannot pass by
        being absent from a hardcoded expectation.
        """
        # Raises if any classified component is missing from every stage.
        model.curriculum_param_groups(model.STAGE_BINDING_CLASS1)

    def test_final_stage_trains_every_parameter(self, model):
        model.curriculum_param_groups(model.STAGE_IMMUNOGENICITY)
        frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
        assert frozen == []

    def test_ms_detectability_is_trained_somewhere(self, model):
        """It was classified into a component no stage listed."""
        model.curriculum_param_groups(model.STAGE_PRESENTATION_MIL)
        component_map = model._parameter_component_map()
        trainable = [
            name
            for name, p in model.named_parameters()
            if component_map.get(name) == "ms_detectability" and p.requires_grad
        ]
        assert trainable, "ms_detectability is frozen at every curriculum stage"

    def test_unreachable_component_is_rejected_loudly(self, model, monkeypatch):
        """A misclassified parameter must fail, not silently freeze."""
        real_map = model._parameter_component_map()
        poisoned = dict(real_map)
        poisoned[next(iter(poisoned))] = "component_that_no_stage_trains"
        monkeypatch.setattr(model, "_parameter_component_map", lambda: poisoned)
        with pytest.raises(ValueError, match="trained by no stage"):
            model.curriculum_param_groups(model.STAGE_IMMUNOGENICITY)


class TestStagesAreSequential:
    def test_advancing_a_stage_unfreezes_what_it_adds(self, model):
        model.curriculum_param_groups(model.STAGE_BINDING_CLASS1)
        frozen_early = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
        assert frozen_early > 0, "stage 1 should freeze something"
        model.curriculum_param_groups(model.STAGE_IMMUNOGENICITY)
        frozen_late = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
        assert frozen_late == 0, (
            "advancing the curriculum left parameters frozen; stages are "
            "called in sequence on one model, so active groups must be "
            "explicitly re-enabled"
        )

    @pytest.mark.parametrize("stage_attr", ALL_STAGES)
    def test_every_stage_trains_something(self, model, stage_attr):
        groups = model.curriculum_param_groups(getattr(model, stage_attr))
        assert any(g["lr"] > 0 and g["params"] for g in groups)


class TestDeliberateInitSurvives:
    def test_zero_init_embeddings_are_not_clobbered(self, model):
        """The submodule opt-out still holds.

        This also checked `processing_condition_embed`, which no longer
        exists: it fed cellular state into the trunk, and that input path was
        removed when the condition axes became output tracks.
        """
        assert not hasattr(model, "processing_condition_embed"), (
            "the cellular-condition input embedding is back"
        )
        assert float(model.excision_head.length_preference.weight.abs().sum()) == 0.0

    def test_helper_registers_what_it_creates(self, model):
        """Registration is inseparable from construction, by construction.

        The previous mechanism required remembering to declare each new
        embedding, and the very next zero-init embedding added forgot.
        """
        created = model._zero_init_embedding(4, 8)
        assert id(created.weight) in model._preserved_init_params
        assert float(created.weight.abs().sum()) == 0.0

    def test_ordinary_embeddings_are_still_initialized(self, model):
        """The blanket init must still do its job for everything else."""
        import torch.nn as nn

        preserved = set(model._preserved_init_params)
        preserved.update(
            id(parameter)
            for module in model.modules()
            for parameter in getattr(module, "preserve_init_parameters", lambda: [])()
        )
        ordinary = [
            module
            for module in model.modules()
            if isinstance(module, nn.Embedding) and id(module.weight) not in preserved
        ]
        assert ordinary, "expected at least one blanket-initialized embedding"
        assert all(float(m.weight.abs().sum()) > 0.0 for m in ordinary)
