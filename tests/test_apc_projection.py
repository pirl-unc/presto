"""Unknown vocabulary projection preserves evidence and strict numeric contracts."""

from dataclasses import asdict, replace

import pytest
import torch

from presto.data.modular import Context, ContextAssertion, Exposure, MHCInput, Observation
from presto.data.modular_context import project_apc_context, resolve_cell_context
from presto.inference.modular import ModularPredictor
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import save_checkpoint, train_step


def test_projection_preserves_claims_absence_and_unsupported_override():
    original = Context(
        cell_type="known",
        cell_type_override="unseen",
        cytokines={"IFNG": Exposure("absent"), "new": Exposure("present", 2, "ng/mL")},
        interventions={"B2M_KO": ContextAssertion("present", {"source": "assay"})},
        processing_pathways={"unknown_path": ContextAssertion()},
        provenance={"source": "original"},
    )
    before = asdict(original)
    projected = project_apc_context(original, cell_types=("known",), cytokines=("IFNG",))
    assert projected.cell_type is None and projected.cell_type_override is None
    assert projected.cytokines == {"IFNG": Exposure("absent")}
    assert projected.interventions == projected.processing_pathways == {}
    omitted = projected.provenance["unrepresented_apc_context"][0]
    assert omitted["cell_type"] == "known" and omitted["cell_type_override"] == "unseen"
    assert omitted["interventions"]["B2M_KO"] == before["interventions"]["B2M_KO"]
    assert omitted["cytokines"]["new"] == before["cytokines"]["new"]
    assert projected.provenance["source"] == "original"
    assert asdict(original) == before
    assert project_apc_context(projected, cell_types=("known",), cytokines=("IFNG",)) is projected
    resolved = resolve_cell_context(projected, ("known",), ("IFNG",), {}, ())
    assert resolved.cell_type.source == "unknown"
    assert resolved.cytokines["IFNG"].state == "absent"


def test_supported_override_and_numeric_contract_remain_strict():
    original = Context(
        cell_type="unseen",
        cell_type_override="known",
        cytokines={"IFNG": Exposure("present", 3, "wrong")},
    )
    projected = project_apc_context(original, cell_types=("known",), cytokines=("IFNG",))
    assert projected is original
    with pytest.raises(ValueError, match="dose unit mismatch"):
        resolve_cell_context(projected, ("known",), ("IFNG",), {"IFNG": "ng/mL"}, ())


@pytest.mark.parametrize("mhc_class", ["I", "II"])
def test_public_projection_is_explicit_and_equals_unknown_input(tmp_path, mhc_class):
    row = Observation(
        observation_id="fixture",
        source="fixture",
        endpoint="elution_observation",
        peptide="ACDEFGHIK",
        mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", mhc_class),),
        value=1,
        unit="binary",
        group_id="fixture",
        context=Context(cell_type="known"),
    )
    model = ModularPresto(
        ModularConfig(
            endpoints=("elution_observation",),
            mhc_classes=(mhc_class,),
            dropout=0,
            cell_types=("known",),
        )
    )
    train_step(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.001),
        [row, replace(row, observation_id="unknown", context=Context())],
        {},
    )
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {})
    predictor = ModularPredictor(path)
    request = replace(row, value=None, split="predict", context=Context(cell_type="unseen"))
    with pytest.raises(ValueError, match="outside checkpoint vocabulary"):
        predictor.predict([request])
    actual = predictor.predict([request], unsupported_apc_policy="unknown")[0]
    expected = predictor.predict([replace(request, context=Context())])[0]
    assert actual["logit"] == expected["logit"]
    assert actual["unrepresented_apc_context"][0]["cell_type"] == "unseen"
    with pytest.raises(ValueError, match="unsupported_apc_policy"):
        predictor.predict([request], unsupported_apc_policy="typo")
