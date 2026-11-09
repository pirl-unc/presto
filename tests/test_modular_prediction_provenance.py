"""Saved predictions must retain the inputs needed for independent auditing."""

from dataclasses import replace
import json

import pytest
import torch

from presto.data.modular import MHCInput, Observation, TargetScale
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import observation_loss
from presto.training.modular_evaluation import evaluate


@pytest.mark.parametrize("training", [False, True])
def test_evaluation_retains_inputs_and_resolved_registers_without_changing_values(training):
    row = Observation(
        "artificial:1",
        "software-fixture",
        "ic50",
        "ACDEFGHIKLMNPQR",
        (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "II", provenance={"allele": "fixture"}),),
        value=100.0,
        unit="nM",
        split="validation",
        group_id="fixture-group",
        provenance={"study_id": "artificial-study", "raw_bounds": [100.0, None]},
    )
    model = ModularPresto(ModularConfig(dropout=0)).eval()
    scales = {"ic50": TargetScale(2.0, 1.5, 2)}
    with torch.no_grad():
        expected = model([row])[0]
        expected_loss = float(observation_loss(expected, row, scales))
    model.train(training)
    _, records = evaluate(model, [row], [0], scales)
    assert model.training is training
    saved = json.loads(json.dumps(records, allow_nan=False))[0]
    assert Observation.from_dict(saved["observation"]) == row
    assert saved["prediction"] == scales["ic50"].decode(float(expected.value))
    assert saved["loss"] == expected_loss
    candidate = saved["diagnostics"]["candidates"][0]
    assert candidate["mhc"]["provenance"] == {"allele": "fixture"}
    registers = candidate["registers"][0]
    assert registers["starts"] == list(range(len(row.peptide) - 8))
    assert sum(registers["weights"]) == pytest.approx(1.0)
    assert len(registers["weights"]) == len(registers["starts"])
    assert saved["component_scores"] == {}
    # Reporting must preserve a bound as reported, rather than rewriting its target.
    bounded = replace(row, qualifier=">")
    _, bounded_records = evaluate(model, [bounded], [0], scales)
    assert bounded_records[0]["observation"]["qualifier"] == ">"
