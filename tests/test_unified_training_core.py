"""Shared execution and full-context software fixtures; no accuracy claims."""

from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path

import pytest
import torch

from presto.cli import train as cli
from presto.data.modular import AA, MHCInput, Observation, TargetScale, fit_target_scales
from presto.models import Presto, PrestoConfig
from presto.models.modular import ModularConfig, ModularPresto
from presto.models.presto import MODEL_VERSION, candidates
from presto.training import Trainer
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def row(endpoint="ic50", mhc_class="II", **changes):
    values = dict(
        observation_id=f"fixture:{endpoint}:{mhc_class}",
        source="software-only",
        endpoint=endpoint,
        peptide="ACDEFGHIKLMNPQR",
        mhc=(MHCInput(AA, AA[::-1], mhc_class),),
        value=100.0 if endpoint == "ic50" else 1.0,
        unit="nM" if endpoint == "ic50" else "binary",
        group_id="fixture",
        tcr_a=AA,
        tcr_b=AA[::-1],
    )
    values.update(changes)
    return Observation(**values)


def test_public_names_are_the_same_model_and_config():
    assert Presto is ModularPresto
    assert PrestoConfig is ModularConfig
    assert Presto.__module__ == "presto.models.presto"
    assert MODEL_VERSION == "presto.shared.v7"
    with pytest.raises(ValueError, match="full ordered"):
        PrestoConfig(peptide_context="three_residue_composition")


@pytest.mark.parametrize("length", [9, 10, 15, 25])
def test_full_context_keeps_variable_register_softmax(length):
    peptide = (AA * 2)[:length]
    model = Presto(PrestoConfig(dropout=0)).eval()
    first, second = model([row(peptide=peptide, mhc_class=kind) for kind in ("I", "II")])
    for output, count in ((first, 1), (second, length - 8)):
        registers = output.diagnostics["candidates"][0]["registers"][0]
        assert len(registers["starts"]) == len(registers["weights"]) == count
        assert float(registers["weights"].sum().detach()) == pytest.approx(1)
        assert registers["flank_representation"] == model.config.peptide_context


def test_full_ordered_context_keeps_distal_residues_without_extra_encoders():
    torch.manual_seed(211)
    model = Presto(PrestoConfig(dropout=0)).eval()
    peptide = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLM"
    original = row(peptide=peptide)
    reordered = replace(original, peptide="CA" + peptide[2:])
    _, states = model._encode([original, reordered])
    start = 8
    core = peptide[start : start + 9]
    registers = [(start, core, [])]
    first = model._interaction_candidates(peptide, original.mhc[0], registers, states)
    second = model._interaction_candidates(reordered.peptide, reordered.mhc[0], registers, states)
    assert not torch.allclose(first, second, rtol=1e-6, atol=1e-6)
    assert candidates(peptide, "II")[start][1:] == candidates(reordered.peptide, "II")[start][1:]
    assert len([name for name, _ in model.named_modules() if name.endswith("encoder")]) == 1


def test_single_nine_mer_uses_identical_interaction_for_both_classes():
    model = Presto(PrestoConfig(dropout=0)).eval()
    peptide = "ACDEFGHIK"
    first = row(peptide=peptide, mhc_class="I")
    second = row(peptide=peptide, mhc_class="II")
    a, b = model([first, second])
    # Same sequence/grooves with one candidate must have one computation;
    # no hidden class-specific scoring path is allowed.
    torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)
    _, states = model._encode([first, row(peptide=AA)])
    assert states[peptide].shape == (9, 128)


@pytest.mark.parametrize("direction", [-1, 1])
def test_register_pool_is_normalized_monotone_and_has_softmax_derivatives(direction):
    scores = torch.tensor([-2.0, 0.7, 3.0], requires_grad=True)
    pooled, weights = Presto._pool_registers(scores, direction)
    gradient = torch.autograd.grad(pooled, scores)[0]
    torch.testing.assert_close(gradient, weights)
    assert torch.all(gradient > 0)
    assert weights.argmax() == (0 if direction == -1 else 2)
    duplicate, _ = Presto._pool_registers(scores.repeat(2), direction)
    torch.testing.assert_close(duplicate, pooled)
    singleton, singleton_weights = Presto._pool_registers(scores[:1], direction)
    assert torch.equal(singleton, scores[0])
    assert torch.equal(singleton_weights, torch.ones(1))


def test_i_only_training_learns_exact_parameters_used_by_ii_registers():
    torch.manual_seed(231)
    source = Presto(PrestoConfig(mhc_classes=("I",), dropout=0))
    rows = [row(mhc_class="I", peptide=AA[:10], value=10), row(mhc_class="I", value=100)]
    result = train_step(
        source, torch.optim.AdamW(source.parameters(), lr=0.001), rows, fit_target_scales(rows)
    )
    assert all(value["gradient_and_update"] for value in result["coverage"].values())
    assert not hasattr(source, "register_score")
    expanded = Presto(replace(source.config, mhc_classes=("I", "II"))).eval()
    expanded.load_state_dict(source.state_dict(), strict=True)
    assert set(dict(expanded.named_parameters())) == set(dict(source.named_parameters()))
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, expanded.state_dict()[name])
    candidate = expanded([row()])[0]
    registers = candidate.diagnostics["candidates"][0]["registers"][0]
    expected, weights = Presto._pool_registers(registers["candidate_scores"], -1)
    torch.testing.assert_close(candidate.value, expected)
    torch.testing.assert_close(registers["weights"], weights)
    # Retain the original model's supported-class boundary.
    with pytest.raises(ValueError, match="supported branches"):
        source([row()])


def test_distal_flanks_receive_gradient_through_contextual_core_attention():
    torch.manual_seed(233)
    model = Presto(PrestoConfig(dropout=0)).eval()
    captured = []

    def capture(_module, _args, result):
        result.retain_grad()
        captured.append(result)

    hook = model.encoder.embedding.register_forward_hook(capture)
    observation = row(peptide="ACDEFGHIKLMNPQRSTVWYACDEFGHIKLM")
    output = model([observation])[0]
    registers = output.diagnostics["candidates"][0]["registers"][0]
    registers["candidate_scores"][8].backward()
    hook.remove()
    # Full peptide is the first sequence inserted into _encode, before core views.
    gradient = captured[0].grad[0, : len(observation.peptide)].abs().sum(-1)
    assert gradient[0] > 0 and gradient[-1] > 0
    assert torch.all(gradient[8:17] > 0)


@pytest.mark.parametrize(
    "endpoints",
    [
        ("ic50",),
        ("bulk_observation",),
        ("elution_observation",),
        ("tcr_binding",),
        ("tcr_response",),
        ("ic50", "bulk_observation", "elution_observation", "tcr_binding", "tcr_response"),
    ],
)
def test_task_subsets_use_one_model_optimizer_and_checkpoint(tmp_path, endpoints):
    torch.manual_seed(223)
    model = Presto(PrestoConfig(endpoints=endpoints, dropout=0))
    observations = [
        row(endpoint, kind, value=value)
        for endpoint in endpoints
        for kind in ("I", "II")
        for value in ((10.0, 100.0) if endpoint == "ic50" else (0.0, 1.0))
    ]
    scales = fit_target_scales(observations)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    result = train_step(model, optimizer, observations, scales)
    assert result["coverage"]["encoder.embedding.weight"]["gradient_and_update"]
    assert set(result["counts"]) == set(endpoints)
    assert not hasattr(model, "alpha_encoder") and not hasattr(model, "beta_encoder")
    checkpoint = tmp_path / "shared.pt"
    save_checkpoint(checkpoint, model, scales)
    restored, _, payload = load_checkpoint(checkpoint)
    assert type(restored) is Presto
    assert payload["format"] == MODEL_VERSION
    expected, actual = model.eval().predict(observations), restored.predict(observations)
    for before, after in zip(expected, actual, strict=True):
        torch.testing.assert_close(before.value, after.value, rtol=0, atol=0)


def test_identity_pretraining_transfers_into_the_same_binding_model(tmp_path):
    torch.manual_seed(227)
    model = Presto(PrestoConfig(endpoints=(), identity_tasks=("mhc_class",), dropout=0))
    observations = [
        row("identity", kind, value=None, unit=None, identity_labels={"mhc_class": kind})
        for kind in ("I", "II")
    ]
    train_step(model, torch.optim.AdamW(model.parameters(), lr=0.001), observations, {})
    path = tmp_path / "pretraining.pt"
    save_checkpoint(path, model, {})
    target = Presto(PrestoConfig(identity_tasks=("mhc_class",), dropout=0))
    inherited = initialize_from_checkpoint(target, path)
    assert "encoder.embedding.weight" in inherited["inherited_parameters"]
    torch.testing.assert_close(target.encoder.embedding.weight, model.encoder.embedding.weight)
    train_step(
        target,
        torch.optim.AdamW(target.parameters(), lr=0.001),
        [row()],
        {"ic50": TargetScale(2.0, 1.0, 2)},
    )


@pytest.mark.parametrize("version", ["v2", "v3", "v4"])
def test_historical_modular_checkpoint_replays_but_cannot_train(tmp_path, version):
    if version == "v2":
        from presto.legacy.modular_v2 import ModularConfig as OldConfig
        from presto.legacy.modular_v2 import ModularPresto as OldModel
    elif version == "v3":
        from presto.legacy.shared_v3 import PrestoConfig as OldConfig
        from presto.legacy.shared_v3 import Presto as OldModel
    else:
        from presto.legacy.shared_v4 import PrestoConfig as OldConfig
        from presto.legacy.shared_v4 import Presto as OldModel

    torch.manual_seed(229)
    old = OldModel(OldConfig(dropout=0)).eval()
    rows = [row()]
    with torch.no_grad():
        expected = old(rows)[0].value
    payload = {
        "format": "presto.modular.v2" if version == "v2" else f"presto.shared.{version}",
        "config": asdict(old.config),
        "state_dict": old.state_dict(),
        "scales": {"ic50": asdict(TargetScale(2.0, 1.0, 2))},
        "support": old.support,
        "trained_parameters": [],
        "trained_features": {name: [] for name in old.feature_layout},
        "exposed_features": {name: [] for name in old.feature_layout},
        "source_label_support": {},
        "metadata": {"scope": "random software fixture, no claimed training"},
    }
    path = tmp_path / "historical.pt"
    torch.save(payload, path)
    restored, scales, _ = load_checkpoint(path)
    with torch.no_grad():
        actual = restored.eval()(rows)[0].value
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="historical models are inference-only"):
        train_step(restored, torch.optim.AdamW(restored.parameters()), rows, scales)
    with pytest.raises(ValueError, match="historical checkpoint feature contracts"):
        initialize_from_checkpoint(Presto(PrestoConfig()), path)


def test_all_script_launchers_are_aliases_of_one_manifest_entrypoint(tmp_path, capsys):
    from test_modular_training import fixture_manifest

    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "tasks/unified_training_core_20261027_archive.json").read_text())
    names = inventory["active_launchers"] + [
        "distributional_ba.train",
        "distributional_ba.pretrain_mhc",
    ]
    manifest, _ = fixture_manifest(tmp_path)
    for name in names:
        module = importlib.import_module(f"presto.scripts.{name}")
        assert module.main is cli.main
        assert module.run is cli.run
        result = module.main(["--manifest", str(manifest), "--preflight"])
        assert result["training_labels"] == {"ic50": 2}
        capsys.readouterr()


def test_public_trainer_executes_the_canonical_fit(tmp_path):
    from test_modular_training import fixture_manifest

    manifest, _ = fixture_manifest(tmp_path)
    trainer = Trainer(manifest)
    result = trainer.fit(tmp_path / "run")
    model, _, _ = load_checkpoint(tmp_path / "run/selected.pt")
    assert type(model) is Presto
    assert result["steps"] == 2


def test_active_training_has_one_optimizer_step():
    """An added standalone loop must not escape the shared training contract."""
    import ast

    root = Path(__file__).resolve().parents[1]
    backward_files, step_files = set(), set()
    for directory in ("cli", "training", "scripts"):
        for path in (root / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                relative = str(path.relative_to(root))
                if node.func.attr == "backward":
                    backward_files.add(relative)
                if node.func.attr == "step" and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "optimizer":
                        step_files.add(relative)
    assert backward_files == step_files == {"training/modular.py"}


def test_observation_predictor_is_an_alias_and_legacy_wrapper_cannot_be_created():
    from presto.inference import ObservationPredictor
    from presto.inference.modular import ModularPredictor
    from presto.models.tcr_specificity import TCRSpecificityModel

    assert ObservationPredictor is ModularPredictor
    with pytest.raises(ValueError, match="Separate TCR models are retired"):
        TCRSpecificityModel(None)
