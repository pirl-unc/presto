"""Subset transfer and frozen-function guarantees, using artificial observations."""

from dataclasses import asdict, replace
import json

import pytest
import torch

from presto.data.modular import AA, Context, MHCInput, Observation, TargetScale
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    file_sha256,
    write_observations,
)
from presto.models.modular import ModularConfig, ModularPresto
from presto.models.modular_dependencies import freeze_dependencies, parameter_dependencies
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from presto.training.modular_run import fit_index, preflight


PAIR = MHCInput(AA, AA[::-1], "I")
SCALE = TargetScale(2, 1, 2)


def row(endpoint="ic50", **kwargs):
    values = dict(
        observation_id=endpoint,
        source="software",
        endpoint=endpoint,
        peptide=AA,
        mhc=(PAIR,),
        value=100,
        unit="nM",
        group_id="g",
        provenance={"original_measurement_id": endpoint},
    )
    if endpoint in ("bulk_observation", "ms_response", "elution_observation", "tcr_binding"):
        values.update(value=1, unit="binary")
    if endpoint in ("bulk_observation", "ms_response"):
        values["mhc"] = ()
    if endpoint == "tcr_binding":
        values.update(tcr_a=AA, tcr_b=AA[::-1])
    values.update(kwargs)
    return Observation(**values)


def model(endpoints=("ic50",), **kwargs):
    return ModularPresto(
        ModularConfig(endpoints=endpoints, mhc_classes=("I",), dropout=0.3, **kwargs)
    )


def trained_affinity(tmp_path):
    previous = model()
    optimizer = torch.optim.AdamW(previous.parameters(), lr=0.001)
    train_step(previous, optimizer, [row(value=10), row(value=1000)], {"ic50": SCALE})
    path = tmp_path / "prior.pt"
    save_checkpoint(path, previous, {"ic50": SCALE})
    return previous, path


def test_freezing_preserves_function_and_dropout_while_new_head_trains(tmp_path):
    _, path = trained_affinity(tmp_path)
    extended = model(("ic50", "kd"))
    initialize_from_checkpoint(extended, path)
    frozen = freeze_dependencies(extended, ["ic50"])
    assert "encoder.embedding.weight" in frozen["parameters"]
    assert "pmhc_fusion.0.weight" in frozen["parameters"]
    assert "assay_layer.factors.endpoint.kd" not in frozen["parameters"]
    assert extended.support["ic50"]["status"] == "pretrained_frozen"
    extended.train()
    assert not extended.encoder.training
    original = extended([row()])[0].value.detach().clone()
    assert torch.equal(original, extended([row()])[0].value)
    before = extended.assay_layer.factors["endpoint"]["kd"].detach().clone()
    optimizer = torch.optim.AdamW([p for p in extended.parameters() if p.requires_grad], lr=0.01)
    train_step(extended, optimizer, [row("kd", value=1000)], {"ic50": SCALE, "kd": SCALE})
    assert not torch.equal(before, extended.assay_layer.factors["endpoint"]["kd"])
    assert torch.equal(original, extended([row()])[0].value)
    with pytest.raises(ValueError, match="no trainable loss"):
        train_step(extended, optimizer, [row()], {"ic50": SCALE})
    save_checkpoint(
        tmp_path / "frozen.pt", extended, {"ic50": SCALE, "kd": SCALE}, optimizer=optimizer
    )
    restored, _, payload = load_checkpoint(tmp_path / "frozen.pt")
    assert restored.frozen_parameters == extended.frozen_parameters
    assert payload["frozen_targets"] == ["ic50"]
    restored.train()
    assert not restored.encoder.training
    assert torch.equal(original, restored([row()])[0].value)
    restored.encoder.embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="freeze flags"):
        save_checkpoint(tmp_path / "bad.pt", restored, {"ic50": SCALE, "kd": SCALE})


def test_random_freezes_and_unsupported_frozen_features_fail(tmp_path):
    random = model()
    with pytest.raises(ValueError, match="without trained support"):
        freeze_dependencies(random, ["ic50"])
    assert all(parameter.requires_grad for parameter in random.parameters())
    previous = model(("ms_response",))
    train_step(
        previous,
        torch.optim.AdamW(previous.parameters(), lr=0.001),
        [row("ms_response", peptide="AAAAAAAAA")],
        {},
    )
    freeze_dependencies(previous, ["ms_response"])
    # Trained encoder tensors do not authorize unseen residue rows.
    from presto.models.modular_support import feature_inputs

    with pytest.raises(ValueError, match="untrained input features"):
        with feature_inputs(previous):
            previous([row("ms_response", peptide="CCCCCCCCC")])
    path = tmp_path / "tampered.pt"
    save_checkpoint(path, previous, {})
    payload = torch.load(path, weights_only=True)
    payload["frozen_parameters"] = []
    torch.save(payload, path)
    with pytest.raises(ValueError, match="frozen dependencies"):
        load_checkpoint(path)


def test_ms_component_freeze_preserves_shared_score_while_digest_updates():
    network = model(("bulk_observation",))
    bulk = row("bulk_observation")
    train_step(network, torch.optim.AdamW(network.parameters(), lr=0.001), [bulk], {})
    freeze_dependencies(network, ["component:ms_response"])
    baseline = network([bulk])[0]
    score = baseline.components["ms_response"].detach().clone()
    digest = baseline.components["digest"].detach().clone()
    train_step(
        network,
        torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=0.01),
        [replace(bulk, value=0)],
        {},
    )
    after = network([bulk])[0]
    assert torch.equal(after.components["ms_response"], score)
    assert not torch.equal(after.components["digest"], digest)
    assert network.capabilities()["components"]["ms_response"]["frozen"]
    assert network.capabilities()["components"]["ms_response"]["status"] == "indirectly_trained"


def test_explicit_subset_removes_tasks_scales_and_masked_receptor_columns(tmp_path):
    previous = model(("ic50", "elution_observation", "tcr_binding"))
    rows = [row(), row("elution_observation"), row("tcr_binding")]
    train_step(previous, torch.optim.AdamW(previous.parameters(), lr=0.001), rows, {"ic50": SCALE})
    previous.eval()
    expected = previous([rows[1]])[0].value
    path = tmp_path / "mixed.pt"
    save_checkpoint(path, previous, {"ic50": SCALE})
    subset = model(("elution_observation",))
    with pytest.raises(ValueError, match="explicit migration"):
        initialize_from_checkpoint(subset, path)
    report = initialize_from_checkpoint(subset, path, allow_task_removal=True)
    subset.eval()
    torch.testing.assert_close(subset([rows[1]])[0].value, expected)
    assert report["removed_tasks"] == ["ic50", "tcr_binding"]
    assert report["scales"] == {}
    assert set(subset.support) == {"elution_observation"}
    assert len(report["feature_projection"]["context_fusion.0.weight"]["removed"]) == 384
    assert not subset.has_tcr
    # The reverse extension copies retained columns but new TCR columns stay unsupported.
    small = tmp_path / "small.pt"
    save_checkpoint(small, subset, {})
    extended = model(("elution_observation", "tcr_binding"))
    report = initialize_from_checkpoint(extended, small)
    assert len(report["feature_projection"]["context_fusion.0.weight"]["new"]) == 384
    extended.eval()
    torch.testing.assert_close(extended([rows[1]])[0].value, expected)
    assert extended.support["tcr_binding"]["status"] == "random/untrained"


def test_transfer_rejects_semantic_changes_without_mutating_model(tmp_path):
    previous = model(("bulk_observation",), enzymes=("a", "b"))
    train_step(
        previous,
        torch.optim.AdamW(previous.parameters(), lr=0.001),
        [row("bulk_observation", context=Context(enzyme="a"))],
        {},
    )
    path = tmp_path / "prior.pt"
    save_checkpoint(path, previous, {})
    changed = model(("bulk_observation",), enzymes=("b", "a"))
    before = {name: value.clone() for name, value in changed.state_dict().items()}
    with pytest.raises(ValueError, match="feature contract"):
        initialize_from_checkpoint(changed, path, allow_task_removal=True)
    assert all(torch.equal(before[name], value) for name, value in changed.state_dict().items())


def test_streamed_new_task_retains_frozen_output_and_original_scaler(tmp_path):
    previous, path = trained_affinity(tmp_path)
    peptides = [AA[i:] + AA[:i] for i in range(6)]
    rows = [
        row(
            "kd",
            observation_id=f"new-{i}",
            peptide=peptide,
            value=10 if i % 2 else 1000,
            split=("train", "validation", "test")[i // 2],
            group_id=f"g{i}",
            provenance={"original_measurement_id": f"k{i}"},
        )
        for i, peptide in enumerate(peptides)
    ]
    # A retained frozen label is evaluable but must not create a zero-gradient batch.
    rows.append(row(observation_id="retained", value=1e6, group_id="g0"))
    rows.append(
        row(
            observation_id="retained-val",
            value=1e6,
            peptide=peptides[2],
            group_id="g2",
            split="validation",
            provenance={"original_measurement_id": "retained-val"},
        )
    )
    written = write_observations(tmp_path / "data.jsonl", rows)
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": asdict(model(("ic50", "kd")).config),
        "pretrained": {"path": str(path), "sha256": file_sha256(path), "freeze": ["ic50"]},
        "training": {"max_epochs": 2, "max_steps": 2, "batch_size": 1},
        "sources": [
            {
                "path": written["path"],
                "sha256": written["sha256"],
                "source": "software",
                "population": "artificial",
                "label_policy": "software-only",
                "split_policy": "disjoint artificial inputs",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    index = ObservationIndex(manifest_path)
    _, _, scales, admission = preflight(index)
    assert scales["ic50"] == SCALE  # one radically different current value cannot refit it
    assert admission["inherited_scalers"] == ["ic50"]
    summary = fit_index(index, tmp_path / "fit")
    assert summary["coverage_gate"]["status"] == "passed"
    assert summary["training_exposures"] == {"kd": 2}
    final, _, selected = load_checkpoint(tmp_path / "fit/selected.pt")
    assert selected["metadata"]["selection_metric"] == pytest.approx(
        summary["metrics"]["validation"]["kd"]["loss"]
    )
    previous.eval()
    final.eval()
    for name in parameter_dependencies(final, "ic50"):
        assert torch.equal(previous.state_dict()[name], final.state_dict()[name])
    # Match inference modes: trainable and frozen attention otherwise select
    # different float32 kernels even when the retained tensors are identical.
    with torch.no_grad():
        assert torch.equal(previous([row()])[0].value, final([row()])[0].value)
    assert final.support["ic50"]["status"] == "pretrained_frozen"
    assert parameter_dependencies(final, "ic50") <= final.frozen_parameters


def test_frozen_annotation_encoder_needs_no_replay_for_new_assay(tmp_path):
    previous = model((), identity_tasks=("mhc_class",))
    labels = [
        row("identity", value=None, unit=None, identity_labels={"mhc_class": cls})
        for cls in ("I", "II")
    ]
    train_step(previous, torch.optim.AdamW(previous.parameters(), lr=0.001), labels, {})
    prior = tmp_path / "annotation.pt"
    save_checkpoint(prior, previous, {})
    rows = [
        row(
            observation_id=f"r{i}",
            peptide=AA[i:] + AA[:i],
            value=10 if i % 2 else 1000,
            split=("train", "validation", "test")[i // 2],
            group_id=f"g{i}",
            provenance={"original_measurement_id": f"r{i}"},
        )
        for i in range(6)
    ]
    written = write_observations(tmp_path / "rows.jsonl", rows)
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": asdict(model(identity_tasks=("mhc_class",)).config),
        "pretrained": {"path": str(prior), "sha256": file_sha256(prior), "freeze": ["mhc_class"]},
        "training": {"max_epochs": 2, "max_steps": 2, "batch_size": 2, "identity_weight": 0},
        "sources": [
            {
                "path": written["path"],
                "sha256": written["sha256"],
                "source": "software",
                "population": "artificial",
                "label_policy": "software-only",
                "split_policy": "disjoint artificial inputs",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    summary = fit_index(ObservationIndex(path), tmp_path / "fit")
    assert summary["coverage_gate"]["status"] == "passed"
    assert summary["support"]["mhc_class"]["status"] == "pretrained_frozen"
    restored, _, _ = load_checkpoint(tmp_path / "fit/selected.pt")
    previous.eval()
    restored.eval()
    # Compare the same inference mode: autograd-enabled and inference attention
    # kernels can differ by one float32 ULP even with identical frozen weights.
    with torch.no_grad():
        torch.testing.assert_close(
            previous(labels)[0].identity["mhc_class"],
            restored(labels)[0].identity["mhc_class"],
            rtol=0,
            atol=0,
        )
