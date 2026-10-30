"""Artificial software checks; these do not measure biological accuracy."""

from dataclasses import asdict, replace

import pytest
import torch
import json

from presto.data.assay_measurement import (
    AssayMethod,
    AssayTrack,
    normalize_assay_method,
    observation_track,
)
from presto.data.modular import MHCInput, Observation, TargetScale
from presto.models.presto import Presto, PrestoConfig
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    observation_loss,
    save_checkpoint,
    train_step,
)
from presto.training.modular_evaluation import evaluate


RADIO = AssayTrack("ic50", "purified", "competitive", "radioactivity")
FLUOR = replace(RADIO, detector="fluorescence")
SCALES = {"ic50": TargetScale(2.0, 1.0, 2)}


def config(mode="factorized", **kwargs):
    return PrestoConfig(
        assay_tracks=(RADIO, FLUOR),
        assay_mode=mode,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0,
        **kwargs,
    )


def row(track=RADIO, mhc_class="I", **kwargs):
    method = AssayMethod(
        track.preparation, track.mechanism, track.detector, "source_tag_unreviewed"
    )
    return Observation(
        "artificial",
        "software-test",
        "ic50",
        "ACDEFGHIKLMNP",
        (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", mhc_class),),
        value=1000.0,
        unit="nM",
        assay_method=method,
        **kwargs,
    )


@pytest.mark.parametrize("mhc_class", ["I", "II"])
def test_fixed_panel_ignores_observed_selector_and_shares_candidates(mhc_class):
    model = Presto(config()).eval()
    with torch.no_grad():
        model.assay_layer.projection.weight.normal_(0, 0.1)
    a, b = model([row(RADIO, mhc_class), row(FLUOR, mhc_class)])
    assert set(a.assay_values) == {RADIO.key, FLUOR.key}
    for key in a.assay_values:
        torch.testing.assert_close(a.assay_values[key], b.assay_values[key], rtol=0, atol=0)
    torch.testing.assert_close(a.reference_value, b.reference_value, rtol=0, atol=0)
    torch.testing.assert_close(a.value, a.assay_values[RADIO.key])
    torch.testing.assert_close(b.value, b.assay_values[FLUOR.key])
    starts = a.diagnostics["candidates"][0]["registers"][0]["starts"]
    assert len(starts) == (1 if mhc_class == "I" else 5)


def test_all_conditions_start_identically_and_preserve_rng():
    predictions, rngs = [], []
    for mode in ("pooled", "calibration", "factorized"):
        torch.manual_seed(123)
        model = Presto(config(mode)).eval()
        rngs.append(torch.get_rng_state())
        predictions.append(model([row()])[0].value)
    for prediction in predictions[1:]:
        torch.testing.assert_close(prediction, predictions[0], rtol=0, atol=0)
    assert all(torch.equal(rngs[0], state) for state in rngs)


def test_only_observed_track_has_direct_calibration_loss():
    model = Presto(config("calibration"))
    observed = row()
    output = model([observed])[0]
    output.assay_values[FLUOR.key].retain_grad()
    observation_loss(output, observed, SCALES).backward()
    assert model.assay_layer.offset.grad[0] != 0
    assert model.assay_layer.offset.grad[1] == 0
    assert output.assay_values[FLUOR.key].grad is None
    assert model.encoder.embedding.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "qualifier,delta,expected", [("<", -1, 0), ("<=", 1, 1), (">", 1, 0), (">=", -1, 1)]
)
def test_selected_track_keeps_both_inequality_directions(qualifier, delta, expected):
    model = Presto(config("calibration"))
    observed = replace(row(), qualifier=qualifier)
    output = model([observed])[0]
    output.value = torch.tensor(1.0 + delta, requires_grad=True)
    assert float(observation_loss(output, observed, SCALES)) == expected


def test_source_normalization_conflict_and_roundtrip():
    raw = {
        "assay_method": "purified MHC/competitive/fluorescence",
        "assay_comments": "PMID 12135423",
    }
    conflict = normalize_assay_method(raw)
    assert conflict.review_status == "protocol_citation_conflict" and not conflict.usable
    observed = replace(row(), assay_method=None, provenance={"raw_record": raw})
    model = Presto(config())
    output = model([observed])[0]
    assert observation_track(observed) is None
    torch.testing.assert_close(output.value, output.reference_value)
    observation_loss(output, observed, SCALES).backward()
    assert model.assay_layer.projection.weight.grad is None
    assert Observation.from_dict(row().to_dict()) == row()
    raw.pop("assay_comments")
    assert observation_track(observed) == FLUOR


def test_route_validation_and_unknown_registry():
    with pytest.raises(ValueError, match="biological endpoint route"):
        PrestoConfig(assay_tracks=(replace(RADIO, preparation="cellular"),))
    cellular = AssayTrack("tcell_response", "cellular", "functional", "fluorescence")
    PrestoConfig(endpoints=("tcell_response",), assay_tracks=(cellular,))
    model = Presto(replace(config(), assay_tracks=(RADIO,)))
    with pytest.raises(ValueError, match="outside checkpoint registry"):
        model([row(FLUOR)])


def test_factor_support_checkpoint_and_unseen_track_guard(tmp_path):
    model = Presto(config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(4):
        train_step(model, optimizer, [row()], SCALES)
    support = model.assay_capabilities()
    assert support[RADIO.key]["status"] == "directly_supervised"
    assert support[FLUOR.key]["status"] == "random/untrained"
    with pytest.raises(ValueError, match="no trained support"):
        model.predict([row(FLUOR)])
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, SCALES)
    loaded, scales, _ = load_checkpoint(path)
    model.eval()
    loaded.eval()
    assert loaded.assay_support == model.assay_support
    torch.testing.assert_close(model([row()])[0].value, loaded([row()])[0].value, rtol=0, atol=0)
    transferred = Presto(config())
    initialize_from_checkpoint(transferred, path)
    assert transferred.assay_support == model.assay_support
    with pytest.raises(ValueError, match="assay_mode"):
        initialize_from_checkpoint(Presto(config("calibration")), path)
    assert scales == SCALES


def test_v5_checkpoint_keeps_exact_inference(tmp_path):
    model = Presto(replace(config(), assay_tracks=()))
    train_step(model, torch.optim.AdamW(model.parameters(), lr=0.01), [row()], SCALES)
    path = tmp_path / "v5.pt"
    save_checkpoint(path, model, SCALES)
    payload = torch.load(path, weights_only=True)
    payload["format"] = "presto.shared.v5"
    for key in ("assay_tracks", "assay_mode"):
        payload["config"].pop(key)
    payload.pop("assay_support")
    torch.save(payload, path)
    historical, _, _ = load_checkpoint(path)
    model.eval()
    historical.eval()
    torch.testing.assert_close(
        model([row()])[0].value, historical([row()])[0].value, rtol=0, atol=0
    )
    assert historical.element_history_complete == model.element_history_complete
    for name, mask in model.trained_elements.items():
        assert torch.equal(mask, historical.trained_elements[name])
    payload.pop("element_coverage")
    torch.save(payload, path)
    older, _, _ = load_checkpoint(path)
    assert not older.element_history_complete
    with pytest.raises(ValueError, match="historical"):
        initialize_from_checkpoint(Presto(config()), path)


def test_assay_metrics_use_selected_output_in_physical_target_space():
    model = Presto(config("calibration"))
    with torch.no_grad():
        model.assay_layer.offset[:] = torch.tensor([0.0, 2.0])
    rows = [row(), row(FLUOR)]
    metrics, records = evaluate(model, rows, [0, 1], SCALES)
    assert records[1]["prediction"] - records[0]["prediction"] == pytest.approx(2.0)
    assert len(metrics["ic50"]["by_assay_method"]) == 2
    assert records[0]["assay_panel"][RADIO.key] == records[0]["prediction"]
    assert records[0]["assay_method"] == asdict(rows[0].assay_method)


def test_common_manifest_trainer_fits_assay_panel(tmp_path):
    from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, write_observations
    from presto.training.modular_run import fit_index

    peptides = ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH", "IKLMNPQRS", "TVWYACDEF", "GHIKLMNPQ"]
    rows = [
        replace(
            row(RADIO if i % 2 else FLUOR),
            observation_id=str(i),
            peptide=peptide,
            value=10.0 if i % 2 else 1000.0,
            group_id=str(i),
            split=("train", "validation", "test")[i // 2],
            provenance={"original_measurement_id": str(i)},
        )
        for i, peptide in enumerate(peptides)
    ]
    source = write_observations(tmp_path / "rows.jsonl", rows)
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": asdict(config(mhc_classes=("I",))),
        "training": {"max_epochs": 4, "max_steps": 4, "batch_size": 2, "validation_every_steps": 4},
        "sources": [
            {
                "path": str(tmp_path / "rows.jsonl"),
                "sha256": source["sha256"],
                "source": "software-test",
                "population": "artificial software fixture",
                "label_policy": "artificial outcomes",
                "split_policy": "disjoint artificial groups",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    summary = fit_index(ObservationIndex(path), tmp_path / "fit")
    assert summary["coverage_gate"]["status"] == "passed"
    assert len(summary["metrics"]["test"]["ic50"]["by_assay_method"]) == 2
    assert all(track["observations"] == 4 for track in summary["assay_tracks"].values())
