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
from presto.models.modular_dependencies import freeze_dependencies
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
QUAL_COMP = AssayTrack("qualitative_binding", "purified", "competitive", "fluorescence")
QUAL_DIRECT = replace(QUAL_COMP, mechanism="direct")
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


def qualitative_row(track=QUAL_COMP, value=1.0, **kwargs):
    return Observation(
        "qualitative",
        "software-test",
        "qualitative_binding",
        "ACDEFGHIKLMNP",
        (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I"),),
        value=value,
        unit="binary",
        assay_method=AssayMethod(
            track.preparation,
            track.mechanism,
            track.detector,
            "database_curated",
        ),
        **kwargs,
    )


def td50_row(value=55.0, **kwargs):
    return Observation(
        "td50",
        "software-test",
        "td50",
        "ACDEFGHIKLMNP",
        (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I"),),
        value=value,
        unit="C",
        assay_method=AssayMethod("purified", "direct", "fluorescence", "database_curated"),
        **kwargs,
    )


def expanded_config():
    return PrestoConfig(
        endpoints=("ic50", "qualitative_binding", "td50"),
        assay_tracks=(RADIO, FLUOR, QUAL_COMP, QUAL_DIRECT),
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        dropout=0,
    )


@pytest.mark.parametrize("mhc_class", ["I", "II"])
def test_fixed_panel_ignores_observed_selector_and_shares_candidates(mhc_class):
    model = Presto(config()).eval()
    a, b = model([row(RADIO, mhc_class), row(FLUOR, mhc_class)])
    assert set(a.assay_values) == {RADIO.key, FLUOR.key}
    for key in a.assay_values:
        torch.testing.assert_close(a.assay_values[key], b.assay_values[key], rtol=0, atol=0)
    torch.testing.assert_close(a.reference_value, b.reference_value, rtol=0, atol=0)
    torch.testing.assert_close(a.value, a.assay_values[RADIO.key])
    torch.testing.assert_close(b.value, b.assay_values[FLUOR.key])
    starts = a.diagnostics["candidates"][0]["registers"][0]["starts"]
    assert len(starts) == (1 if mhc_class == "I" else 5)


def test_new_training_requires_one_factorized_path():
    for mode in ("pooled", "calibration"):
        with pytest.raises(ValueError, match="factorized assay_mode"):
            config(mode)


def test_only_observed_factors_receive_direct_loss():
    model = Presto(config())
    observed = row()
    output = model([observed])[0]
    output.assay_values[FLUOR.key].retain_grad()
    observation_loss(output, observed, SCALES).backward()
    for axis in ("endpoint", "preparation", "mechanism", "detector"):
        factor = model.assay_layer.factors[axis][getattr(RADIO, axis)]
        assert factor.grad.abs().sum() > 0
    assert model.assay_layer.factors["detector"]["fluorescence"].grad is None
    assert output.assay_values[FLUOR.key].grad is None
    assert model.encoder.embedding.weight.grad.abs().sum() > 0
    assert model.assay_layer.decoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "qualifier,delta,expected", [("<", -1, 0), ("<=", 1, 1), (">", 1, 0), (">=", -1, 1)]
)
def test_selected_track_keeps_both_inequality_directions(qualifier, delta, expected):
    model = Presto(config())
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
    assert model.assay_layer.decoder[0].weight.grad.abs().sum() > 0
    assert model.assay_layer.factors["endpoint"]["ic50"].grad.abs().sum() > 0
    for axis in ("preparation", "mechanism", "detector"):
        assert all(parameter.grad is None for parameter in model.assay_layer.factors[axis].values())
    assert Observation.from_dict(row().to_dict()) == row()
    raw.pop("assay_comments")
    assert observation_track(observed) == FLUOR


def test_database_curated_is_distinct_usable_evidence_not_output_identity():
    source = row(FLUOR)
    curated = replace(
        source,
        assay_method=replace(source.assay_method, review_status="database_curated"),
    )
    assert curated.assay_method.usable
    assert observation_track(source) == observation_track(curated) == FLUOR
    assert observation_track(source).key == observation_track(curated).key
    assert Observation.from_dict(curated.to_dict()) == curated


def test_factorized_qualitative_tracks_and_explicit_td50_head():
    model = Presto(expanded_config())
    competitive = qualitative_row()
    direct = qualitative_row(QUAL_DIRECT, value=0)
    a, b = model([competitive, direct])
    assert set(a.assay_values) == {QUAL_COMP.key, QUAL_DIRECT.key}
    assert set(b.assay_values) == set(a.assay_values)
    torch.testing.assert_close(a.value, a.assay_values[QUAL_COMP.key])
    torch.testing.assert_close(b.value, b.assay_values[QUAL_DIRECT.key])
    assert "qualitative_binding" not in model.heads
    assert "td50" in model.heads
    assert set(model.affinity_endpoints) == {"ic50"}
    assert set(model.assay_endpoints) == {"ic50", "qualitative_binding"}
    a.assay_values[QUAL_DIRECT.key].retain_grad()
    observation_loss(a, competitive, {**SCALES, "td50": TargetScale(55, 5, 2)}).backward()
    assert model.assay_layer.factors["endpoint"]["qualitative_binding"].grad.abs().sum() > 0
    assert model.assay_layer.factors["mechanism"]["competitive"].grad.abs().sum() > 0
    assert model.assay_layer.factors["mechanism"]["direct"].grad is None
    assert a.assay_values[QUAL_DIRECT.key].grad is None
    td50 = model([td50_row()])[0]
    assert td50.assay_values == {}
    assert "measurement" not in td50.diagnostics
    torch.testing.assert_close(td50.value, td50.reference_value)
    bound = replace(td50_row(), qualifier=">=")
    prediction = model([bound])[0]
    prediction.value = torch.tensor(2.0, requires_grad=True)
    assert float(observation_loss(prediction, bound, {"td50": TargetScale(55, 5, 2)})) == 0
    prediction.value = torch.tensor(-1.0, requires_grad=True)
    assert float(observation_loss(prediction, bound, {"td50": TargetScale(55, 5, 2)})) == 1


def test_qualitative_binding_reverses_concentration_decoder_polarity():
    layer = Presto(expanded_config()).assay_layer
    hidden = torch.randn(3, expanded_config().d_model)

    def unsigned(endpoint, track):
        vectors = [hidden]
        for axis in ("endpoint", "preparation", "mechanism", "detector"):
            value = endpoint if axis == "endpoint" else getattr(track, axis)
            vectors.append(layer.factors[axis][value].expand(3, layer.embedding_dim))
        return layer.decoder(torch.cat(vectors, dim=-1)).squeeze(-1)

    torch.testing.assert_close(layer.score(hidden, "ic50", RADIO), unsigned("ic50", RADIO))
    torch.testing.assert_close(
        layer.score(hidden, "qualitative_binding", QUAL_COMP),
        -unsigned("qualitative_binding", QUAL_COMP),
    )


def test_route_validation_and_unknown_registry():
    with pytest.raises(ValueError, match="biological endpoint route"):
        PrestoConfig(assay_tracks=(replace(RADIO, preparation="cellular"),))
    cellular = AssayTrack("tcell_response", "cellular", "functional", "fluorescence")
    with pytest.raises(ValueError, match="factorized assay-track endpoint"):
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


def test_affinity_checkpoint_extends_tracks_and_trains_new_outputs_without_drift(tmp_path):
    prior = Presto(config())
    train_step(
        prior,
        torch.optim.AdamW(prior.parameters(), lr=0.01),
        [row(RADIO), row(FLUOR)],
        SCALES,
    )
    prior.eval()
    affinity_rows = [row(RADIO), row(FLUOR)]
    expected = [output.value.detach().clone() for output in prior(affinity_rows)]
    path = tmp_path / "affinity.pt"
    save_checkpoint(path, prior, SCALES)

    expanded = Presto(expanded_config())
    report = initialize_from_checkpoint(expanded, path)
    assert set(report["added_assay_tracks"]) == {QUAL_COMP.key, QUAL_DIRECT.key}
    assert report["removed_assay_tracks"] == []
    assert expanded.assay_support[QUAL_COMP.key] == expanded.assay_support[QUAL_DIRECT.key] == 0
    inherited = {name: value.detach().clone() for name, value in prior.state_dict().items()}

    expanded.eval()
    for before, after in zip(expected, expanded(affinity_rows), strict=True):
        assert torch.equal(before, after.value)
    freeze_dependencies(expanded, ["ic50"])
    optimizer = torch.optim.AdamW(
        (parameter for parameter in expanded.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    train_step(
        expanded,
        optimizer,
        [qualitative_row(), qualitative_row(QUAL_DIRECT, value=0), td50_row()],
        {**SCALES, "td50": TargetScale(55, 5, 2)},
    )
    assert all(torch.equal(value, expanded.state_dict()[name]) for name, value in inherited.items())
    expanded.eval()
    for before, after in zip(expected, expanded(affinity_rows), strict=True):
        # The byte-exact state assertion above is the retention contract.  Linux
        # BLAS may choose a different reduction kernel after the mixed-size
        # training batch, so redundant inference can differ by one float32 ULP.
        torch.testing.assert_close(
            before,
            after.value,
            rtol=0,
            atol=torch.finfo(after.value.dtype).eps,
        )
    assert expanded.assay_support[QUAL_COMP.key] == 1
    assert expanded.assay_support[QUAL_DIRECT.key] == 1


@pytest.mark.parametrize("version", [5, 6])
def test_historical_checkpoint_keeps_exact_inference(tmp_path, version):
    import importlib

    legacy = importlib.import_module(f"presto.legacy.shared_v{version}")
    args = dict(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0)
    if version == 6:
        args["assay_tracks"] = (RADIO, FLUOR)
    model = legacy.Presto(legacy.PrestoConfig(**args)).eval()
    if version == 6:
        with torch.no_grad():
            model.assay_layer.projection.weight.fill_(0.2)
        model.assay_support[RADIO.key] = 7
    path = tmp_path / "historical.pt"
    payload = {
        "format": legacy.MODEL_VERSION,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "scales": {key: asdict(value) for key, value in SCALES.items()},
        "support": model.support,
        "trained_parameters": [],
        "trained_features": {key: sorted(value) for key, value in model.trained_features.items()},
        "exposed_features": {key: sorted(value) for key, value in model.exposed_features.items()},
        "source_label_support": {
            key: sorted(value) for key, value in model.source_label_support.items()
        },
        "element_coverage": {
            "masks": model.trained_elements,
            "complete_history": sorted(model.element_history_complete),
        },
    }
    if version == 6:
        payload["assay_support"] = model.assay_support
    torch.save(payload, path)
    historical, _, _ = load_checkpoint(path)
    historical.eval()
    torch.testing.assert_close(
        model([row()])[0].value, historical([row()])[0].value, rtol=0, atol=0
    )
    if version == 6:
        assert historical.assay_support == model.assay_support
        for key, value in model([row()])[0].assay_values.items():
            torch.testing.assert_close(
                value, historical([row()])[0].assay_values[key], rtol=0, atol=0
            )
    assert historical.element_history_complete == model.element_history_complete
    payload.pop("element_coverage")
    torch.save(payload, path)
    older, _, _ = load_checkpoint(path)
    assert not older.element_history_complete
    with pytest.raises(ValueError, match="historical"):
        initialize_from_checkpoint(Presto(config()), path)


def test_assay_metrics_use_selected_output_in_physical_target_space():
    model = Presto(config())
    rows = [row(), row(FLUOR)]
    metrics, records = evaluate(model, rows, [0, 1], SCALES)
    expected = [SCALES["ic50"].decode(float(x.value.detach())) for x in model(rows)]
    assert [x["prediction"] for x in records] == pytest.approx(expected)
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


def test_common_manifest_trainer_fits_mixed_assay_aware_outputs(tmp_path):
    from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, write_observations
    from presto.training.modular_run import fit_index

    templates = [
        replace(row(RADIO), value=10.0),
        replace(row(FLUOR), value=1000.0),
        qualitative_row(QUAL_COMP, value=1),
        qualitative_row(QUAL_DIRECT, value=0),
        td50_row(45.0),
        td50_row(65.0),
    ]
    peptides = [
        "ACDEFGHIK",
        "LMNPQRSTV",
        "WYACDEFGH",
        "IKLMNPQRS",
        "TVWYACDEF",
        "GHIKLMNPQ",
        "RSTVWYACD",
        "EFGHIKLMN",
        "PQRSTVWYA",
        "CDEFGHIKL",
    ]
    rows = [
        replace(
            template,
            observation_id=f"train-{i}",
            peptide=peptides[i],
            group_id=f"train-{i}",
            provenance={"original_measurement_id": f"train-{i}"},
        )
        for i, template in enumerate(templates)
    ]
    rows += [
        replace(
            template,
            observation_id=f"validation-{i}",
            peptide=peptides[len(templates) + i],
            group_id=f"validation-{i}",
            split="validation",
            provenance={"original_measurement_id": f"validation-{i}"},
        )
        for i, template in enumerate(
            (row(RADIO), qualitative_row(), qualitative_row(QUAL_DIRECT, value=0), td50_row())
        )
    ]
    source = write_observations(tmp_path / "mixed.jsonl", rows)
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": asdict(replace(expanded_config(), mhc_classes=("I",))),
        "training": {
            "max_epochs": 4,
            "max_steps": 4,
            "batch_size": len(templates),
            "validation_every_steps": 4,
            "test_policy": "no_test_training_diagnostic",
            "no_test_reason": "artificial mixed-output software fixture",
        },
        "sources": [
            {
                "path": str(tmp_path / "mixed.jsonl"),
                "sha256": source["sha256"],
                "source": "software-test",
                "population": "artificial mixed assay-aware outputs",
                "label_policy": "artificial literal endpoints and methods",
                "split_policy": "disjoint artificial groups",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    summary = fit_index(ObservationIndex(path), tmp_path / "mixed-fit")
    assert summary["coverage_gate"]["status"] == "passed"
    assert set(summary["metrics"]["validation"]) == {
        "ic50",
        "qualitative_binding",
        "td50",
    }
    assert len(summary["metrics"]["validation"]["qualitative_binding"]["by_assay_method"]) == 2
    assert set(summary["assay_tracks"]) == {
        RADIO.key,
        FLUOR.key,
        QUAL_COMP.key,
        QUAL_DIRECT.key,
    }
    assert summary["support"]["qualitative_binding"]["status"] == "directly_supervised"
    assert summary["support"]["td50"]["status"] == "directly_supervised"


def test_all_affinity_quantities_and_classes_train_one_decoder():
    from presto.models.assay_measurement import AFFINITY_ENDPOINTS

    endpoints = tuple(sorted(AFFINITY_ENDPOINTS))
    tracks = tuple(replace(RADIO, endpoint=endpoint) for endpoint in endpoints)
    model = Presto(replace(config(endpoints=endpoints), assay_tracks=tracks))
    rows = [replace(row(), endpoint=endpoint, observation_id=endpoint) for endpoint in endpoints]
    rows += [replace(rows[0], mhc=(replace(rows[0].mhc[0], mhc_class="II"),))]
    assert not set(endpoints) & set(model.heads)
    called = []
    handle = model.assay_layer.decoder.register_forward_hook(
        lambda module, args, out: called.append(id(module))
    )
    result = train_step(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.01),
        rows,
        {e: SCALES["ic50"] for e in endpoints},
    )
    handle.remove()
    assert called and set(called) == {id(model.assay_layer.decoder)}
    for endpoint in endpoints:
        assert result["coverage"][f"assay_layer.factors.endpoint.{endpoint}"]["gradient_and_update"]
        assert model.support[endpoint]["status"] == "directly_supervised"
    assert all(x["status"] == "directly_supervised" for x in model.assay_capabilities().values())
    assert result["coverage"]["assay_layer.decoder.0.weight"]["gradient_and_update"]
    assert result["coverage"]["encoder.embedding.weight"]["gradient_and_update"]


def test_direct_decoder_learns_assay_dependent_ordering():
    """Artificial numbers verify expressivity, never biological prediction accuracy."""
    from itertools import product
    from presto.models.assay_measurement import AssayMeasurementLayer

    torch.manual_seed(551)
    tracks = tuple(
        AssayTrack(*values)
        for values in product(
            ("ic50", "kd"),
            ("purified", "cellular"),
            ("competitive", "direct"),
            ("radioactivity", "fluorescence"),
        )
    )
    layer = AssayMeasurementLayer(16, ("ic50", "kd"), tracks)
    x = torch.linspace(-1, 1, 8)
    hidden = torch.zeros(8, 16)
    hidden[:, 0] = x

    def targets(track):
        sign = 1 if track.detector == "radioactivity" else -1
        return (
            sign * x
            + 0.4 * (track.endpoint == "kd")
            + 0.3 * (track.preparation == "cellular")
            + 0.2 * (track.mechanism == "direct")
        )

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    for _ in range(350):
        optimizer.zero_grad()
        loss = torch.stack(
            [(layer.score(hidden, t.endpoint, t) - targets(t)).square().mean() for t in tracks]
        ).mean()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.002
    radio = layer.score(hidden, RADIO.endpoint, RADIO).detach()
    fluor = layer.score(hidden, FLUOR.endpoint, FLUOR).detach()
    assert (radio[1:] > radio[:-1]).all()
    assert (fluor[1:] < fluor[:-1]).all()
