"""Quantity contracts and routing; artificial labels establish software behavior only."""

from dataclasses import asdict, replace
import json
import math

import pytest
import torch

from presto.data.modular import (
    Context,
    MHCInput,
    Observation,
    TargetScale,
    endpoint_spec,
    fit_target_scales,
    observation_target,
)
from presto.data.modular_quantitative import QuantitativeDomain
from presto.data.modular_ordinal import OrdinalScale, OrdinalValue
from presto.data.modular_lineage import semantic_fingerprint
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    normalized_observation,
    write_observations,
)
from presto.inference.modular import ModularPredictor
from presto.models.modular import ModularConfig, ModularPresto, Prediction
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    observation_loss,
    save_checkpoint,
    train_step,
)
from presto.training.modular_evaluation import evaluate
from presto.training.modular_run import fit_index, preflight


DOMAIN = QuantitativeDomain(
    "relative instrument response",
    "declared_signal",
    "log2",
    "software fixture normalization",
    "software-only:domain-v1",
)
PAIR = MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I")
CONTEXT_SCALE = OrdinalScale(("low", "high"), "software-only:context-v1")


def sample(family="bulk_quantity", domain="fixture", **kwargs):
    endpoint = f"{family}:{domain}"
    spec = endpoint_spec(endpoint)
    provenance = {"original_measurement_id": f"m-{family}", "measured_outcome": True}
    if len(spec.parts) == 1:
        provenance["component_measurement"] = {
            "component": spec.parts[0],
            "basis": "independent_component_measurement",
            "reference": "software-only",
        }
    values = dict(
        observation_id=family,
        source="quantity-software-fixture",
        endpoint=endpoint,
        peptide="ACDEFGHIK",
        mhc=(PAIR,) if spec.route == "cellular" else (),
        value=8.0,
        unit=DOMAIN.unit,
        quantity_contract=DOMAIN,
        group_id="group",
        provenance=provenance,
    )
    values.update(kwargs)
    return Observation(**values)


def network(families=("bulk_quantity",), **kwargs):
    settings = dict(
        endpoints=tuple(f"{name}:fixture" for name in families),
        quantity_domains={"fixture": DOMAIN},
        mhc_classes=("I",),
        dropout=0,
    )
    settings.update(kwargs)
    return ModularPresto(ModularConfig(**settings)).eval()


def test_explicit_log_encoding_and_censor_residuals():
    original = sample()
    encoded = replace(original, value=3, value_encoding="log2")
    assert observation_target(original) == observation_target(encoded) == 3
    assert semantic_fingerprint(original) == semantic_fingerprint(encoded)
    assert Observation.from_dict(encoded.to_dict()) == encoded
    log10 = replace(DOMAIN, transform="log10")
    assert log10.convert(3, DOMAIN.unit, "log2") == pytest.approx(math.log10(8))
    assert DOMAIN.convert(1000, DOMAIN.unit, "log10") == pytest.approx(1000 * math.log2(10))
    assert DOMAIN.inverse(3) == 8
    assert DOMAIN.inverse(10000) is None
    with pytest.raises(ValueError, match="floating-point"):
        replace(DOMAIN, transform="identity").convert(10000, DOMAIN.unit, "log10")
    with pytest.raises(ValueError, match="positive"):
        DOMAIN.convert(0, DOMAIN.unit)
    with pytest.raises(ValueError, match="unit mismatch"):
        DOMAIN.convert(8, "a different scale")
    scales = {original.endpoint: TargetScale(100, 2, 2)}
    prediction = Prediction(original.observation_id, original.endpoint, torch.tensor(5.0))
    assert (
        observation_loss(prediction, original, scales) == 1
    )  # raw transformed units, not z-score prediction
    assert observation_loss(prediction, replace(original, qualifier=">"), scales) == 0
    assert observation_loss(prediction, replace(original, qualifier="<"), scales) == 1
    fitted = fit_target_scales(
        [
            original,
            replace(original, value=32),
            replace(original, value=1e9, qualifier=">"),
            replace(original, value=1e8, split="test"),
        ]
    )
    assert fitted[original.endpoint] == TargetScale(4, 1, 2)
    with pytest.raises(ValueError, match="incompatible quantity"):
        fit_target_scales(
            [
                original,
                replace(original, value=32, quantity_contract=replace(DOMAIN, normalization="raw")),
            ]
        )


def test_domains_reject_silent_normalization_or_identity_mix():
    model = network()
    with pytest.raises(ValueError, match="contract differs"):
        model([sample(quantity_contract=replace(DOMAIN, normalization="raw"))])
    with pytest.raises(ValueError, match="measurement contract"):
        sample(quantity_contract=None).validate()
    with pytest.raises(ValueError, match="quantity contracts"):
        replace(sample(), endpoint="ic50", mhc=(PAIR,), unit="nM").validate()
    with pytest.raises(ValueError, match="declared measurement domains"):
        ModularConfig(endpoints=("bulk_quantity:missing",))
    with pytest.raises(ValueError, match="unsupported quantity endpoint"):
        endpoint_spec("bulk_quantity:bad.name")
    for field in ("quantity", "unit", "normalization", "reference"):
        with pytest.raises(ValueError, match="requires meaning"):
            replace(DOMAIN, **{field: ""})


def test_shared_component_readouts_preparation_masks_and_subset_allocation():
    families = (
        "bulk_quantity",
        "elution_quantity",
        "ms_quantity",
        "digest_quantity",
        "cellular_quantity",
    )
    model = network(
        families,
        enzymes=("digest_a", "digest_b"),
        preparation_features=("duration",),
        preparation_units={"duration": "hours"},
    )
    rows = [
        sample(
            family,
            context=Context(
                enzyme="digest_a",
                preparation={"duration": 2},
                preparation_units={"duration": "hours"},
            ),
        )
        for family in families
    ]
    outputs = model(rows)
    for index in (1, 2):
        torch.testing.assert_close(
            outputs[index].components["ms_response"], outputs[0].components["ms_response"]
        )
    torch.testing.assert_close(outputs[3].components["digest"], outputs[0].components["digest"])
    torch.testing.assert_close(outputs[4].components["cellular"], outputs[1].components["cellular"])
    changed = replace(
        rows[0], context=replace(rows[0].context, enzyme="digest_b", preparation={"duration": 8})
    )
    altered = model([changed])[0]
    torch.testing.assert_close(
        altered.components["ms_response"], outputs[0].components["ms_response"]
    )
    assert not torch.allclose(altered.components["digest"], outputs[0].components["digest"])
    outputs[0].value.backward()
    assert model.quantity_scores["fixture"]["ms_response"].weight.grad.norm() > 0
    labels = model.feature_layout["digest_fusion.0.weight"][1]
    assert (
        model.digest_fusion[0].weight.grad[:, labels.index("preparation:duration:value")].norm() > 0
    )
    assert model.quantity_scores["fixture"]["cellular"].weight.grad is None
    assert not hasattr(model, "ms_score") and not hasattr(model, "digest_score")
    assert not model.heads
    only_detector = network(("ms_quantity",))
    assert not hasattr(only_detector, "digest_fusion") and not hasattr(only_detector, "pmhc_fusion")
    assert set(only_detector.quantity_scores["fixture"]) == {"ms_response"}
    two_scales = network(
        endpoints=("bulk_quantity:fixture", "elution_quantity:other"),
        quantity_domains={
            "fixture": DOMAIN,
            "other": replace(DOMAIN, normalization="another normalization"),
        },
    )
    assert (
        two_scales.quantity_scores["fixture"]["ms_response"]
        is not two_scales.quantity_scores["other"]["ms_response"]
    )


def test_quantity_candidate_bag_has_one_loss_and_correct_log_base():
    model = network(("elution_quantity",))
    original = sample("elution_quantity")
    repeated = replace(original, mhc=(PAIR, replace(PAIR, provenance={"export": "copy"})))
    outputs = model([original, repeated])
    torch.testing.assert_close(outputs[0].value, outputs[1].value)
    scores = torch.tensor([1.0, 3.0])
    assert model._candidate_mean(original.endpoint, scores) == pytest.approx(math.log2((2 + 8) / 2))
    linear = network(
        ("elution_quantity",), quantity_domains={"fixture": replace(DOMAIN, transform="identity")}
    )
    assert linear._candidate_mean(original.endpoint, scores) == 2
    assert len(outputs[1].diagnostics["candidates"]) == 1


def test_binary_and_quantity_outputs_share_features_but_keep_distinct_scales():
    model = network(endpoints=("bulk_observation", "bulk_quantity:fixture", "ic50"))
    quantity = sample()
    binary = replace(
        quantity, endpoint="bulk_observation", value=1, unit="binary", quantity_contract=None
    )
    affinity = replace(binary, endpoint="ic50", value=100, unit="nM", mhc=(PAIR,))
    outputs = model([quantity, binary, affinity])
    assert all(output.value.ndim == 0 for output in outputs)
    assert model.ms_score is not model.quantity_scores["fixture"]["ms_response"]
    outputs[0].value.backward()
    assert model.ms_fusion[0].weight.grad.norm() > 0
    assert model.quantity_scores["fixture"]["ms_response"].weight.grad.norm() > 0
    assert model.ms_score.weight.grad is None and model.heads["ic50"].weight.grad is None
    scopes = fit_target_scales(
        [quantity, replace(quantity, value=32), affinity, replace(affinity, value=1000)]
    )
    assert set(scopes) == {"bulk_quantity:fixture", "ic50"}


def test_both_mhc_classes_update_the_same_quantity_core():
    torch.manual_seed(103)
    model = network(
        endpoints=("elution_quantity:fixture", "bulk_quantity:fixture", "ic50"),
        mhc_classes=("I", "II"),
        ordinal_scales={"factor_ratio": CONTEXT_SCALE},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for mhc_class in ("I", "II"):
        rows = [
            sample(
                "elution_quantity",
                observation_id=f"{mhc_class}-{level}",
                peptide="ACDEFGHIKL",
                mhc=(replace(PAIR, mhc_class=mhc_class),),
                context=Context(
                    ordinal_values={"factor_ratio": OrdinalValue(level, CONTEXT_SCALE.reference)}
                ),
                value=value,
            )
            for level, value in (("low", 8.0), ("high", 32.0))
        ]
        step = train_step(model, optimizer, rows, fit_target_scales(rows))
        for name in (
            "encoder.embedding.weight",
            "pmhc_fusion.0.weight",
            "context_fusion.0.weight",
            "ms_fusion.0.weight",
            "quantity_scores.fixture.cellular.weight",
            "quantity_scores.fixture.ms_response.weight",
        ):
            parameter = model.get_parameter(name)
            assert step["coverage"][name]["gradient_and_update"]
            assert optimizer.state[parameter]["step"] == (1 if mhc_class == "I" else 2)
        assert model.heads["ic50"].weight.grad is None
        assert model.digest_fusion[0].weight.grad is None
        if mhc_class == "I":
            assert model.register_score.weight.grad is None
        else:
            assert model.register_score.weight.grad.norm() > 0
        labels = model.feature_layout["pmhc_fusion.0.weight"][1]
        # The existing feature-coverage gate must see both class channels after both updates.
        if mhc_class == "II":
            assert all(
                labels.index(name) in model.trained_features["pmhc_fusion.0.weight"]
                for name in ("class:I", "class:II")
            )
        model.eval()
        bulk = sample(peptide=rows[0].peptide)
        low, high, bulk_output = model([*rows, bulk])
        assert not torch.allclose(low.value, high.value)
        for other in (high, bulk_output):
            torch.testing.assert_close(
                low.components["ms_response"], other.components["ms_response"], rtol=0, atol=0
            )


def test_anchor_evidence_and_support_are_separate_from_mixtures():
    anchor = sample("ms_quantity")
    with pytest.raises(ValueError, match="independent measurement evidence"):
        normalized_observation(
            replace(anchor, provenance={"measured_outcome": True}).to_dict(),
            evidence_policy="fixture",
        )
    with pytest.raises(ValueError, match="attested measured"):
        normalized_observation(replace(anchor, provenance={}).to_dict(), evidence_policy="fixture")
    assert normalized_observation(anchor.to_dict(), evidence_policy="fixture").value == 8
    model = network(("bulk_quantity", "ms_quantity"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    row = sample()
    train_step(model, optimizer, [row], {row.endpoint: TargetScale(3, 1, 2)})
    status = model.capabilities()["quantity_components"]["fixture"]["ms_response"]
    assert status["status"] == "indirectly_trained" and not status["anchor_endpoints"]
    train_step(model, optimizer, [anchor], {anchor.endpoint: TargetScale(3, 1, 2)})
    status = model.capabilities()["quantity_components"]["fixture"]["ms_response"]
    assert status["status"] == "directly_supervised" and status["validation"] == "not_evaluated"
    assert model.capabilities()["components"]["ms_response"]["status"] == "not_allocated"


def test_checkpoint_evaluation_and_public_quantity_units(tmp_path):
    model = network(("ms_quantity",))
    rows = [sample("ms_quantity"), sample("ms_quantity", observation_id="other", value=32)]
    scales = fit_target_scales(rows)
    train_step(model, torch.optim.AdamW(model.parameters(), lr=0.001), rows, scales)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, scales)
    restored, restored_scales, _ = load_checkpoint(path)
    prediction = restored.predict(rows)[0]
    result = ModularPredictor(path).predict(
        [replace(rows[0], value=None, unit=None, split="predict")]
    )[0]
    assert result["transformed_value"] == float(prediction.value)
    assert result["value"] == pytest.approx(2 ** result["transformed_value"])
    assert result["unit"] == DOMAIN.unit and result["quantity_contract"] == asdict(DOMAIN)
    metrics, records = evaluate(restored, rows, [0, 1], restored_scales)
    assert records[0]["prediction"] == float(prediction.value)
    assert records[0]["target"] == 3 and records[0]["target_space"] == DOMAIN.target_space
    assert "binding_at_500_nM" not in metrics[rows[0].endpoint]["exact"]
    incompatible = network(
        ("ms_quantity",), quantity_domains={"fixture": replace(DOMAIN, normalization="other")}
    )
    with pytest.raises(ValueError, match="measurement domain"):
        initialize_from_checkpoint(incompatible, path)


def quantity_index(tmp_path, *, preparation=False, mixed_classes=False):
    peptides = ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH", "IKLMNPQRS", "TVWYACDEF", "GHIKLMNPQ"]
    rows = []
    for i, peptide in enumerate(peptides):
        for family in ("bulk_quantity", "elution_quantity"):
            identifier = f"{family}-{i}"
            rows.append(
                sample(
                    family,
                    observation_id=identifier,
                    peptide=peptide,
                    group_id=f"g{i}",
                    split=("train", "validation", "test")[i // 2],
                    value=8.0 if i % 2 else 32.0,
                    context=Context(
                        preparation={"duration": 2}, preparation_units={"duration": "hours"}
                    )
                    if preparation
                    else Context(),
                    provenance={"original_measurement_id": identifier, "measured_outcome": True},
                )
            )
    if mixed_classes:
        rows = [
            replace(
                row,
                # Vary cores with the same trained flanking-residue support in every split.
                peptide="A" + row.peptide[1:] + "A",
                context=Context(
                    ordinal_values={
                        "factor_ratio": OrdinalValue(
                            "low" if row.value == 8 else "high", CONTEXT_SCALE.reference
                        )
                    }
                ),
            )
            for row in rows
        ]
        rows.extend(
            [
                replace(
                    row,
                    observation_id=row.observation_id + "-II",
                    mhc=(replace(PAIR, mhc_class="II"),),
                    provenance={
                        **row.provenance,
                        "original_measurement_id": row.observation_id + "-II",
                    },
                )
                for row in rows
                if row.mhc
            ]
        )
    receipt = write_observations(tmp_path / "rows.jsonl", rows)
    settings = network(
        ("bulk_quantity", "elution_quantity"),
        preparation_features=("duration",) if preparation else (),
        preparation_units={"duration": "hours"} if preparation else {},
        mhc_classes=("I", "II") if mixed_classes else ("I",),
        ordinal_scales={"factor_ratio": CONTEXT_SCALE} if mixed_classes else {},
    ).config
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": asdict(settings),
        "training": {"max_epochs": 2, "batch_size": 4, "max_steps": 2},
        "sources": [
            {
                "path": receipt["path"],
                "sha256": receipt["sha256"],
                "source": "quantity-software-fixture",
                "population": "artificial",
                "label_policy": "software-only",
                "split_policy": "unique synthetic groups",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return ObservationIndex(path)


@pytest.mark.parametrize("mixed_classes", [False, True])
def test_streamed_quantity_fit_closes_all_active_paths(tmp_path, mixed_classes):
    index = quantity_index(tmp_path, mixed_classes=mixed_classes)
    config, _, _, admission = preflight(index)
    assert config.quantity_domains == {"fixture": DOMAIN}
    assert admission["training_labels"] == {
        "bulk_quantity:fixture": 2,
        "elution_quantity:fixture": 4 if mixed_classes else 2,
    }
    output = tmp_path / "output"
    output.mkdir()
    summary = fit_index(index, output)
    assert summary["coverage_gate"]["status"] == "passed"
    records = [
        json.loads(line) for line in (output / "test-predictions.jsonl").read_text().splitlines()
    ]
    assert len(records) == (6 if mixed_classes else 4)
    assert all(row["quantity_contract"] == asdict(DOMAIN) for row in records)
    model, _, _ = load_checkpoint(output / "selected.pt")
    assert not hasattr(model, "ms_score") and not hasattr(model, "cellular_score")
    assert (
        model.capabilities()["quantity_components"]["fixture"]["ms_response"]["status"]
        == "indirectly_trained"
    )
    if mixed_classes:
        assert model.config.mhc_classes == ("I", "II")
        assert model.config.ordinal_scales == {"factor_ratio": CONTEXT_SCALE}
        rows = [Observation.from_dict(record["observation"]) for record in records]
        predictor = ModularPredictor(output / "selected.pt")
        batch_size = index.manifest["training"]["batch_size"]
        public = [
            result
            for start in range(0, len(rows), batch_size)
            for result in predictor.predict(
                [
                    replace(row, value=None, split="predict")
                    for row in rows[start : start + batch_size]
                ]
            )
        ]
        for saved, predicted in zip(records, public, strict=True):
            assert predicted["transformed_value"] == saved["prediction"]
        candidates = [
            result["diagnostics"]["candidates"][0]
            for result in public
            if "candidates" in result["diagnostics"]
        ]
        assert {row["mhc"]["mhc_class"] for row in candidates} == {"I", "II"}
        for row in candidates:
            assert row["registers"][0]["starts"] == (
                [0] if row["mhc"]["mhc_class"] == "I" else [0, 1]
            )


def test_preparation_admission_requires_variation(tmp_path):
    with pytest.raises(ValueError, match="preparation feature lacks varied"):
        preflight(quantity_index(tmp_path, preparation=True))
