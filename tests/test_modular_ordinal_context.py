"""Artificial fixtures for source-relative levels, never biological validation."""

from dataclasses import asdict, replace
import json

import pytest
import torch

from presto.data.modular import Context, MHCInput, Observation
from presto.data.modular_context import resolve_cell_context
from presto.data.modular_lineage import semantic_fingerprint
from presto.data.modular_ordinal import OrdinalScale, OrdinalValue
from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, write_observations
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from presto.training.modular_run import preflight

SCALE = OrdinalScale(("low", "medium", "high"), "artificial-comparison-frame-v1")


def settings(**changes):
    return ModularConfig(
        **(
            {
                "endpoints": ("elution_observation",),
                "mhc_classes": ("II",),
                "dropout": 0,
                "ordinal_scales": {"factor_ratio": SCALE},
            }
            | changes
        )
    )


def context(level="low"):
    value = (
        OrdinalValue(level, SCALE.reference, {"source": "artificial-fixture"})
        if level
        else OrdinalValue()
    )
    return Context(ordinal_values={"factor_ratio": value})


def observation(**changes):
    return Observation(
        **(
            {
                "observation_id": "fixture",
                "source": "software-fixture",
                "endpoint": "elution_observation",
                "peptide": "ACDEFGHIKL",
                "mhc": (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "II"),),
                "context": context(),
                "value": 1,
                "unit": "binary",
                "group_id": "fixture-group",
            }
            | changes
        )
    )


def resolve(value, **changes):
    return resolve_cell_context(
        value, (), (), {}, (), ordinal_scales={"factor_ratio": SCALE}, **changes
    )


@pytest.mark.parametrize(
    "levels", [("low",), ("low", "low"), ("low", "unknown"), ("low", ""), ("low", 1)]
)
def test_scale_rejects_ambiguous_vocabularies(levels):
    with pytest.raises(ValueError, match="distinct ordered level"):
        OrdinalScale(levels, "reference")


def test_scale_requires_reference_and_structured_levels():
    with pytest.raises(ValueError, match="explicit reference"):
        OrdinalScale(("low", "high"), "")
    with pytest.raises(ValueError, match="ordered sequence"):
        OrdinalScale.from_dict({"levels": "low", "reference": "frame"})
    with pytest.raises(ValueError, match="cellular outcome"):
        settings(endpoints=("ic50",))
    with pytest.raises(ValueError, match="named typed scales"):
        settings(ordinal_scales={"ratio": {"levels": ["low", "high"], "reference": "frame"}})


@pytest.mark.parametrize(
    "level,scale",
    [(None, "frame"), ("low", None), ("", "frame"), ("unknown", "frame"), (True, "frame")],
)
def test_value_rejects_missing_scale_and_unknown_aliases(level, scale):
    with pytest.raises(ValueError, match="ordinal value"):
        OrdinalValue(level, scale)


def test_override_unknown_and_roundtrip_retain_provenance():
    supplied = replace(context("high"), ordinal_overrides={"factor_ratio": OrdinalValue()})
    resolved = resolve(supplied)
    assert resolved.ordinal_values["factor_ratio"].level is None
    assert resolved.ordinal_sources == {"factor_ratio": "override"}
    assert resolved.supplied_ordinals["factor_ratio"].level == "high"
    assert resolved.supplied_ordinals["factor_ratio"].evidence == {"source": "artificial-fixture"}
    row = observation(context=supplied)
    assert Observation.from_dict(row.to_dict()) == row
    assert ModularConfig.from_dict(json.loads(json.dumps(asdict(settings())))) == settings()
    other_evidence = replace(
        row,
        context=replace(
            supplied,
            ordinal_values={
                "factor_ratio": OrdinalValue("high", SCALE.reference, {"different": "locator"})
            },
        ),
    )
    assert semantic_fingerprint(other_evidence) == semantic_fingerprint(row)
    for value in (OrdinalValue("medium", SCALE.reference), OrdinalValue("high", "other-frame")):
        changed = replace(row, context=replace(supplied, ordinal_values={"factor_ratio": value}))
        assert semantic_fingerprint(changed) != semantic_fingerprint(row)


def test_no_defaults_and_strict_scale_validation():
    assert resolve(Context(species={"apc": "human"}, cell_type="fixture_cell")).ordinal_values == {
        "factor_ratio": OrdinalValue()
    }
    with pytest.raises(ValueError, match="scale reference mismatch"):
        resolve(Context(ordinal_values={"factor_ratio": OrdinalValue("low", "other-frame")}))
    with pytest.raises(ValueError, match="level outside checkpoint"):
        resolve(Context(ordinal_values={"factor_ratio": OrdinalValue("extra", SCALE.reference)}))
    with pytest.raises(ValueError, match="ordinal context outside checkpoint"):
        resolve(Context(ordinal_values={"other_ratio": OrdinalValue("low", SCALE.reference)}))
    ignored = resolve_cell_context(context(), (), (), {}, (), ordinal_scales={})
    assert ignored.ignored_fields == ["ordinal:factor_ratio"]
    with pytest.raises(ValueError, match="named typed values"):
        Context(ordinal_values={"factor_ratio": "low"}).validate()
    cleared = Context(
        ordinal_values={"factor_ratio": OrdinalValue("low", "other-frame")},
        ordinal_overrides={"factor_ratio": OrdinalValue()},
    )
    assert resolve(cleared).ordinal_values["factor_ratio"].level is None


def test_levels_are_distinct_without_numerical_spacing():
    torch.manual_seed(73)
    model = ModularPresto(settings()).eval()
    inputs = []
    handle = model.context_fusion[0].register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0].detach().clone())
    )
    rows = [observation(context=context(level)) for level in (*SCALE.levels, None)]
    try:
        outputs = model(rows)
    finally:
        handle.remove()
    names = model.feature_layout["context_fusion.0.weight"][1]
    positions = [
        names.index(f"ordinal:factor_ratio:{level}") for level in (*SCALE.levels, "unknown")
    ]
    # Each two-register peptide generates a contextual vector per register.
    observed = torch.stack(inputs).reshape(-1, len(names))[:, positions]
    assert {tuple(row.tolist()) for row in observed} == set(map(tuple, torch.eye(4).tolist()))
    assert len({float(output.value.detach()) for output in outputs}) == 4
    assert all(SCALE.reference not in name for name in names)


@pytest.mark.parametrize("mhc_class", ["I", "II"])
def test_noncellular_outputs_and_ms_component_ignore_ordinal_inputs(mhc_class):
    torch.manual_seed(79)
    model = ModularPresto(
        settings(
            mhc_classes=("I", "II"),
            endpoints=(
                "ic50",
                "tcr_binding",
                "elution_observation",
                "bulk_observation",
                "ms_response",
                "digest_observation",
            ),
        )
    ).eval()
    for endpoint in model.config.endpoints:
        row = observation(
            endpoint=endpoint,
            mhc=(replace(observation().mhc[0], mhc_class=mhc_class),),
            tcr_a="ACDEFGHIKLM",
            tcr_b="NPQRSTVWYAC",
            value=10 if endpoint == "ic50" else 1,
            unit="nM" if endpoint == "ic50" else "binary",
        )
        low, high = model([row, replace(row, context=context("high"))])
        if endpoint == "elution_observation":
            assert not torch.allclose(low.value, high.value)
        else:
            torch.testing.assert_close(low.value, high.value, rtol=0, atol=0)
            wrong_frame = replace(
                row,
                context=Context(
                    ordinal_values={"factor_ratio": OrdinalValue("low", "unallocated-frame")}
                ),
            )
            torch.testing.assert_close(model([wrong_frame])[0].value, low.value, rtol=0, atol=0)
        if "ms_response" in low.components:
            torch.testing.assert_close(
                low.components["ms_response"], high.components["ms_response"], rtol=0, atol=0
            )


def test_mapping_order_does_not_reorder_features_or_weights():
    a = {"a_ratio": SCALE, "z_ratio": SCALE}
    b = dict(reversed(list(a.items())))
    torch.manual_seed(83)
    first = ModularPresto(settings(ordinal_scales=a))
    torch.manual_seed(83)
    second = ModularPresto(settings(ordinal_scales=b))
    assert first.feature_layout == second.feature_layout
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


def test_training_support_and_checkpoint_contract(tmp_path):
    torch.manual_seed(89)
    model = ModularPresto(settings())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rows = [observation(value=0), observation(observation_id="high", context=context("high"))]
    train_step(model, optimizer, rows, {})
    names = model.feature_layout["context_fusion.0.weight"][1]
    trained = {names[i] for i in model.trained_features["context_fusion.0.weight"]}
    assert {"ordinal:factor_ratio:low", "ordinal:factor_ratio:high"} <= trained
    assert "ordinal:factor_ratio:medium" not in trained
    expected = model.eval().predict(rows)[0].value
    path = tmp_path / "ordinal.pt"
    save_checkpoint(path, model, {})
    restored, _, _ = load_checkpoint(path)
    torch.testing.assert_close(restored.predict(rows)[0].value, expected, rtol=0, atol=0)
    for level in ("medium", None):
        with pytest.raises(ValueError, match="untrained input features.*ordinal"):
            restored.predict([observation(context=context(level))])
    for changed in (
        OrdinalScale(SCALE.levels, "other-frame"),
        OrdinalScale(tuple(reversed(SCALE.levels)), SCALE.reference),
    ):
        target = ModularPresto(settings(ordinal_scales={"factor_ratio": changed}))
        with pytest.raises(ValueError, match="ordinal_scales"):
            initialize_from_checkpoint(target, path)
    target = ModularPresto(settings(ordinal_scales={"other_ratio": SCALE}))
    with pytest.raises(ValueError, match="ordinal_scales"):
        initialize_from_checkpoint(target, path)


def test_preflight_requires_all_levels_in_training_cellular_rows(tmp_path):
    rows = [
        observation(
            peptide=peptide,
            observation_id=f"r{i}",
            group_id=f"g{i}",
            split="train" if i < 3 else "validation" if i == 3 else "test",
            context=context(("low", "medium", "high", "low", "medium")[i]),
            provenance={"original_measurement_id": f"m{i}"},
        )
        for i, peptide in enumerate(
            ("ACDEFGHIKL", "MNPQRSTVWY", "ACFGHILMNP", "KLMNPQRSTV", "AEHLPSWYCF")
        )
    ]

    def index(records, name):
        receipt = write_observations(tmp_path / f"{name}.jsonl", records)
        manifest = {
            "schema": MANIFEST_VERSION,
            "model": asdict(settings()),
            "sources": [
                {
                    "path": receipt["path"],
                    "sha256": receipt["sha256"],
                    "source": "software-fixture",
                    "population": "artificial",
                    "label_policy": "software-only",
                    "split_policy": "distinct fixtures",
                }
            ],
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(manifest))
        return ObservationIndex(path)

    assert preflight(index(rows, "valid"))[3]["ordinal_levels"] == {
        "factor_ratio": ["high", "low", "medium"]
    }
    for level in ("low", None):
        altered = [
            replace(row, context=context(level)) if i == 1 else row for i, row in enumerate(rows)
        ]
        with pytest.raises(ValueError, match="ordinal levels lack training support.*medium"):
            preflight(index(altered, f"missing-{level}"))


def test_checkpoint_and_observation_without_optional_extension(tmp_path):
    torch.manual_seed(97)
    model = ModularPresto(settings(ordinal_scales={}))
    row = observation(context=Context())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_step(model, optimizer, [row], {})
    expected = model.eval().predict([row])[0].value
    path = tmp_path / "without-ordinal-extension.pt"
    save_checkpoint(path, model, {})
    payload = torch.load(path, weights_only=False)
    assert payload["config"].pop("ordinal_scales") == {}
    torch.save(payload, path)
    legacy = row.to_dict()
    assert legacy["context"].pop("ordinal_values") == {}
    assert legacy["context"].pop("ordinal_overrides") == {}
    restored_row = Observation.from_dict(legacy)
    assert restored_row == row
    assert semantic_fingerprint(restored_row) == semantic_fingerprint(row)
    restored, _, _ = load_checkpoint(path)
    torch.testing.assert_close(restored.predict([restored_row])[0].value, expected, rtol=0, atol=0)
