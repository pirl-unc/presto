"""Cellular metadata contracts using artificial observations only."""

from dataclasses import asdict, replace
import json

import pytest
import torch

from presto.data.modular import (
    Component,
    Context,
    Exposure,
    MachineryProfile,
    MHCInput,
    Observation,
)
from presto.data.modular_context import resolve_cell_context, resolve_components, resolve_field
from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, write_observations
from presto.data.modular_lineage import semantic_fingerprint
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from presto.training.modular_run import preflight


def example(endpoint="elution_observation", **kwargs):
    values = dict(
        observation_id="example",
        source="software-fixture",
        endpoint=endpoint,
        peptide="ACDEFGHIK",
        mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I"),),
        value=1,
        unit="binary",
        group_id="group",
    )
    values.update(kwargs)
    return Observation(**values)


def context(present=False):
    return Context(
        cell_type="cell_b" if present else "cell_a",
        cytokines={
            "signal_a": Exposure(
                "present" if present else "absent",
                2 if present else 0,
                "ng/mL",
                24 if present else 0,
            ),
            "signal_b": Exposure("present" if present else "absent"),
        },
    )


def config(**kwargs):
    values = dict(
        endpoints=("elution_observation",),
        mhc_classes=("I",),
        dropout=0,
        cell_types=("cell_a", "cell_b"),
        cytokines=("signal_a", "signal_b"),
        cytokine_dose_units={"signal_a": "ng/mL"},
        cytokine_duration_features=("signal_a",),
    )
    values.update(kwargs)
    return ModularConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"state": "invalid"},
        {"state": "present", "dose": -1, "unit": "ng/mL"},
        {"state": "present", "dose": 1},
        {"state": "present", "duration_hours": float("nan")},
        {"state": "absent", "dose": 1, "unit": "ng/mL"},
        {"duration_hours": 1},
    ],
)
def test_exposure_validation(values):
    with pytest.raises(ValueError, match="exposure"):
        Exposure(**values)


def test_concurrent_cytokines_overrides_and_unknown_round_trip():
    original = context(True)
    changed = replace(
        original, cell_type_override="cell_a", cytokine_overrides={"signal_a": Exposure()}
    )
    resolved = resolve_cell_context(
        changed,
        config().cell_types,
        config().cytokines,
        config().cytokine_dose_units,
        config().cytokine_duration_features,
    )
    assert resolved.cell_type.value == "cell_a"
    assert resolved.cell_type.supplied == "cell_b" and resolved.cell_type.source == "override"
    assert resolved.cytokines["signal_a"].state == "unknown"
    assert resolved.cytokines["signal_b"].state == "present"
    assert resolved.supplied_cytokines["signal_a"].dose == 2
    assert resolved.cytokine_sources["signal_a"] == "override"
    record = example(context=changed)
    assert Observation.from_dict(record.to_dict()) == record
    new_evidence = replace(
        record,
        context=replace(
            changed,
            cytokines={
                name: replace(value, evidence={"export": "different"})
                for name, value in changed.cytokines.items()
            },
        ),
    )
    assert semantic_fingerprint(new_evidence) == semantic_fingerprint(record)
    with pytest.raises(ValueError, match="unit mismatch"):
        resolve_cell_context(
            original, config().cell_types, config().cytokines, {"signal_a": "different"}, ()
        )


def test_reference_precedence_provenance_and_explicit_unknown():
    general = MachineryProfile(
        "general", "species_a", {"a": Component("reference")}, "fixture-reference", "v1"
    )
    specific = MachineryProfile(
        "specific",
        "species_a",
        {"a": Component("reduced"), "b": Component("intact")},
        "fixture-reference",
        "v2",
        "cell_a",
    )
    species = resolve_field(("species_a",), supplied="species_a")
    defaults = resolve_components(
        Context(),
        species,
        ("a", "b"),
        {},
        cell_type="cell_a",
        reference_profiles=(general, specific),
    )
    assert defaults["a"].state == "reduced"
    assert defaults["a"].evidence["profile_id"] == "specific"
    assert defaults["a"].evidence["assumption"] is True
    explicit = Context(components={"a": Component()}, component_overrides={"b": Component("lost")})
    effective = resolve_components(
        explicit,
        species,
        ("a", "b"),
        {},
        cell_type="cell_a",
        reference_profiles=(general, specific),
    )
    assert effective == {"a": Component(), "b": Component("lost")}
    other = resolve_components(
        Context(),
        species,
        ("a", "b"),
        {},
        cell_type="cell_b",
        reference_profiles=(general, specific),
    )
    assert other["a"].state == "reference" and other["b"].state == "unknown"
    settings = config(
        species=("species_a",), components=("a", "b"), machinery_references=(general, specific)
    )
    assert ModularConfig.from_dict(asdict(settings)) == settings
    with pytest.raises(ValueError, match="ambiguous"):
        replace(settings, machinery_references=(general, general))


def test_cellular_sensitivity_and_noncellular_invariance():
    torch.manual_seed(31)
    network = ModularPresto(
        config(
            endpoints=(
                "ic50",
                "tcr_binding",
                "tcr_response",
                "elution_observation",
                "bulk_observation",
            )
        )
    ).eval()
    for endpoint in (
        "ic50",
        "tcr_binding",
        "tcr_response",
        "elution_observation",
        "bulk_observation",
    ):
        original = example(
            endpoint,
            context=context(),
            tcr_a="ACDEFGHIKLM",
            tcr_b="NPQRSTVWYAC",
            value=10 if endpoint == "ic50" else 1,
            unit="nM" if endpoint == "ic50" else "binary",
        )
        before, after = network([original, replace(original, context=context(True))])
        if endpoint in ("ic50", "tcr_binding", "bulk_observation"):
            torch.testing.assert_close(before.value, after.value)
        else:
            assert not torch.allclose(before.value, after.value)
        if "ms_response" in before.components:
            torch.testing.assert_close(
                before.components["ms_response"], after.components["ms_response"]
            )
    # Unknown vocabulary cannot silently become another cell type on cellular paths.
    with pytest.raises(ValueError, match="cell type outside"):
        network([example(context=replace(context(), cell_type="new_cell"))])
    # Irrelevant annotations do not prevent a purified-affinity request.
    network(
        [example("ic50", value=10, unit="nM", context=replace(context(), cell_type="new_cell"))]
    )


def test_new_columns_train_and_checkpoint_rejects_unseen_states(tmp_path):
    torch.manual_seed(37)
    network = ModularPresto(config())
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001)
    rows = [
        example(context=context(), value=0),
        example(context=context(True), observation_id="other"),
    ]
    train_step(network, optimizer, rows, {})
    labels = network.feature_layout["context_fusion.0.weight"][1]
    learned = {labels[index] for index in network.trained_features["context_fusion.0.weight"]}
    assert {
        "cell_type:cell_a",
        "cell_type:cell_b",
        "cytokine:signal_a:present",
        "cytokine:signal_a:absent",
        "cytokine:signal_a:log1p_dose",
        "cytokine:signal_a:log1p_duration_hours",
    } <= learned
    expected = network.predict(rows)
    path = tmp_path / "model.pt"
    save_checkpoint(path, network, {})
    restored, scales, _ = load_checkpoint(path)
    assert not scales and restored.config == network.config
    torch.testing.assert_close(restored.predict(rows)[0].value, expected[0].value)
    with pytest.raises(ValueError, match="untrained input features.*unknown"):
        restored.predict(
            [
                replace(
                    rows[0], context=replace(context(), cytokine_overrides={"signal_a": Exposure()})
                )
            ]
        )
    changed_units = ModularPresto(replace(config(), cytokine_dose_units={"signal_a": "other"}))
    with pytest.raises(ValueError, match="cytokine_dose_units"):
        initialize_from_checkpoint(changed_units, path)


def test_profile_checkpoint_and_unallocated_context_are_explicit(tmp_path):
    profile = MachineryProfile(
        "p", "species_a", {"a": Component("reference")}, "fixture", "v1", "cell_a"
    )
    settings = config(species=("species_a",), components=("a",), machinery_references=(profile,))
    network = ModularPresto(settings).eval()
    record = example(context=replace(context(), species={"apc": "species_a"}))
    evidence = network([record])[0].diagnostics["candidates"][0]
    assert evidence["components"]["a"]["evidence"]["profile_id"] == "p"
    path = tmp_path / "profile.pt"
    save_checkpoint(path, network, {})
    restored, _, _ = load_checkpoint(path)
    assert restored.config.machinery_references == (profile,)
    ignored = resolve_cell_context(context(True), (), (), {}, ())
    assert set(ignored.ignored_fields) == {"cell_type", "cytokine:signal_a", "cytokine:signal_b"}


def test_cell_context_preflight_requires_real_contrasts(tmp_path):
    sequences = ("ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH", "IKLMNPQRS")
    rows = [
        example(
            peptide=peptide,
            observation_id=f"r{i}",
            group_id=f"g{i}",
            split=("train", "train", "validation", "test")[i],
            context=context(bool(i % 2)),
            provenance={"original_measurement_id": f"m{i}"},
        )
        for i, peptide in enumerate(sequences)
    ]

    def index(records, name):
        receipt = write_observations(tmp_path / f"{name}.jsonl", records)
        manifest = {
            "schema": MANIFEST_VERSION,
            "model": asdict(config()),
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

    report = preflight(index(rows, "valid"))[3]
    assert report["cell_types"] == ["cell_a", "cell_b"]
    assert report["cytokine_states"]["signal_a"] == ["absent", "present"]
    missing = [
        replace(record, context=replace(record.context, cell_type="cell_a"))
        if record.split == "train"
        else record
        for record in rows
    ]
    with pytest.raises(ValueError, match="cell types have no observed"):
        preflight(index(missing, "missing-cell"))
    missing = [
        replace(
            record, context=replace(record.context, cytokine_overrides={"signal_b": Exposure()})
        )
        if record.split == "train"
        else record
        for record in rows
    ]
    with pytest.raises(ValueError, match="present/absent training contrast"):
        preflight(index(missing, "unknown-is-not-control"))
    missing = [
        replace(
            record,
            context=replace(
                record.context,
                cytokine_overrides={
                    "signal_a": replace(record.context.cytokines["signal_a"], duration_hours=24)
                },
            ),
        )
        if record.split == "train"
        else record
        for record in rows
    ]
    with pytest.raises(ValueError, match="numeric feature lacks varied"):
        preflight(index(missing, "constant-duration"))
