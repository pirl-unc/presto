"""Genuine chain annotations without fictitious pair partners."""

from dataclasses import replace
import json

import pytest
import torch

from presto.data.modular import AA, MHCInput, Observation
from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, write_observations
from presto.models.modular import ModularConfig, ModularPresto
from presto.models.modular_dependencies import freeze_dependencies
from presto.training.modular import initialize_from_checkpoint, load_checkpoint, train_step
from presto.training.modular_run import fit_index


def single(slot, sequence=AA, *, label="human", **kwargs):
    inputs = (
        {"mhc": (MHCInput(sequence if slot == "mhc1" else "", sequence if slot == "mhc2" else ""),)}
        if slot.startswith("mhc")
        else {slot: sequence}
    )
    return Observation(
        "single", "software", "identity", identity_labels={slot: label}, **inputs, **kwargs
    )


@pytest.mark.parametrize("slot", ["mhc1", "mhc2", "tcr_a", "tcr_b"])
def test_single_chain_encodes_only_present_input_and_predicts_only_species(slot):
    row = single(slot)
    row.validate()
    assert Observation.from_dict(row.to_dict()) == row
    task = "mhc_species" if slot.startswith("mhc") else "tcr_species"
    paired = (
        ("mhc_class", "mhc_pairing") if slot.startswith("mhc") else ("tcr_lineage", "tcr_pairing")
    )
    model = ModularPresto(
        ModularConfig(
            endpoints=(), identity_tasks=(task, *paired), species=("human", "mouse"), dropout=0
        )
    )
    pooled, states = model._encode([row])
    assert set(pooled) == set(states) == {AA}
    output = model([row])[0]
    assert set(output.identity) == {slot}
    assert output.value is None and output.diagnostics["class"] is None
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    train_step(model, optimizer, [row], {})
    assert all(model.support[name]["status"] == "random/untrained" for name in paired)
    # Random pair-level readouts must not prevent serving a trained single-chain head.
    assert set(model.predict([row])[0].identity) == {slot}


@pytest.mark.parametrize(
    "slot,target,label",
    [
        ("mhc1", "mhc2", "human"),
        ("mhc2", "mhc1", "human"),
        ("mhc1", "mhc_class", "I"),
        ("mhc2", "mhc_pairing", "matched"),
        ("tcr_a", "tcr_b", "human"),
        ("tcr_b", "tcr_a", "human"),
        ("tcr_a", "tcr_lineage", "CD4"),
        ("tcr_b", "tcr_pairing", "matched"),
    ],
)
def test_missing_target_inputs_fail_before_training(slot, target, label):
    row = replace(single(slot), identity_labels={target: label})
    with pytest.raises(ValueError, match="identity input missing"):
        row.validate()


def test_assays_and_direct_pair_validation_remain_strict():
    with pytest.raises(ValueError, match="required canonical"):
        MHCInput(AA, "").validate()
    with pytest.raises(ValueError, match="required canonical"):
        replace(
            single("mhc1"),
            endpoint="ic50",
            peptide="ACDEFGHIK",
            value=10,
            unit="nM",
            identity_labels={},
        ).validate()
    with pytest.raises(ValueError, match="paired TCR"):
        replace(
            single("tcr_a"),
            endpoint="tcr_binding",
            peptide="ACDEFGHIK",
            mhc=(MHCInput(AA, AA),),
            value=1,
            unit="binary",
            identity_labels={},
        ).validate()


def test_mhc_bags_still_require_trained_internal_annotation_heads():
    model = ModularPresto(
        ModularConfig(
            endpoints=("elution_observation",),
            mhc_classes=("I",),
            identity_tasks=("mhc_class", "mhc_species"),
            species=("human", "mouse"),
            dropout=0,
        )
    )
    row = Observation(
        "bag",
        "software",
        "elution_observation",
        AA[:9],
        (MHCInput(AA, AA, "I"), MHCInput(AA[::-1], AA, "I")),
        value=1,
        unit="binary",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    train_step(model, optimizer, [row], {})
    with pytest.raises(ValueError, match="mhc_class: no trained output support"):
        model.predict([row])


def manifest(tmp_path, rows):
    receipt = write_observations(tmp_path / "rows.jsonl", rows)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_VERSION,
                "model": {
                    "endpoints": [],
                    "identity_tasks": ["mhc_species", "tcr_species"],
                    "species": ["human", "mouse"],
                    "dropout": 0,
                },
                "training": {"max_epochs": 2, "batch_size": 4},
                "sources": [
                    {
                        "path": "rows.jsonl",
                        "sha256": receipt["sha256"],
                        "source": "software",
                        "population": "artificial software verification",
                        "label_policy": "artificial annotations; no biological claim",
                        "split_policy": "distinct rotated sequences",
                    }
                ],
            }
        )
    )
    return path


def test_partial_and_paired_annotation_training_reload_and_freezing(tmp_path):
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for slot_index, slot in enumerate(("mhc1", "mhc2", "tcr_a", "tcr_b")):
            offset = split_index * 4 + slot_index
            row = single(
                slot,
                AA[offset:] + AA[:offset],
                label=("human", "mouse")[slot_index % 2],
                split=split,
                group_id=f"g{offset}",
                provenance={"original_measurement_id": str(offset)},
            )
            rows.append(replace(row, observation_id=f"row{offset}"))
    # Full and single views of the same chains may share one split.
    paired = Observation(
        "full",
        "software",
        "identity",
        mhc=(MHCInput(rows[0].mhc[0].groove1, rows[1].mhc[0].groove2),),
        tcr_a=rows[2].tcr_a,
        tcr_b=rows[3].tcr_b,
        identity_labels={
            slot: ("human", "mouse")[i % 2]
            for i, slot in enumerate(("mhc1", "mhc2", "tcr_a", "tcr_b"))
        },
        group_id="paired-train",
        provenance={"original_measurement_id": "paired"},
    )
    index = ObservationIndex(manifest(tmp_path, [*rows, paired]))
    summary = fit_index(index, tmp_path / "fit")
    assert summary["coverage_gate"]["status"] == "passed"
    restored, scales, _ = load_checkpoint(tmp_path / "fit/selected.pt")
    assert not scales
    assert set(restored.predict([paired])[0].identity) == set(paired.identity_labels)
    assert set(restored.predict([rows[5]])[0].identity) == {"mhc2"}
    continuation = ModularPresto(restored.config)
    initialize_from_checkpoint(continuation, tmp_path / "fit/selected.pt")
    freeze_dependencies(continuation, ("mhc_species",))
    before = continuation.predict([rows[0]])[0].identity["mhc1"].clone()
    optimizer = torch.optim.AdamW(
        [p for p in continuation.parameters() if p.requires_grad], lr=0.001
    )
    train_step(continuation, optimizer, rows[2:4], {})
    assert torch.equal(before, continuation.predict([rows[0]])[0].identity["mhc1"])


@pytest.mark.parametrize("family", ["mhc", "tcr"])
def test_single_chain_cannot_hide_cross_split_exposure_by_changing_partner(tmp_path, family):
    first = single(
        "mhc1" if family == "mhc" else "tcr_a",
        split="train",
        group_id="train",
        provenance={"original_measurement_id": "one"},
    )
    if family == "mhc":
        second = replace(
            first,
            mhc=(MHCInput(AA, "LMNPQRSTVW"),),
            identity_labels={"mhc1": "human", "mhc2": "mouse"},
        )
    else:
        second = replace(
            first, tcr_b="LMNPQRSTVW", identity_labels={"tcr_a": "human", "tcr_b": "mouse"}
        )
    second = replace(
        second,
        observation_id="second",
        group_id="validation",
        split="validation",
        provenance={"original_measurement_id": "two"},
    )
    with pytest.raises(ValueError, match="identity chain leaks"):
        ObservationIndex(manifest(tmp_path, [first, second]))
