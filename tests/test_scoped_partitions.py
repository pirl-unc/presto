"""Frozen scoped membership and the canonical consumer, using a synthetic source fixture."""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from test_scoped_bulk import scoped_bundle as _scoped_bundle
from presto.data.loaders import PrestoDataset
from presto.data.scoped_bulk import load_scoped_bulk_records
from presto.data.scoped_partitions import prepare_scoped_partitions
from presto.data.frozen_hitlist import _sha256

scoped_bundle = _scoped_bundle


def prepare(bundle, **kwargs):
    directory, digest, cache = bundle
    records, stats = load_scoped_bulk_records(directory, bundle_sha256=digest, cache_dir=cache)
    frozen = prepare_scoped_partitions(
        records,
        stats,
        seed=42,
        val_frac=0.25,
        test_frac=0.25,
        cache_dir=cache / "partitions",
        **kwargs,
    )
    return records, stats, frozen


def test_frozen_membership_is_reusable_and_honest_about_independence(scoped_bundle):
    records, stats, frozen = prepare(scoped_bundle)
    indices, receipt = frozen.finalize(PrestoDataset(bulk_ms_records=records))
    assert {name: len(rows) for name, rows in indices.items()} == {"train": 2, "val": 1, "test": 1}
    root = Path(receipt["directory"])
    frame = pd.read_csv(root / "membership.tsv", sep="\t", keep_default_na=False)
    assert frame.sample_id.nunique() == 4
    assert frame.loc[frame.detectability_label.eq(0), "evidence_row_id"].eq("").all()
    assert pd.read_parquet(root / "parent_graph.parquet").empty
    audit = json.loads((root / "leakage_audit.json").read_text())
    assert audit["peptide_disjoint"] is True
    assert "not established" in audit["experimental_independence"]
    assert "inconclusive" in audit["donor_independence"]
    assert all(pair["n_shared_peptides"] == 0 for pair in audit["pairwise"].values())
    assert all(pair["n_shared_search_scopes"] == 1 for pair in audit["pairwise"].values())
    assert prepare(scoped_bundle)[2].receipt == receipt


@pytest.mark.parametrize("change", ["drop", "duplicate", "partition", "target", "peptide", "flank"])
def test_conversion_cannot_change_frozen_membership(scoped_bundle, change):
    records, _, frozen = prepare(scoped_bundle)
    dataset = PrestoDataset(bulk_ms_records=records)
    if change == "drop":
        dataset.samples.pop()
    elif change == "duplicate":
        dataset.samples.append(copy.deepcopy(dataset.samples[0]))
    elif change == "partition":
        dataset.samples[0].source_partition = "elsewhere"
    elif change == "target":
        dataset.samples[0].ms_detectability_label = 1 - dataset.samples[0].ms_detectability_label
    elif change == "flank":
        dataset.samples[0].flank_c = "YYYY"
    else:
        dataset.samples[0].peptide = "YYYYYYYYK"
    with pytest.raises(ValueError, match="frozen"):
        frozen.finalize(dataset)


def test_external_membership_is_pinned_and_exact(scoped_bundle, tmp_path):
    records, _, first = prepare(scoped_bundle)
    path = Path(first.receipt["directory"]) / "assignments.parquet"
    _, _, restored = prepare(scoped_bundle, assignment_file=path, assignment_sha256=_sha256(path))
    assert restored.membership.equals(first.membership)
    invalid = pd.read_parquet(path)
    invalid.loc[0, "peptide"] = "YYYYYYYYK"
    bad = tmp_path / "changed.parquet"
    invalid.to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="peptide identities"):
        prepare(scoped_bundle, assignment_file=bad, assignment_sha256=_sha256(bad))
    with pytest.raises(ValueError, match="(?i)sha"):
        prepare(scoped_bundle, assignment_file=path, assignment_sha256="0" * 64)


@pytest.mark.parametrize(
    "change", ["duplicate", "scope", "fake_observation", "excision", "candidate", "flank"]
)
def test_invalid_records_cannot_acquire_a_partition_contract(scoped_bundle, change):
    records, stats, _ = prepare(scoped_bundle)
    if change == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif change == "scope":
        records[0].source_search_space_id = "other"
    elif change == "fake_observation":
        next(record for record in records if not record.observed).evidence_row_id = "invented"
    elif change == "candidate":
        records[0].candidate_id = "invented-candidate"
    elif change == "flank":
        records[0].flank_n = "YYYY"
    else:
        records[0].excision_label = 1.0
    with pytest.raises(ValueError, match="Scoped partitions"):
        prepare_scoped_partitions(records, stats, seed=42, val_frac=0.25, test_frac=0.25)


def test_canonical_scoped_preflight_never_reads_unrelated_sources(
    scoped_bundle, tmp_path, monkeypatch
):
    import presto.data.scoped_bulk as scoped
    import presto.data.scoped_partitions as partitions
    from presto.legacy.scripts import train_iedb

    directory, digest, cache = scoped_bundle
    load, freeze = scoped.load_scoped_bulk_records, partitions.prepare_scoped_partitions
    monkeypatch.setattr(
        scoped, "load_scoped_bulk_records", lambda *a, **k: load(*a, **k, cache_dir=cache)
    )
    monkeypatch.setattr(
        partitions,
        "prepare_scoped_partitions",
        lambda *a, **k: freeze(*a, **k, cache_dir=cache / "canonical"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Scoped preflight touched an unrelated source or constructed a model")

    monkeypatch.setattr(train_iedb, "load_records_from_merged_tsv", forbidden)
    monkeypatch.setattr(train_iedb, "resolve_mhc_inputs_from_index", forbidden)
    monkeypatch.setattr(train_iedb, "Presto", forbidden)
    (tmp_path / "merged_deduped.tsv").write_text("unrelated input\n")
    run_dir = tmp_path / "run"
    train_iedb.run(
        argparse.Namespace(
            data_source="scoped_bulk",
            scoped_bundle=str(directory),
            scoped_bundle_sha256=digest,
            data_dir=str(tmp_path),
            run_dir=str(run_dir),
            mhc_augmentation_samples=0,
            uniprot_negative_ratio=0,
            val_frac=0.25,
            test_frac=0.25,
            data_preflight_only=True,
            require_split_target=["ms_detectability"],
            device="cpu",
        )
    )
    funnel = json.loads((run_dir / "data_funnel.json").read_text())
    assert funnel["stages"]["loaded_post_cap"]["bulk_ms"] == 4
    assert funnel["frozen_membership"]["policy"] == "peptide_disjoint_within_search"
    assert (run_dir / "split_support.json").exists()
    configured = json.loads((run_dir / "config.json").read_text())
    effective = json.loads((run_dir / "regularization.json").read_text())
    assert configured["binding_orthogonality_weight"] == 0.01
    assert configured["balanced_batches"] is True
    assert effective["base"]["binding_orthogonality_weight"] == 0.01
    assert len(effective["epochs"]) == configured["epochs"]
    assert effective["epochs"][0]["parameters"]["consistency_cascade_weight"] == 0.0
    assert effective["epochs"][0]["parameters"]["binding_orthogonality_weight"] == 0.01


@pytest.mark.parametrize(
    "override",
    [
        {"split_mode": "random_rows"},
        {"frozen_partition_policy": "independent_experiments"},
        {"bulk_ms": True},
        {"mhc_augmentation_samples": 1},
        {"uniprot_negative_ratio": 0.1},
    ],
)
def test_canonical_scoped_guards_reject_mixed_or_overstated_inputs(override):
    from presto.legacy.scripts.train_iedb import run

    args = dict(data_source="scoped_bulk", mhc_augmentation_samples=0, uniprot_negative_ratio=0)
    args.update(override)
    with pytest.raises(ValueError, match="Scoped bulk"):
        run(argparse.Namespace(**args))


def test_archived_source_pins_and_new_manifest_only_interface(monkeypatch):
    from presto.cli.main import create_parser
    from presto.legacy.scripts import train_iedb

    flags = [
        "--data-source",
        "scoped_bulk",
        "--scoped-bundle",
        "/frozen",
        "--scoped-bundle-sha256",
        "a" * 64,
        "--max-scoped-bulk",
        "100",
        "--frozen-assignment-file",
        "/assignments.parquet",
        "--frozen-assignment-sha256",
        "b" * 64,
    ]
    with pytest.raises(SystemExit, match="2"):
        create_parser().parse_args(["train", "--manifest", "fixture.json", *flags])
    captured = []
    monkeypatch.setattr(train_iedb, "run", captured.append)
    train_iedb.main(flags)
    expected = dict(
        data_source="scoped_bulk",
        scoped_bundle="/frozen",
        scoped_bundle_sha256="a" * 64,
        max_scoped_bulk=100,
        frozen_assignment_file="/assignments.parquet",
        frozen_assignment_sha256="b" * 64,
        frozen_partition_policy="peptide_disjoint",
    )
    for name, value in expected.items():
        assert getattr(captured[0], name) == value
