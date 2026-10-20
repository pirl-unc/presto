"""Prospective label-blind reassignment of the exact public preflight components."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import resource
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datacache import fetch_file, materialize, validate_file
from hitlist import audit_splits
from presto import __version__
from presto.data.frozen_partitions import _assign_components

FAMILY = Path(__file__).resolve().parents[1]
POLICIES = ("absolute_deficit_v1", "relative_deficit_v2")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pin(path):
    path = Path(path)
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": value, "size_bytes": path.stat().st_size}


def selected_rows(path, selected, columns=None):
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns):
        frame = batch.to_pandas()
        keep = frame.evidence_row_id.isin(selected)
        if keep.any():
            parts.append(frame.loc[keep].copy())
    result = pd.concat(parts, ignore_index=True)
    assert not result.evidence_row_id.duplicated().any()
    assert set(result.evidence_row_id) == selected
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--local-inputs")
    args = parser.parse_args()
    cache = Path(args.cache_dir)
    repository = FAMILY.parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)
    if dirty:
        raise ValueError("Execute only a clean registered source snapshot")
    declared = json.loads((FAMILY / "reproduce/inputs.json").read_text())
    local = json.loads(Path(args.local_inputs).read_text()) if args.local_inputs else {}
    sources = {}
    for name, expected in declared.items():
        path = local.get(name) or fetch_file(
            expected["url"],
            filename=name,
            subdir=expected["sha256"],
            cache_root=cache / "downloads",
            expected_sha256=expected["sha256"],
            expected_size=expected["size"],
            raw=True,
            show_progress=False,
        )
        validate_file(path, expected["sha256"])
        assert Path(path).stat().st_size == expected["size"]
        sources[name] = {"path": str(path), "sha256": expected["sha256"], "size": expected["size"]}
    sources["recipe.py"] = {"path": __file__, "sha256": pin(__file__)["sha256"]}
    allocator = Path(_assign_components.__code__.co_filename)
    assert pin(allocator)["sha256"] == pin(repository / "data/frozen_partitions.py")["sha256"]
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    options = {
        "commit": commit,
        "presto_version": __version__,
        "allocator_sha256": pin(allocator)["sha256"],
        "dirty": False,
        "config": config,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["hitlist", "datacache", "pandas", "numpy", "pyarrow"]
        },
        "python": sys.version,
    }

    def build(inputs, outputs):
        started = time.monotonic()
        started_utc = datetime.now(timezone.utc).isoformat()
        with tarfile.open(inputs["preflight.tar.gz"]) as archive:
            inventory = json.loads(Path(inputs["preflight-inventory.json"]).read_text())
            data = {}
            for name in [
                "frozen_assignments/assignments.parquet",
                "frozen_membership/membership.parquet",
            ]:
                data[name] = archive.extractfile(name).read()
                assert hashlib.sha256(data[name]).hexdigest() == inventory[name]["sha256"]
            original = pd.read_parquet(io.BytesIO(data["frozen_assignments/assignments.parquet"]))
            membership = pd.read_parquet(io.BytesIO(data["frozen_membership/membership.parquet"]))
        selected = set(original.evidence_row_id)
        assert len(original) == len(selected) == config["n_selected_observations"]
        components = [
            sorted(group.evidence_row_id) for _, group in original.groupby("component_id")
        ]
        identities = selected_rows(inputs["identities.parquet"], selected)
        context = selected_rows(
            inputs["training.parquet"],
            selected,
            [
                "evidence_row_id",
                "mhc_class",
                "evidence_kind",
                "condition_id",
                "cell_type",
                "cell_line_name",
                "assay_method",
            ],
        )
        records = membership.merge(context, on="evidence_row_id", validate="one_to_one")
        registry = json.loads(Path(inputs["lineage.json"]).read_text())
        summaries = {}
        with tempfile.TemporaryDirectory(prefix="component-allocation-") as tmp:
            work = Path(tmp)
            for policy in POLICIES:
                assigned = _assign_components(
                    components,
                    seed=config["seed"],
                    val_frac=config["val_frac"],
                    test_frac=config["test_frac"],
                    allocation=policy,
                )
                result = original.copy()
                result["source_partition"] = result.evidence_row_id.map(
                    lambda key: assigned[key][0]
                )
                assert all(
                    assigned[row.evidence_row_id][1] == row.component_id
                    for row in result.itertuples()
                )
                if policy == "absolute_deficit_v1":
                    assert result.equals(original), (
                        "Historical assignments did not reproduce exactly"
                    )
                assert result.groupby("component_id").source_partition.nunique().eq(1).all()
                variant = work / policy
                variant.mkdir()
                result.to_parquet(variant / "assignments.parquet", index=False)
                result.to_csv(variant / "assignments.tsv", sep="\t", index=False)
                labels = records.drop(columns="source_partition").merge(
                    result[["evidence_row_id", "source_partition", "component_id"]],
                    on="evidence_row_id",
                    validate="one_to_one",
                )
                labels.to_csv(variant / "observation_contexts.tsv", sep="\t", index=False)
                for report, axes in {
                    "assay_support": ["source_partition", "assay_group"],
                    "endpoint_support": [
                        "source_partition",
                        "source_endpoint",
                        "source_units",
                        "source_inequality",
                    ],
                    "class_support": ["source_partition", "mhc_class", "evidence_kind"],
                    "component_sizes": ["source_partition", "component_id"],
                }.items():
                    labels.groupby(axes, dropna=False).size().rename(
                        "n_observations"
                    ).reset_index().to_csv(variant / (report + ".tsv"), sep="\t", index=False)
                partitions = {
                    name: identities[
                        identities.evidence_row_id.isin(
                            result.loc[result.source_partition.eq(name), "evidence_row_id"]
                        )
                    ]
                    for name in ["train", "val", "test"]
                }
                audit = audit_splits(partitions, policy="peptide_disjoint", registry=registry)
                write_json(variant / "lineage_audit.json", audit)
                assert audit["verdict"] == "pass"
                summaries[policy] = {
                    "partition_counts": result.source_partition.value_counts().to_dict(),
                    "n_components": len(components),
                    "n_largest_component_observations": max(map(len, components)),
                    "endpoint_counts": labels.groupby(
                        ["source_endpoint", "source_partition"], dropna=False
                    )
                    .size()
                    .rename("n_observations")
                    .reset_index()
                    .to_dict("records"),
                    "n_moved_observations": int(
                        result.source_partition.ne(original.source_partition).sum()
                    ),
                    "audit_verdict": audit["verdict"],
                }
            write_json(
                work / "policy.json",
                {
                    **options,
                    "source_pins": declared,
                    "scope": "proposed root assignments only; no model constructed or predictive outcome used",
                    "limitations": "Original Hitlist1.76 source/registry; unknown biological identities remain unknown. No independent-study or training-readiness claim.",
                },
            )
            inventory = {
                p.relative_to(work).as_posix(): pin(p)
                for p in sorted(work.rglob("*"))
                if p.is_file()
            }
            write_json(outputs["inventory.json"], inventory)
            with tarfile.open(outputs["run.tar.gz"], "w:gz") as archive:
                for name in inventory:
                    archive.add(work / name, arcname=name, recursive=False)
            write_json(
                outputs["summary.json"],
                {
                    **options,
                    "status": "success",
                    "started_utc": started_utc,
                    "runtime_seconds": time.monotonic() - started,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                    "n_selected_observations": len(selected),
                    "policies": summaries,
                    "no_predictive_outcomes_used": True,
                    "original_v1_membership_reproduced_exactly": True,
                },
            )

    paths = materialize(
        cache / "audit",
        sources,
        transform={"version": "component-allocation-audit-v1", "options": options},
        outputs={"run.tar.gz": {}, "inventory.json": {}, "summary.json": {}},
        builder=build,
    )
    receipt = {
        "files": {n: {"path": str(p), **pin(p)} for n, p in paths.items()},
        "summary": json.loads(Path(paths["summary.json"]).read_text()),
    }
    write_json(FAMILY / "results/audit.json", receipt)
    print(json.dumps(receipt["summary"], indent=2))


if __name__ == "__main__":
    main()
