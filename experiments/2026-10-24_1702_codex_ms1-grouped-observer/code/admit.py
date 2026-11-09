"""Canonical cohort admission and explicit separation of test covariates/outcomes."""

import json
import resource
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from hitlist.evidence_recipe import materialize_recipe, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.proteometools_ms1 import write_proteometools_ms1_package
from hitlist.proteometools_response import RECIPE_ID
from hitlist.provenance import file_digest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = REPO / "data/measured_ms1_grouped_observer/2026-10-24_1702_v1"
DATA = REPO / "data/measured_ms1_grouped_observer/2026-10-24_1702_canonical"


def dump(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def main():
    started = time.perf_counter()
    assert not DATA.exists()
    DATA.mkdir(parents=True)
    shutil.copy2(__file__, ROOT / "reproduce/source/admit.py")
    selection = json.loads((ROOT / "results/selection.json").read_text())
    pools = {row["pool"]: row for row in selection["selected"]}
    archives = json.loads((ROOT / "reproduce/archive-receipts.json").read_text())
    assert len(archives) == len(pools) == 24
    assert {r["source_name"] for r in archives} == {r["source_name"] for r in pools.values()}
    write_proteometools_ms1_package(RAW, DATA / "source", archives=archives)
    store = EvidenceStore.open(DATA / "source")
    materialize_recipe(
        store,
        {
            "schema_version": 1,
            "name": "prospective_24_pool_MS1_observer",
            "registered_recipe": RECIPE_ID,
        },
        DATA / "recipe",
    )
    verify_recipe_package(DATA / "recipe", source_package=DATA / "source")
    runs = pq.read_table(DATA / "source/proteometools_ms1/typed/runs.parquet").to_pylist()
    contexts = []
    for run in runs:
        intended = pools[run["pool_name"]]
        assert (
            run["source_name"] == intended["source_name"]
            and run["raw_file"] == intended["raw_file"]
        )
        reasons = []
        if run["fixed_modifications_raw"] != "Carbamidomethyl (C)":
            reasons.append("fixed_modification_context_mismatch")
        if run["software_version_raw"] != "1.5.3.30":
            reasons.append("software_context_mismatch")
        if run["search_enzyme_mode_raw"] != "Unspecific":
            reasons.append("search_context_mismatch")
        contexts.append({**run, "split": intended["split"], "context_exclusions": reasons})
    (ROOT / "results/run-contexts.json").write_text(json.dumps(contexts, indent=2) + "\n")
    assert len(contexts) == 24 and all(not r["context_exclusions"] for r in contexts), (
        "Prospective context gate failed"
    )
    examples = json.loads((DATA / "recipe/examples.json").read_text())
    counts = Counter(row["pool_name"] for row in examples)
    coverage = {
        "n_examples": len(examples),
        "n_pools": len(counts),
        "examples_per_pool": dict(counts),
        "examples_per_split": dict(Counter(pools[row["pool_name"]]["split"] for row in examples)),
    }
    (ROOT / "results/coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    assert len(counts) == 24 and min(counts.values()) >= 50, (
        "Prospective minimum per-pool coverage gate failed"
    )
    groups = json.loads((DATA / "recipe/dependence_groups.json").read_text())
    assert len(groups) == 24 and all(len(group["pools"]) == 1 for group in groups)
    splits = {key: [] for key in ("train", "validation", "test")}
    for row in sorted(examples, key=lambda r: (r["pool_name"], r["sequence"], r["measurement_id"])):
        pool = pools[row["pool_name"]]
        assert row["sequence"] in pool["intended_sequences"]
        splits[pool["split"]].append(
            {
                "measurement_id": row["measurement_id"],
                "source_record_id": row["source_record_id"],
                "sequence": row["sequence"],
                "run_id": row["run_id"],
                "pool": row["pool_name"],
                "library": pool["library"],
                "dependence_component": row["dependence_component"],
                "target": row["target_centered_log10_signal"],
            }
        )
    # Only the admission process reads recipe quantities. Fitting opens explicitly
    # separated files and never loads this recipe or the test outcome file.
    for split in ("train", "validation"):
        dump(DATA / (split + ".jsonl"), splits[split])
    dump(
        DATA / "test-covariates.jsonl",
        [{k: v for k, v in row.items() if k != "target"} for row in splits["test"]],
    )
    dump(
        DATA / "test-targets.jsonl",
        [
            {"measurement_id": row["measurement_id"], "target": row["target"]}
            for row in splits["test"]
        ],
    )
    sequence_sets = {split: {r["sequence"] for r in rows} for split, rows in splits.items()}
    assert (
        not sequence_sets["train"] & sequence_sets["validation"]
        and not sequence_sets["train"] & sequence_sets["test"]
        and not sequence_sets["validation"] & sequence_sets["test"]
    )
    for name in ("summary", "contract", "dependence_groups"):
        shutil.copy2(
            DATA / "recipe" / (name + ".json"), ROOT / "results" / ("recipe-" + name + ".json")
        )
    metadata = {
        "status": "admitted",
        "argv": sys.argv,
        "python": sys.executable,
        "data_directory": str(DATA),
        "coverage": coverage,
        "source_manifest": file_digest(DATA / "source/manifest.json"),
        "recipe_manifest": file_digest(DATA / "recipe/manifest.json"),
        "split_files": {p.name: file_digest(p) for p in sorted(DATA.glob("*.jsonl"))},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "raw_and_package_logical_bytes": sum(
            p.stat().st_size for folder in (RAW, DATA) for p in folder.rglob("*") if p.is_file()
        ),
        "free_disk_bytes": shutil.disk_usage(REPO).free,
    }
    (ROOT / "results/admission.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert metadata["peak_rss_gib"] <= 3
    assert metadata["raw_and_package_logical_bytes"] <= 2 * 1024**3
    assert metadata["free_disk_bytes"] >= 2 * 1024**3
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
