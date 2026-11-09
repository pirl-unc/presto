"""Select new intended panels using immutable metadata, without response quantities."""

import csv
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from hitlist.evidence_store import EvidenceStore
from hitlist.provenance import file_digest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
HITLIST = Path("/private/tmp/hitlist-proteometools-ms1")
SOURCE = REPO / "data/measured_ms1_source_admission/2026-10-24_1635_v1/source"
RAW = REPO / "data/measured_ms1_grouped_observer/2026-10-24_1702_v1"
LIBRARIES = ("HLA Class I", "HLA Class II")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def main():
    started = time.perf_counter()
    assert not (ROOT / "results/selection.json").exists()
    metadata = {
        "agent_model": "Codex / GPT-6",
        "phase": "metadata_only_selection",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "argv": sys.argv,
        "presto_commit": git(REPO, "rev-parse", "HEAD"),
        "presto_tracked_status": git(REPO, "status", "--short", "--untracked-files=no"),
        "hitlist_commit": git(HITLIST, "rev-parse", "HEAD"),
        "hitlist_tracked_status": git(HITLIST, "status", "--short", "--untracked-files=no"),
        "environment": {
            k: os.environ[k]
            for k in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if k in os.environ
        },
        "source_manifest": file_digest(SOURCE / "manifest.json"),
    }
    write(ROOT / "reproduce/launch.json", metadata)
    shutil.copy2(__file__, ROOT / "reproduce/source/select.py")
    store = EvidenceStore.open(SOURCE)
    mapping = store.query(
        "proteometools_planned_products",
        select=["native.pool_name", "peptide_sequence", "native.library_origin"],
        limit=None,
    ).table.to_pylist()
    pools = defaultdict(set)
    library = {}
    for row in mapping:
        pool = row["native.pool_name"]
        pools[pool].add(row["peptide_sequence"])
        library[pool] = row["native.library_origin"]
    with (SOURCE / "proteometools_ms1/source_assets/PXD021013-README.txt").open() as stream:
        original = list(csv.DictReader(stream, delimiter="\t"))
    index = {row["ID"]: row for row in original}
    searches = {row["NAME"]: row for row in original if row["TYPE"] == "SEARCH"}
    pilot_sequences = pools["TUM_HLA_1"] | pools["TUM_HLA2_1"]
    candidates, skipped = defaultdict(list), []
    for pool in sorted(pools):
        if library[pool] not in LIBRARIES:
            continue
        filename = pool + "_01_01_DDA-1h-R1-unspecific.zip"
        search = searches.get(filename)
        reasons = []
        if pools[pool] & pilot_sequences:
            reasons.append("intended_overlap_with_development_pilot")
        if (
            search is None
            or search["MAPPINGS"] not in index
            or index[search["MAPPINGS"]]["TYPE"] != "RAW"
        ):
            reasons.append("exact_single_run_SEARCH_RAW_mapping_unavailable")
        if reasons:
            skipped.append({"pool": pool, "library": library[pool], "reasons": reasons})
            continue
        candidates[library[pool]].append(
            {
                "pool": pool,
                "library": library[pool],
                "source_name": filename,
                "search_index_id": search["ID"],
                "raw_index_id": search["MAPPINGS"],
                "raw_file": index[search["MAPPINGS"]]["NAME"],
                "selection_hash": hashlib.sha256(
                    ("presto_ms1_observer_v1:" + pool).encode()
                ).hexdigest(),
            }
        )
    for items in candidates.values():
        items.sort(key=lambda row: row["selection_hash"])
    selected, counts, offsets, used = [], defaultdict(int), defaultdict(int), set()
    while any(counts[lib] < 12 for lib in LIBRARIES):
        changed = False
        for lib in LIBRARIES:
            if counts[lib] == 12:
                continue
            while offsets[lib] < len(candidates[lib]):
                row = candidates[lib][offsets[lib]]
                offsets[lib] += 1
                pool = row["pool"]
                if pools[pool] & used:
                    skipped.append(
                        {
                            "pool": pool,
                            "library": lib,
                            "reasons": ["intended_overlap_with_prior_selected_pool"],
                        }
                    )
                    continue
                rank = counts[lib]
                selected.append(
                    {
                        **row,
                        "library_selection_rank": rank,
                        "split": "test" if rank < 2 else "validation" if rank < 4 else "train",
                        "n_intended_sequences": len(pools[pool]),
                        "intended_sequences": sorted(pools[pool]),
                    }
                )
                counts[lib] += 1
                used.update(pools[pool])
                changed = True
                break
        if not changed:
            break
    report = {
        "status": "selected" if len(selected) == 24 else "insufficient_metadata_eligible_pools",
        "selected": selected,
        "skipped": skipped,
        "n_candidates_by_library": {lib: len(candidates[lib]) for lib in LIBRARIES},
        "n_selected_pools": len(selected),
        "n_distinct_intended_sequences": len(used),
        "source_manifest": metadata["source_manifest"],
        "raw_destination": str(RAW),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    write(ROOT / "results/selection.json", report)
    assert len(selected) == 24, report["status"]
    assert sum(row["n_intended_sequences"] for row in selected) == len(used)
    assert report["peak_rss_gib"] <= 3
    with (ROOT / "results/pools.csv").open("w") as stream:
        fields = [
            "pool",
            "library",
            "split",
            "library_selection_rank",
            "n_intended_sequences",
            "source_name",
            "raw_file",
            "selection_hash",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in selected)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("selected", "skipped")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
