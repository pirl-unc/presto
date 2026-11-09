"""Inventory stability labels and TRAIN overlap without admitting new measurements."""

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import gzip
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import time

import mhcgnomes

F = Path(__file__).resolve().parents[1]
A = Path("/Users/iskander/code/presto/artifacts") / F.name


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=None)
def allele(name):
    return mhcgnomes.parse(name).to_string()


def main():
    started = time.monotonic()
    source = A / "sources/netmhcstabpan-stability/Stability.txt"
    cohort = A / "curated/train.jsonl.gz"
    invocation = dict(
        argv=[sys.executable, str(Path(__file__).resolve())],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=subprocess.check_output(["git", "status", "--short"], text=True),
        source_sha256=sha(Path(__file__)),
        inputs={str(p): sha(p) for p in (source, cohort)},
        environment={"mhcgnomes": importlib.metadata.version("mhcgnomes")},
        requested_gpu=None,
        started_unix=time.time(),
    )
    receipt = F / "reproduce/stability-review-launch.json"
    assert not receipt.exists()
    receipt.write_text(json.dumps(invocation, indent=2) + "\n")
    shutil.copyfile(__file__, F / "reproduce/source/review_stability.py")
    rows = []
    with source.open() as stream:
        assert next(stream).split() == ["HLA", "Pep", "Thalf"]
        for line_number, line in enumerate(stream, start=2):
            mhc, peptide, value = line.split()
            hours = float(value)
            assert math.isfinite(hours) and hours >= 0
            rows.append((allele(mhc), peptide, hours, line_number))
    keys = {(mhc, pep) for mhc, pep, _, _ in rows}
    existing = defaultdict(list)
    with gzip.open(cohort, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["split"] == "train"
            names = {
                allele(name)
                for m in row["mhc"]
                for name in m["provenance"].get("catalog_alleles", [])
            }
            for name in names:
                key = name, row["peptide"]
                if key in keys:
                    existing[key].append(row)
    counts, matches, endpoints = Counter(), [], Counter()
    for mhc, peptide, hours, line_number in rows:
        key = mhc, peptide
        related = existing[key]
        kind = "positive" if hours > 0 else "zero"
        counts[kind + "_rows"] += 1
        counts[kind + "_rows_with_any_physical_train_pmhc"] += bool(related)
        half_lives = [r for r in related if r["endpoint"] == "half_life"]
        counts[kind + "_rows_with_half_life_train_pmhc"] += bool(half_lives)
        equal = [
            r
            for r in half_lives
            if r["qualifier"] == "="
            and math.isclose(r["value"], hours * 60, rel_tol=1e-9, abs_tol=1e-9)
        ]
        rounded = [
            r
            for r in half_lives
            if r["qualifier"] == "=" and abs(r["value"] / 60 - hours) <= 0.005 + 1e-9
        ]
        counts[kind + "_rows_matching_exact_half_life_value"] += bool(equal)
        counts[kind + "_rows_matching_two_decimal_hour_rounding"] += bool(rounded)
        endpoints.update(set(r["endpoint"] for r in related))
        if related:
            matches.append(
                dict(
                    source_line=line_number,
                    pmhc_sha256=hashlib.sha256((mhc + ":" + peptide).encode()).hexdigest(),
                    train_observation_ids=[r["observation_id"] for r in related],
                    exact_half_life_ids=[r["observation_id"] for r in equal],
                    rounded_half_life_ids=[r["observation_id"] for r in rounded],
                )
            )
    values = [r[2] for r in rows if r[2] > 0]
    result = dict(
        status="reviewed_not_admitted",
        rows=len(rows),
        unique_pmhcs=len(keys),
        alleles=len({r[0] for r in rows}),
        peptide_lengths=dict(Counter(len(r[1]) for r in rows)),
        counts=dict(counts),
        overlapping_endpoints=dict(endpoints),
        positive_half_life_hours_min=min(values),
        positive_half_life_hours_max=max(values),
        top_values_hours=[
            dict(value=k, rows=n) for k, n in Counter(r[2] for r in rows).most_common(12)
        ],
        paper="https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/",
        source_url="https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt",
        decision=(
            "Half-life measurements are distinct from affinity. Do not map zero hours to "
            "an arbitrary log floor. Archive lacks row-level inequality, modification and "
            "individual-study provenance; verify those and all heldout overlaps before "
            "admission. TRAIN pMHC non-overlap does not establish independent measurements. "
            "No active fit, label, loss or split is changed."
        ),
        inputs=invocation["inputs"],
        overlap_lineage=matches,
        seconds=time.monotonic() - started,
    )
    (F / "results/stability-review.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "overlap_lineage"}), flush=True)


if __name__ == "__main__":
    main()
