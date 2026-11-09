"""Literal source policy and sequence-dependency inventory, without scoring."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
ADMISSION = ROOT / "experiments/2026-10-26_1825_codex_mhc2-ms-admission/results/admission-attempt01"
INPUT = Path("/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01")
CURATION = Path(
    "/private/tmp/presto-mhc2-20261026/hitlist-current/hitlist/data/pmid_overrides.yaml"
)
AFFINITY = (
    ROOT
    / "experiments/2026-10-26_1929_codex_mhc2-censor-refinement"
    / "reproduce/exact_continuation-manifest.json"
)
RESULTS = FAMILY / "results/inventory"
REPRO = FAMILY / "reproduce"
FIELDS = (
    "restriction_evidence",
    "mhc_allele_provenance",
    "source",
    "ms_outcome_status",
    "sample_attribution",
    "condition_id",
    "is_monoallelic",
    "has_peptide_level_allele",
    "is_potential_contaminant",
    "mhc_allele_set",
    "sample_note",
    "cell_name",
    "cell_line_name",
    "sample_mhc",
    "sample_mhc_origin",
    "condition_mhc_context",
    "condition_transduction",
    "condition_knockout_genes",
    "condition_control",
    "ip_antibody",
    "acquisition_status",
    "donor_status",
    "specimen_status",
    "experimental_origin_status",
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def kmers(peptide):
    return {peptide[i : i + 9] for i in range(len(peptide) - 8)}


def guard():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert peak < 2 * 1024**3, "RSS limit"
    assert psutil.virtual_memory().available >= 10 * 1024**3, "available RAM floor"
    assert shutil.disk_usage(INPUT).free >= 10 * 1024**3, "disk reserve"
    assert sum(p.stat().st_size for p in RESULTS.iterdir() if p.is_file()) <= 250 * 1024**2, (
        "output cap"
    )


def main():
    started = time.monotonic()
    signal.alarm(300)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    RESULTS.mkdir(parents=True, exist_ok=False)
    guard()
    summary = read(ADMISSION / "summary.json")
    assert read(ADMISSION / "independent-verification.json")["status"] == "passed"
    pins = {Path(item["path"]).name: item["sha256"] for item in summary["artifacts"]}
    launch = {
        "status": "running",
        "agent_model": "Codex / GPT-6",
        "utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.executable,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "code_sha256": sha(Path(__file__)),
        "input_hashes": pins,
        "curation": {"path": str(CURATION), "sha256": sha(CURATION)},
        "affinity_manifest": {"path": str(AFFINITY), "sha256": sha(AFFINITY)},
        "device": "cpu",
        "requested_modal_gpu": None,
        "environment": {
            k: os.environ.get(k)
            for k in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "limits": {
            "seconds": 300,
            "rss_bytes": 2 * 1024**3,
            "ram_floor_bytes": 10 * 1024**3,
            "disk_floor_bytes": 10 * 1024**3,
            "output_bytes": 250 * 1024**2,
        },
    }
    REPRO.mkdir(exist_ok=True)
    (REPRO / "source").mkdir(exist_ok=True)
    shutil.copyfile(__file__, REPRO / "source/inventory.py")
    if (REPRO / "launch.json").exists():
        raise FileExistsError(REPRO / "launch.json")
    write(REPRO / "launch.json", launch)
    for name in ("observations.jsonl", "contributor-summary.parquet", "contributor-links.parquet"):
        assert sha(INPUT / name) == pins[name]
    manifest = read(AFFINITY)
    known = {split: set() for split in ("train", "validation", "test")}
    for item in manifest["sources"]:
        path = Path(item["path"])
        assert sha(path) == item["sha256"]
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                known[row["split"]].add(row["peptide"])
    known_kmers = {
        split: set().union(*(kmers(p) for p in values)) for split, values in known.items()
    }
    contributors = {}
    for item in pq.read_table(INPUT / "contributor-summary.parquet").to_pylist():
        assert not item["overlap_unresolved"] and item["original_pmids"]
        contributors[item["provenance_id"]] = sorted(set(item["original_pmids"]))
    policy, studies, exposure = Counter(), defaultdict(set), Counter()
    seen, compact, multi_study = set(), [], 0
    for ordinal, line in enumerate((INPUT / "observations.jsonl").open()):
        row = json.loads(line)
        raw = row["provenance"]["raw_record"]
        identity, peptide = row["observation_id"], row["peptide"]
        assert identity not in seen
        seen.add(identity)
        pmids = contributors[raw["provenance_id"]]
        multi_study += len(pmids) > 1
        assert str(raw["pmid"]) in pmids
        for pmid in pmids:
            studies[pmid].add(peptide)
        policy[(str(raw["pmid"]), *(raw.get(key) for key in FIELDS))] += 1
        peptide_kmers = kmers(peptide)
        record = {
            "observation_id": identity,
            "admission_ordinal": ordinal,
            "source_ordinal": row["provenance"]["source_ordinal"],
            "peptide": peptide,
            "displayed_pmid": str(raw["pmid"]),
            "contributor_pmids": pmids,
            "provenance_id": raw["provenance_id"],
            "mhc_allele_set": raw["mhc_allele_set"],
            "condition_id": raw.get("condition_id"),
            "restriction_evidence": raw.get("restriction_evidence"),
        }
        for split in known:
            exact, shared = peptide in known[split], bool(peptide_kmers & known_kmers[split])
            record[f"ic50_{split}_exact"] = exact
            record[f"ic50_{split}_9mer"] = shared
            if exact:
                exposure[(str(raw["pmid"]), split, "exact")] += 1
            if shared:
                exposure[(str(raw["pmid"]), split, "any_9mer")] += 1
        compact.append(record)
        if ordinal % 4096 == 0:
            guard()
    assert len(compact) == 98220
    pq.write_table(pa.Table.from_pylist(compact), RESULTS / "metadata.parquet", compression="zstd")
    study_names = sorted(studies)
    kmer_mask = defaultdict(int)
    for i, pmid in enumerate(study_names):
        for peptide in studies[pmid]:
            for kmer in kmers(peptide):
                kmer_mask[kmer] |= 1 << i
    overlaps = []
    for pmid in study_names:
        shared_counts = Counter()
        for peptide in studies[pmid]:
            mask = 0
            for kmer in kmers(peptide):
                mask |= kmer_mask[kmer]
            for j, other in enumerate(study_names):
                if mask & (1 << j):
                    shared_counts[other] += 1
        for other in study_names:
            overlaps.append(
                {
                    "study": pmid,
                    "other_study": other,
                    "distinct_peptides": len(studies[pmid]),
                    "exact_overlap_peptides": len(studies[pmid] & studies[other]),
                    "any_9mer_overlap_peptides": shared_counts[other],
                }
            )
        guard()
    curation = yaml.safe_load(CURATION.read_text())
    assert isinstance(curation, list)
    curated = [item for item in curation if str(item["pmid"]) in studies]
    write(RESULTS / "current-curation.json", curated)
    write(
        RESULTS / "policy-groups.json",
        [{**dict(zip(("pmid", *FIELDS), key)), "rows": value} for key, value in policy.items()],
    )
    write(RESULTS / "study-overlap.json", overlaps)
    write(
        RESULTS / "initializer-exposure.json",
        [
            {"pmid": k[0], "ic50_split": k[1], "overlap": k[2], "rows": v}
            for k, v in sorted(exposure.items())
        ],
    )
    report = {
        "status": "inventory_complete_unverified",
        "observations": len(compact),
        "contributor_studies": study_names,
        "multiple_contributor_study_rows": multi_study,
        "distinct_peptides": len({r["peptide"] for r in compact}),
        "policy_groups": len(policy),
        "current_curated_studies": sorted(str(item["pmid"]) for item in curated),
        "ic50_distinct_input_peptides": {k: len(v) for k, v in known.items()},
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "interpretation": (
            "literal evidence and dependency inventory; no predictions or train/test assignment"
        ),
    }
    write(RESULTS / "summary.json", report)
    launch.update(
        status="inventory_complete_unverified",
        elapsed_seconds=report["elapsed_seconds"],
        peak_rss_bytes=report["peak_rss_bytes"],
        artifacts=[
            {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(RESULTS.iterdir())
            if p.is_file()
        ],
    )
    assert sha(Path(__file__)) == launch["code_sha256"]
    assert sha(CURATION) == launch["curation"]["sha256"]
    write(REPRO / "launch.json", launch)
    guard()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
