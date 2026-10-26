"""Independent replay from admitted originals and individual contributor links."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import signal
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

FAMILY = Path(__file__).resolve().parents[1]
INPUT = Path("/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01")
RESULTS = FAMILY / "results/inventory"
KEYS = (
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
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pieces(peptide):
    return set(re.findall(r"(?=(.{9}))", peptide))


def guard():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    assert rss < 2 * 1024**3
    assert psutil.virtual_memory().available >= 10 * 1024**3
    assert shutil.disk_usage(INPUT).free >= 10 * 1024**3
    return rss


def verify():
    started = time.monotonic()
    signal.alarm(300)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    target = FAMILY / "results/independent-verification.json"
    if target.exists():
        raise FileExistsError(target)
    launch = read(FAMILY / "reproduce/launch.json")
    for name, expected in launch["input_hashes"].items():
        assert sha(INPUT / name) == expected
    for artifact in launch["artifacts"]:
        assert sha(artifact["path"]) == artifact["sha256"]
    assert sha(launch["affinity_manifest"]["path"]) == launch["affinity_manifest"]["sha256"]
    assert sha(launch["curation"]["path"]) == launch["curation"]["sha256"]
    known, all_pieces = defaultdict(set), defaultdict(set)
    for item in read(launch["affinity_manifest"]["path"])["sources"]:
        assert sha(item["path"]) == item["sha256"]
        with open(item["path"]) as stream:
            for line in stream:
                row = json.loads(line)
                known[row["split"]].add(row["peptide"])
                all_pieces[row["split"]].update(pieces(row["peptide"]))
    contributors = defaultdict(set)
    link_count = 0
    for batch in pq.ParquetFile(INPUT / "contributor-links.parquet").iter_batches(
        use_threads=False
    ):
        for item in batch.to_pylist():
            link_count += 1
            assert item["original_pmid"]
            contributors[item["provenance_id"]].add(str(item["original_pmid"]))
    metadata = pq.read_table(RESULTS / "metadata.parquet").to_pylist()
    assert len({item["observation_id"] for item in metadata}) == len(metadata) == 98220
    policies, exposure, studies = Counter(), Counter(), defaultdict(set)
    peptide_pieces = {}
    with (INPUT / "observations.jsonl").open() as stream:
        n = 0
        for n, (line, saved) in enumerate(zip(stream, metadata, strict=True), 1):
            original = json.loads(line)
            raw = original["provenance"]["raw_record"]
            peptide = original["peptide"]
            pmids = sorted(contributors[raw["provenance_id"]])
            assert saved["observation_id"] == original["observation_id"]
            assert saved["admission_ordinal"] == n - 1
            assert saved["source_ordinal"] == original["provenance"]["source_ordinal"]
            assert saved["peptide"] == peptide and saved["displayed_pmid"] == str(raw["pmid"])
            assert saved["provenance_id"] == raw["provenance_id"]
            assert saved["contributor_pmids"] == pmids
            for field in ("condition_id", "restriction_evidence", "mhc_allele_set"):
                assert saved[field] == raw[field]
            policies[(str(raw["pmid"]), *(raw.get(key) for key in KEYS))] += 1
            peptide_pieces[peptide] = pieces(peptide)
            for pmid in pmids:
                studies[pmid].add(peptide)
            for split in ("train", "validation", "test"):
                exact = peptide in known[split]
                shared = any(fragment in all_pieces[split] for fragment in peptide_pieces[peptide])
                assert saved[f"ic50_{split}_exact"] == exact
                assert saved[f"ic50_{split}_9mer"] == shared
                if exact:
                    exposure[(str(raw["pmid"]), split, "exact")] += 1
                if shared:
                    exposure[(str(raw["pmid"]), split, "any_9mer")] += 1
            if n % 4096 == 0:
                guard()
    expected_policies = Counter(
        {
            tuple(row[k] for k in ("pmid", *KEYS)): row["rows"]
            for row in read(RESULTS / "policy-groups.json")
        }
    )
    assert policies == expected_policies
    expected_exposure = Counter(
        {
            (row["pmid"], row["ic50_split"], row["overlap"]): row["rows"]
            for row in read(RESULTS / "initializer-exposure.json")
        }
    )
    assert exposure == expected_exposure
    study_pieces = {
        pmid: set().union(*(peptide_pieces[p] for p in peptides))
        for pmid, peptides in studies.items()
    }
    overlaps = read(RESULTS / "study-overlap.json")
    assert len(overlaps) == len(studies) ** 2
    assert len({(x["study"], x["other_study"]) for x in overlaps}) == len(overlaps)
    for row in overlaps:
        left, right = row["study"], row["other_study"]
        assert row["distinct_peptides"] == len(studies[left])
        assert row["exact_overlap_peptides"] == len(studies[left].intersection(studies[right]))
        assert row["any_9mer_overlap_peptides"] == sum(
            bool(peptide_pieces[p] & study_pieces[right]) for p in studies[left]
        )
        guard()
    current = yaml.safe_load(Path(launch["curation"]["path"]).read_text())
    assert read(RESULTS / "current-curation.json") == [
        r for r in current if str(r["pmid"]) in studies
    ]
    report = {
        "status": "passed",
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_rows_and_metadata_verified": n,
        "contributor_links_verified": link_count,
        "policy_groups_verified": len(policies),
        "study_overlap_pairs_verified": len(overlaps),
        "distinct_peptides": len(peptide_pieces),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": guard(),
        "verification_source_sha256": sha(__file__),
        "predictive_qualification": False,
    }
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    verify()
