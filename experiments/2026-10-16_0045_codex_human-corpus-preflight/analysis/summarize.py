"""Verify the original run and derive small, inspectable qualification reports."""

import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

family = Path(__file__).resolve().parents[1]
receipt = json.loads((family / "results/preflight.json").read_text())
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(receipt["local_paths"]["run.tar.gz"]).parent
inventory = json.loads((root / "inventory.json").read_text())
for name, pin in receipt["files"].items():
    data = (root / name).read_bytes()
    assert len(data) == pin["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == pin["sha256"]
with tarfile.open(root / "run.tar.gz") as archive:
    files = {}
    for member in archive.getmembers():
        assert member.isfile() and member.name in inventory
        data = archive.extractfile(member).read()
        pin = inventory[member.name]
        assert len(data) == pin["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == pin["sha256"]
        assert member.name not in files
        files[member.name] = data
    assert set(files) == set(inventory)

report_names = [
    "data_funnel.json",
    "data_funnel.csv",
    "split_support.csv",
    "mhc_sequence_coverage.json",
    "mhc_sequence_coverage_post_filter.json",
    "frozen_assignments/coverage.json",
    "frozen_assignments/lineage_audit.json",
    "frozen_membership/coverage.json",
    "frozen_membership/model_peptide_audit.json",
]
for name in report_names:
    (family / "results" / name.replace("/", "-")).write_bytes(files[name])
rows = list(csv.DictReader(io.StringIO(files["split_support.csv"].decode())))
counts = {
    split: {
        r["target"]: {
            k: int(r[k])
            for k in ["count", "exact", "censored_lower", "censored_upper", "positive", "negative"]
            if r[k]
        }
        for r in rows
        if r["split"] == split
    }
    for split in ["train", "val", "test"]
}
audit = json.loads(files["frozen_assignments/lineage_audit.json"])
funnel = json.loads(files["data_funnel.json"])
summary = {
    "schema_version": 1,
    "status": "data preflight passed; output readiness not established",
    "n_verified_run_files": len(files),
    "partition_coverage": json.loads(files["frozen_membership/coverage.json"]),
    "supervision_counts": counts,
    "source_counts_before_cap": funnel["source_loader"]["counts_before_cap"],
    "source_counts_after_cap": funnel["source_loader"]["counts"],
    "drop_reasons": funnel["drop_reasons"],
    "overlaps": [
        {
            **{k: pair[k] for k in ["left", "right"]},
            "n_shared_values": {k: v["n_shared_values"] for k, v in pair["overlaps"].items()},
        }
        for pair in audit["pairs"]
    ],
    "lineage_coverage": audit["partitions"],
    "sampling_deviation": {
        "requested": "reservoir",
        "actual": "independent per-family random.Random(59).sample of full eligible lists",
        "issue": "https://github.com/pirl-unc/presto/issues/103",
    },
    "limitations": [
        "Producer Hitlist 1.76.0 predates curation fixes #694/#696.",
        "No model constructed; no predictions or performance metrics exist.",
        "All 221 output declarations remain undeclared; support is exploratory.",
        "Positive-only elution and human-source-only foreignness cannot qualify binary discrimination.",
        "Validation has no direct KD or kinetics observations; test has four direct KD and no kinetics.",
        "All donor/specimen/acquisition identities unresolved; most experimental origins unresolved.",
        "Studies overlap across partitions; peptide-disjoint does not imply study or biological independence.",
        "Sequence coverage counts allele occurrences; species buckets use assay host context first. Nine murine host occurrences do not mean nonhuman MHC: every retained resolved allele begins HLA-.",
        "Strict MHC filtering removes unresolved alleles within bags and then empty bags; original sets remain in membership metadata.",
        "Source exports have unverified release dates; historical original input and actual runtime receipts are preserved.",
    ],
}
(family / "results/qualification-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(
    json.dumps(
        {
            "n_verified_run_files": len(files),
            "n_samples": 9000,
            "output": str(family / "results/qualification-summary.json"),
        }
    )
)
