"""Verify the archived proposed allocations and create unit-labelled summaries."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

family = Path(__file__).resolve().parents[1]
r = json.loads((family / "results/audit.json").read_text())
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r["files"]["run.tar.gz"]["path"]).parent
for name, expected in r["files"].items():
    data = (root / name).read_bytes()
    assert len(data) == expected["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == expected["sha256"]
inventory = json.loads((root / "inventory.json").read_text())
reports = []
summary = {"n_verified_artifacts": len(inventory), "policies": {}}
with tarfile.open(root / "run.tar.gz") as archive:
    files = {}
    for member in archive.getmembers():
        assert member.isfile() and member.name in inventory and member.name not in files
        data = archive.extractfile(member).read()
        assert len(data) == inventory[member.name]["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == inventory[member.name]["sha256"]
        files[member.name] = data
    assert set(files) == set(inventory)
for policy in ["absolute_deficit_v1", "relative_deficit_v2"]:
    assignments = pd.read_parquet(io.BytesIO(files[policy + "/assignments.parquet"]))
    if policy == "absolute_deficit_v1":
        original = assignments
    else:
        assert assignments.drop(columns="source_partition").equals(
            original.drop(columns="source_partition")
        )
    assert assignments.groupby("component_id").source_partition.nunique().eq(1).all()
    counts = pd.read_csv(io.BytesIO(files[policy + "/assay_support.tsv"]), sep="\t")
    counts.insert(0, "allocation", policy)
    reports.append(counts)
    audit = json.loads(files[policy + "/lineage_audit.json"])
    summary["policies"][policy] = {
        "n_observations_by_partition": assignments.source_partition.value_counts().to_dict(),
        "n_components": assignments.component_id.nunique(),
        "n_moved_observations": int(
            assignments.source_partition.ne(original.source_partition).sum()
        ),
        "n_split_components": 0,
        "lineage_audit": audit,
    }
    for name in [
        "assay_support.tsv",
        "endpoint_support.tsv",
        "class_support.tsv",
        "lineage_audit.json",
    ]:
        (family / "results" / (policy + "-" + name)).write_bytes(files[policy + "/" + name])
pd.concat(reports).to_csv(family / "results/support-comparison.tsv", sep="\t", index=False)
(family / "results/qualification-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print("Verified", len(files), "archive members and all intact component/observation identities")
