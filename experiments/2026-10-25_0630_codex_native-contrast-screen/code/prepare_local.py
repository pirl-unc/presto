"""Bind exact merged/full-CI code and covariates before local fitting."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

from core import CFG, ROOT, dump, inner_fold

assert json.loads((ROOT / "results/qualification.json").read_text())["passed"]
evidence_path = ROOT / "reproduce/merged-code-evidence.json"
evidence = json.loads(evidence_path.read_text())
assert evidence["full_CI_passed"]
expected_jobs = {
    "lint",
    "integration",
    "coverage",
    *["unit (" + version + ")" for version in ("3.10", "3.11", "3.12", "3.13", "3.14")],
}
assert {j["name"] for j in evidence["tests"]} == expected_jobs
assert all(
    j["status"] == "completed" and j["conclusion"] == "success"
    for j in evidence["tests"]
)
assert any(
    j["name"] == "build" and j["conclusion"] == "success" for j in evidence["docs"]
)
trees = [
    subprocess.check_output(
        ["git", "rev-parse", revision + "^{tree}"], cwd=CFG["hitlist_repo"], text=True
    ).strip()
    for revision in (evidence["tested_head"], evidence["merged_commit"])
]
assert trees[0] == trees[1], "Merged tree differs from fully tested tree"
for name, checksum in CFG["recipe_code_sha256"].items():
    content = subprocess.check_output(
        ["git", "show", evidence["merged_commit"] + ":" + name], cwd=CFG["hitlist_repo"]
    )
    assert hashlib.sha256(content).hexdigest() == checksum, name
bindings = dict(
    evidence=evidence,
    evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    identical_tree=trees[0],
    publication_still_required=True,
)

package = Path(CFG["package"])
manifest = json.loads((package / "manifest.json").read_text())
assert manifest["coverage"]["qualified_for_screen"]
assert manifest["recipe"]["registered_recipe"] == "pxd018722_native_contrast_v1"
for relative, record in manifest["files"].items():
    p = package / relative
    assert p.stat().st_size == record["size_bytes"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"]
rows = pq.read_table(package / "typed/covariates.parquet").to_pylist()
assert all("target_log10_ratio_change" not in r for r in rows)
group_parts, parent_parts, endpoint_parts = (
    defaultdict(set),
    defaultdict(set),
    defaultdict(set),
)
for row in rows:
    group_parts[row["dependency_group_id"]].add(row["split"])
    parent_parts[row["parent_sequence_sha256"]].add(row["split"])
    for key in ("peptide_id", "reference_peptide_id"):
        endpoint_parts[row[key]].add(row["split"])
assert all(
    len(parts) == 1
    for mapping in (group_parts, parent_parts, endpoint_parts)
    for parts in mapping.values()
)
summary = {}
for part in ("train", "val", "test"):
    selected = [r for r in rows if r["split"] == part]
    assert len(selected) == CFG["expected_n_contrasts"][part]
    assert (
        len({r["dependency_group_id"] for r in selected})
        == CFG["expected_n_components"][part]
    )
    assert abs(sum(r["weight"] for r in selected) - 1) < 1e-12
    summary[part] = dict(
        n_contrasts=len(selected),
        n_components=len({r["dependency_group_id"] for r in selected}),
    )
training = [r for r in rows if r["split"] == "train"]
folds = Counter(
    inner_fold(group) for group in {r["dependency_group_id"] for r in training}
)
assert set(folds) == {0, 1, 2}
dump("results/code-bindings.json", bindings)
dump(
    "results/preparation.json",
    dict(
        passed=True,
        partitions=summary,
        inner_training_components=dict(folds),
        numerical_targets_read=False,
    ),
)
print("Merged full-CI code and covariate partitions verified", flush=True)
