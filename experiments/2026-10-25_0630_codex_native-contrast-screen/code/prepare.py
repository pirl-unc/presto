"""Bind tested releases and check covariates before allowing any fit."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

from core import CFG, ROOT, dump, inner_fold

assert json.loads((ROOT / "results/qualification.json").read_text())["passed"]
bindings = {}
for role, path in CFG["required_release_receipts"].items():
    receipt_path = Path(path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["version"] == CFG["required_versions"][role]
    assert len(receipt["files"]) == 2 and all(f["byte_equal"] for f in receipt["files"])
    release_dir = Path(receipt["verified_artifacts"])
    release = json.loads((release_dir / "release.json").read_text())
    assert (
        release["version"] == receipt["version"]
        and release["source_commit"] == receipt["source_commit"]
    )
    assert int(release["workflow_run_id"]) == int(receipt["successful_release_run"])
    for f in receipt["files"]:
        artifact = release_dir / f["filename"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == f["sha256"]
        assert release["files"][f["filename"]]["sha256"] == f["sha256"]
    if role == "source":
        assert receipt["source_commit"] == CFG["required_source_commit"]
    else:
        for name, checksum in CFG["recipe_code_sha256"].items():
            content = subprocess.check_output(
                ["git", "show", receipt["source_commit"] + ":" + name],
                cwd=CFG["hitlist_repo"],
            )
            assert hashlib.sha256(content).hexdigest() == checksum, name
    bindings[role] = dict(
        receipt=receipt,
        receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
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
dump("results/release-bindings.json", bindings)
dump(
    "results/preparation.json",
    dict(
        passed=True,
        partitions=summary,
        inner_training_components=dict(folds),
        numerical_targets_read=False,
    ),
)
print("Released source/recipe bindings and covariate partitions verified", flush=True)
