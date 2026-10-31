"""Qualify intended-product dependence without changing admitted measurements."""

import json
import resource
import shutil
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as pq
from hitlist.evidence_recipe import materialize_recipe, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.proteometools_response import RECIPE_ID
from hitlist.provenance import file_digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "data/measured_ms1_source_admission/2026-10-24_1635_v1"
HITLIST = Path("/private/tmp/hitlist-proteometools-ms1")
started = time.perf_counter()
assert not (DATA / "recipe-intended-groups").exists()
for relative in (
    "hitlist/proteometools_response.py",
    "tests/test_proteometools_ms1.py",
    "tasks/proteometools-ms1-spec.md",
):
    target = ROOT / "reproduce/final-grouping-source" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HITLIST / relative, target)
shutil.copy2(__file__, ROOT / "reproduce/final-grouping-source/refine_groups.py")
store = EvidenceStore.open(DATA / "source")
materialize_recipe(
    store,
    {
        "schema_version": 1,
        "name": "audited_two_pool_relative_MS1_intended_groups",
        "registered_recipe": RECIPE_ID,
    },
    DATA / "recipe-intended-groups",
)
verify_recipe_package(DATA / "recipe-intended-groups", source_package=DATA / "source")
old = json.loads((DATA / "recipe/examples.json").read_text())
new = json.loads((DATA / "recipe-intended-groups/examples.json").read_text())
assert [{k: v for k, v in r.items() if k != "dependence_component"} for r in old] == [
    {k: v for k, v in r.items() if k != "dependence_component"} for r in new
]
planned = pq.read_table(
    DATA / "source/proteometools_ms1/typed/planned_products.parquet"
).to_pylist()
pools = sorted({r["pool_name"] for r in new})
intended = {p: {r["peptide_sequence"] for r in planned if r["pool_name"] == p} for p in pools}
assert len(pools) == 2
assert not intended[pools[0]] & intended[pools[1]]
assert len({r["dependence_component"] for r in new}) == 2
prior = json.loads((ROOT / "results/pilot-overlap.json").read_text())
assert not set(prior["overlap"]["observed"]) & set.union(*intended.values())
for name in (
    "summary",
    "examples",
    "dispositions",
    "planned_dispositions",
    "dependence_groups",
    "contract",
):
    shutil.copy2(
        ROOT / "results" / f"{name}.json", ROOT / "results/initial-grouping" / f"{name}.json"
    )
    shutil.copy2(
        DATA / "recipe-intended-groups" / f"{name}.json", ROOT / "results" / f"{name}.json"
    )
shutil.copy2(
    ROOT / "results/recipe-manifest.json", ROOT / "results/initial-grouping/recipe-manifest.json"
)
shutil.copy2(DATA / "recipe-intended-groups/manifest.json", ROOT / "results/recipe-manifest.json")
report = {
    "status": "passed",
    "n_unchanged_examples": len(new),
    "n_intended_components": 2,
    "n_shared_observed_sequences": len(prior["overlap"]["observed"]),
    "n_shared_observed_sequences_intended_in_either_pool": 0,
    "seconds": time.perf_counter() - started,
    "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    "recipe_manifest": file_digest(DATA / "recipe-intended-groups/manifest.json"),
    "invocation": "PYTHONPATH=/private/tmp/hitlist-proteometools-ms1 /Users/iskander/code/shared-virtual-env/bin/python "
    + str(Path(__file__).resolve()),
    "hitlist_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HITLIST, text=True
    ).strip(),
    "frozen_source": {
        str(p.relative_to(ROOT)): file_digest(p)
        for p in sorted((ROOT / "reproduce/final-grouping-source").rglob("*"))
        if p.is_file()
    },
}
assert report["peak_rss_gib"] <= 3
(ROOT / "results/dependence-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
