"""Shared recipe admission and separate model-facing inputs/outcomes."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hitlist.evidence_recipe import materialize_recipe, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.intact_kinetics import RECIPE_ID

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
store = EvidenceStore.open(config["source_package"])
assert store._manifest_hash() == config["source_manifest_sha256"]
destination = Path(config["recipe_package"])
materialize_recipe(store, {"schema_version": 1, "name": "intact_kinetic_forecast", "registered_recipe": RECIPE_ID}, destination)
manifest = verify_recipe_package(destination, source_package=store.directory)
(ROOT / "results/admission.json").write_text(json.dumps(manifest, indent=2) + "\n")
assert manifest["coverage"]["support_passed"], "Frozen support gate fails; no model may run"
covariates = pq.read_table(destination / "typed/covariates.parquet").to_pylist()
examples = {r["assignment_id"]: r for r in pq.read_table(destination / "typed/examples.parquet").to_pylist()}
fingerprints = {}
for split in ("train", "val", "test"):
    rows = [r for r in covariates if r["split"] == split]
    out = ROOT / "results/data" / split
    out.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "covariates.parquet")
    pq.write_table(pa.Table.from_pylist([{"assignment_id": r["assignment_id"], "target": examples[r["assignment_id"]]["target_log10_ratio"]} for r in rows]), out / "outcomes.parquet")
    fingerprints[split] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
(ROOT / "results/data-fingerprints.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
print(json.dumps(manifest["coverage"]), flush=True)
