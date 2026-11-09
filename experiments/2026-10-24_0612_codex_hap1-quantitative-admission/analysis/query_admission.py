"""Exercise the shared query API against the already verified original package."""

import hashlib
import json
from pathlib import Path

import hitlist.evidence_store as module

experiment = Path(__file__).resolve().parents[1]
record_dir = experiment / "results/admission-v2"
receipt = json.loads((record_dir / "result.json").read_text())
directory = Path(receipt["destination"])
manifest_bytes = (directory / "manifest.json").read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == receipt["manifest_sha256"]
store = module.EvidenceStore(directory, json.loads(manifest_bytes))
examples = store.query(
    "hap1_quantities",
    select=[
        "quantitative_value",
        "value_status",
        "context.condition_knockout_genes",
        "context.mhc_genotype",
        "context.measured_pool",
    ],
    joins=["context"],
    where={"native.quantity_kind": "normalized_peptide_intensity"},
    limit=3,
).table.to_pylist()
unknown = store.query(
    "hap1_precursors",
    select=["sample_id", "context.condition_knockout_genes"],
    joins=["context"],
    where={"sample_id": {"op": "is_null", "value": True}},
    limit=2,
).table.to_pylist()
assert len(examples) == 3 and len(unknown) == 2
assert all(row["context.condition_knockout_genes"] is None for row in unknown)
result = {
    "status": "passed",
    "package_manifest_sha256": receipt["manifest_sha256"],
    "scope": "query-only after writer source replay and independent cell check; no extra open/replay",
    "evidence_store_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    "n_bindings": len(store.bindings),
    "quantities": examples,
    "unmapped": unknown,
    "prior_failure": "all-null optional join raised ArrowTypeError; Hitlist805 regression fails before correction",
}
(record_dir / "evidence-store-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print("Actual package query and nullable context joins pass;", len(store.bindings), "relations.")
