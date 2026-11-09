"""Use the shared source/recipe; save label-free inputs and separately sealed labels."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq
from hitlist.evidence_recipe import materialize_recipe
from hitlist.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((ROOT / "reproduce/launch.json").read_text())
    source, package = REPO / config["source"], REPO / config["recipe_directory"]
    assert sha(source / "manifest.json") == config["source_manifest_sha256"]
    for relative, digest in config["hitlist_files_sha256"].items():
        assert sha(Path(config["hitlist_checkout"]) / relative) == digest
    start = time.perf_counter()
    store = EvidenceStore.open(source)
    manifest_path = materialize_recipe(
        store,
        {
            "schema_version": 1,
            "name": "prospective_baseline",
            "registered_recipe": config["recipe"],
        },
        package,
    )
    manifest = json.loads(manifest_path.read_text())
    coverage = manifest["coverage"]
    shutil.copy2(manifest_path, ROOT / "results/recipe-manifest.json")
    result = {
        "recipe_manifest_sha256": sha(manifest_path),
        "coverage": coverage,
        "wall_seconds": time.perf_counter() - start,
    }
    (ROOT / "results/admission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(coverage, indent=2), flush=True)
    if not coverage["support_passed"]:
        raise RuntimeError("Frozen component/group support gate failed; no fitting")
    path = package / "composition/typed"
    covariates = pq.read_table(path / "covariates.parquet").to_pylist()
    examples = pq.read_table(path / "examples.parquet").to_pylist()
    assert [r["measurement_id"] for r in covariates] == [r["measurement_id"] for r in examples]
    (ROOT / "results/covariates.json").write_text(
        json.dumps(covariates, separators=(",", ":")) + "\n"
    )
    for split in ("train", "val", "test"):
        labels = {
            r["measurement_id"]: {
                "y": r["target_centered_log10_intensity"],
                "p": r["target_intensity_fraction"],
            }
            for r in examples
            if r["split"] == split
        }
        (ROOT / f"results/{split}-labels.json").write_text(
            json.dumps(labels, separators=(",", ":")) + "\n"
        )
    hashes = {p.name: sha(p) for p in sorted((ROOT / "results").glob("*labels.json"))}
    hashes["covariates.json"] = sha(ROOT / "results/covariates.json")
    (ROOT / "results/input-seal.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
