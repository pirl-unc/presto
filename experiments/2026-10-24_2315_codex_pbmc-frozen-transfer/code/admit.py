"""Canonical recipe admission and sealed label-free transfer inputs."""

import hashlib
import json
import signal
from pathlib import Path

import pyarrow.parquet as pq
from hitlist.evidence_recipe import materialize_recipe
from hitlist.evidence_store import EvidenceStore
from hitlist.pbmc_composition import RECIPE_ID

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stop(_signal, _frame):
    raise RuntimeError("Resource supervisor terminated admission")


signal.signal(signal.SIGTERM, stop)
config = json.loads((ROOT / "reproduce/launch.json").read_text())
assert json.loads((ROOT / "results/source-resource.json").read_text())["passed"]
for relative, digest in config["predictor_hashes"].items():
    assert sha(Path(config["predictor_directory"]) / relative) == digest
runtime = {}
for relative in (
    "hitlist/pbmc_composition.py",
    "hitlist/evidence_recipe.py",
    "hitlist/distribution_recipe.py",
    "hitlist/version.py",
):
    source = Path(config["hitlist_checkout"]) / relative
    destination = ROOT / "reproduce/recipe_source" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    runtime[relative] = sha(source)
write(ROOT / "results/recipe-runtime.json", runtime)
store = EvidenceStore.open(config["package"])
recipe = {
    "schema_version": 1,
    "name": "pbmc_frozen_external_transfer",
    "registered_recipe": RECIPE_ID,
}
materialize_recipe(store, recipe, config["recipe_package"])
package = Path(config["recipe_package"])
summary = json.loads((package / "summary.json").read_text())
write(ROOT / "results/admission.json", summary)
write(ROOT / "results/recipe-manifest.json", json.loads((package / "manifest.json").read_text()))
covariates = sorted(
    pq.read_table(package / "composition/typed/covariates.parquet").to_pylist(),
    key=lambda r: r["measurement_id"],
)
examples = pq.read_table(package / "composition/typed/examples.parquet").to_pylist()
write(ROOT / "results/covariates.json", covariates)
files = ["covariates.json"]
for cohort in ("CTRL", "LUPUS"):
    labels = {
        r["measurement_id"]: {
            "y": r["target_centered_log10_intensity"],
            "p": r["target_intensity_fraction"],
        }
        for r in examples
        if r["cohort"] == cohort
    }
    name = cohort + "-labels.json"
    write(ROOT / "results" / name, labels)
    files.append(name)
write(ROOT / "results/input-seal.json", {name: sha(ROOT / "results" / name) for name in files})
print(json.dumps(summary), flush=True)
