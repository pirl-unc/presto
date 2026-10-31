"""Close the registered support failure without scoring target outcomes."""

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
summary = json.loads((ROOT / "results/admission.json").read_text())
audit = json.loads((ROOT / "results/data-audit.json").read_text())
assert audit["passed"] and not any(summary["support_passed"].values())
resources = {
    s: json.loads((ROOT / f"results/{s}-resource.json").read_text())
    for s in ("source", "admit", "audit")
}
assert all(r["passed"] for r in resources.values())
assert sum(r["wall_seconds"] for r in resources.values()) < config["limits"]["family_seconds"]
for relative, digest in config["predictor_hashes"].items():
    path = Path(config["predictor_directory"]) / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
source = Path(config["package"]) / "external/pxd009438/typed"
for alias in pq.read_table(source / "label_aliases.parquet").to_pylist():
    experiment = alias["summary_experiment_raw"]
    assert alias["match_rule"] == "literal_equal"
    assert alias["quantity_column_raw"] == "Intensity " + experiment
    assert alias["channel_id"] == "pxd009438:run:" + experiment
    assert alias["source_locator"] == "txt/peptides.txt:header;txt/summary.txt"
package = Path(config["recipe_package"])
published = ROOT / "results/cohort"
published.mkdir()
for path in (package / "composition/typed").glob("*.parquet"):
    shutil.copyfile(path, published / path.name)
    assert pq.read_table(path).equals(pq.read_table(published / path.name))
covariates = json.loads((ROOT / "results/covariates.json").read_text())
examples = pq.read_table(published / "examples.parquet").to_pylist()
labels = {
    "quantitative_value",
    "log10_intensity",
    "target_centered_log10_intensity",
    "target_intensity_fraction",
}
assert sorted(covariates, key=lambda r: r["measurement_id"]) == sorted(
    [{k: v for k, v in r.items() if k not in labels} for r in examples],
    key=lambda r: r["measurement_id"],
)
for cohort in ("CTRL", "LUPUS"):
    actual = json.loads((ROOT / f"results/{cohort}-labels.json").read_text())
    expected = {
        r["measurement_id"]: {
            "y": r["target_centered_log10_intensity"],
            "p": r["target_intensity_fraction"],
        }
        for r in examples
        if r["cohort"] == cohort
    }
    assert actual == expected
decision = {
    "status": "closed_support_failure_before_prediction",
    "cohort_support": summary["cohort_support"],
    "required": {"n_components": 20, "n_groups": 20, "n_observations": 100},
    "support_passed": summary["support_passed"],
    "n_models_scored": 0,
    "n_training_updates": 0,
    "prediction_dumps": "none: neither cohort passes prospectively required support",
    "no_training_or_validation_split": "registered external-only evaluation; never reached model scoring",
    "biological_prediction_result": None,
    "cohort_audit": audit,
    "resources": resources,
    "supplemental_closure_checks": "all14literal aliases; label-free projection and separate label files agree",
    "full_availability_goal": "active_unachieved",
}
(ROOT / "results/decision.json").write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps(decision), flush=True)
