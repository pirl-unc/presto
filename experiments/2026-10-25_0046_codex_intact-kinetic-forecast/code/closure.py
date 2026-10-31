"""Close frozen biological outcomes without refitting or revising their gates."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


config = read("reproduce/launch.json")
summary = read("results/evaluation/summary.json")
replay = read("results/independent-prediction-replay.json")
assert replay["passed"] and read("results/independent-admission.json")["passed"]
resources = {}
for stage in ["admit", "audit", "qualify", "controls", *config["fit_conditions"], "evaluate", "replay"]:
    resources[stage] = read(f"results/{stage}-resource.json")
    assert resources[stage]["passed"]

rows = pq.read_table(ROOT / "results/evaluation/predictions.parquet").to_pylist()
metrics = summary["all_models"]
lookup = {(r["model"], r["split"], r["space"]): r for r in metrics}
controls = list(dict.fromkeys(r["model"] for r in metrics if not r["model"].startswith("fit-")))
exacts = [c for c in config["fit_conditions"] if "-exact-" in c]
comparator = min(controls, key=lambda c: lookup[c, "val", "ratio"]["rmse"])
physical = min(exacts, key=lambda c: lookup[c, "val", "ratio"]["rmse"])
assert comparator == summary["comparator_selected_on_validation"]
assert physical == summary["physical_selected_on_validation"]

# Independently reduce squared errors to peptides before resampling; evaluation
# instead resampled row weights. No new samples, tuning, models or outcome access.
rng = np.random.default_rng(31415)
bootstrap_error = 0.0
for split in ("val", "test"):
    base = [r for r in rows if r["model"] == comparator and r["split"] == split]
    predicted = {r["assignment_id"]: r for r in rows if r["model"] == physical and r["split"] == split}
    peptide_errors = []
    for sequence in sorted({r["peptide_sequence"] for r in base}):
        cells = [r for r in base if r["peptide_sequence"] == sequence]
        w = np.array([r["weight"] for r in cells])
        w /= w.sum()
        peptide_errors.append([
            w @ np.array([(r["prediction"] - r["target"]) ** 2 for r in cells]),
            w @ np.array([(predicted[r["assignment_id"]]["prediction"] - r["target"]) ** 2 for r in cells]),
        ])
    errors = np.array(peptide_errors)
    gains = []
    for _ in range(1000):
        sampled = errors[rng.integers(len(errors), size=len(errors))].mean(axis=0)
        gains.append(100 * (1 - np.sqrt(sampled[1] / sampled[0])))
    interval = np.quantile(gains, [.025, .975])
    expected = summary["bootstrap"][split]["gain_percent_95_interval"]
    bootstrap_error = max(bootstrap_error, float(np.max(np.abs(interval - expected))))
    gain = 100 * (1 - lookup[physical, split, "ratio"]["rmse"] / lookup[comparator, split, "ratio"]["rmse"])
    assert abs(gain - summary["gain_percent"][split]) < 1e-12
assert bootstrap_error < 1e-10
assert not summary["primary_gate_passed"]

with (ROOT / "results/evaluation/metrics.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(metrics)

artifacts = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts:
        continue
    relative = str(path.relative_to(ROOT))
    if relative in {"results/closure.json", "results/closure-resource.json", "results/closure-started.json", "launch_logs/closure.log", "artifact-manifest.json"}:
        continue
    artifacts[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
(ROOT / "artifact-manifest.json").write_text(json.dumps({"files": artifacts}, indent=2) + "\n")
result = {
    "passed": True,
    "biological_primary_gate_passed": False,
    "n_completed_physical_fits": len(config["fit_conditions"]),
    "n_controls": len(controls),
    "n_prediction_rows": len(rows),
    "n_overall_metric_records": len(metrics),
    "validation_selection_replayed": True,
    "bootstrap_max_absolute_error": bootstrap_error,
    "independent_prediction_replay": "results/independent-prediction-replay.json",
    "total_stage_wall_seconds_before_closure": sum(r["wall_seconds"] for r in resources.values()),
    "maximum_polled_stage_rss_bytes_before_closure": max(r["peak_polled_rss_bytes"] for r in resources.values()),
    "artifact_logical_bytes": sum(r["size_bytes"] for r in artifacts.values()),
    "n_hashed_artifacts": len(artifacts),
    "unvalidated_expected_counts_are_not_concentrations": True,
    "software_publication_pending": True,
    "report_status": "biological_evaluation_and_independent_replay_complete",
}
(ROOT / "results/closure.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
