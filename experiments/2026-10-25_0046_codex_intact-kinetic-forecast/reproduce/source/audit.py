"""Independent temporal recipe replay from the prior source-verified census."""

import collections
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
prior = Path(config["prior_support_directory"])
inventory = json.loads((prior / "artifact-manifest.json").read_text())
for name in ("results/independent-replay.json", "results/kinetic-candidate-decisions.jsonl"):
    assert hashlib.sha256((prior / name).read_bytes()).hexdigest() == inventory[name]["sha256"]
assert json.loads((prior / "results/independent-replay.json").read_text())["passed"]
original = [json.loads(line) for line in (prior / "results/kinetic-candidate-decisions.jsonl").read_text().splitlines()]
by_id = {r["assignment_id"]: r for r in original}
accepted = [r for r in original if not r["reasons"]]
parents = {r["acquisition_id"]: r for r in accepted if r["start_1based"] == 1 and r["end_1based"] == 140}
assert len(parents) == 21
products = [r for r in accepted if len(r["peptide_sequence"]) != 140]
train_times = collections.defaultdict(set)
for r in products:
    if r["time_minutes"] <= 80:
        train_times[r["peptide_sequence"]].add(r["time_minutes"])
eligible = {s for s, times in train_times.items() if len(times) >= 2}
expected = {r["assignment_id"]: r for r in products if r["peptide_sequence"] in eligible}
package = Path(config["recipe_package"])
rows = pq.read_table(package / "typed/examples.parquet").to_pylist()
assert {r["assignment_id"] for r in rows} == set(expected)
assert len(rows) == len(expected)
maximum_error = 0.
for row in rows:
    original_row = expected[row["assignment_id"]]
    parent = parents[original_row["acquisition_id"]]
    assert row["normalizer_assignment_id"] == parent["assignment_id"]
    assert row["normalizer_measurement_id"] == parent["measurement_id"]
    assert row["quantitative_value"] == original_row["raw_intensity"]
    assert row["normalizer_value"] == parent["raw_intensity"]
    target = math.log10(original_row["raw_intensity"] / parent["raw_intensity"])
    maximum_error = max(maximum_error, abs(row["target_log10_ratio"] - target))
    assert abs(row["target_log10_ratio"] - target) < 2e-14
    assert row["split"] == ("train" if row["time_minutes"] <= 80 else "val" if row["time_minutes"] == 100 else "test")
    assert row["start_1based"] == original_row["start_1based"] and row["end_1based"] == original_row["end_1based"]
    p, s = row["parent_sequence"], row["peptide_sequence"]
    assert [i + 1 for i in range(len(p)-len(s)+1) if p[i:i+len(s)] == s] == [row["start_1based"]]
support = {}
for split in ("train", "val", "test"):
    selected = [r for r in rows if r["split"] == split]
    seqs = {r["peptide_sequence"] for r in selected}
    times = {s: {r["time_minutes"] for r in selected if r["peptide_sequence"] == s} for s in seqs}
    frequency = collections.Counter((r["peptide_sequence"], r["time_minutes"]) for r in selected)
    for row in selected:
        weight = 1 / len(seqs) / len(times[row["peptide_sequence"]]) / frequency[row["peptide_sequence"], row["time_minutes"]]
        assert abs(row["weight"] - weight) < 1e-16
    assert abs(sum(r["weight"] for r in selected) - 1) < 1e-13
    covariates = pq.read_table(ROOT / f"results/data/{split}/covariates.parquet")
    assert not {"quantitative_value", "normalizer_value", "target_log10_ratio"} & set(covariates.column_names)
    outcomes = pq.read_table(ROOT / f"results/data/{split}/outcomes.parquet").to_pylist()
    assert outcomes == [{"assignment_id": r["assignment_id"], "target": r["target_log10_ratio"]} for r in selected]
    assert all({k:v for k,v in r.items() if k in covariates.column_names} == c for r,c in zip(selected,covariates.to_pylist(),strict=True))
    support[split] = {"n_sequences":len(seqs), "n_observations":len(selected)}
decisions = pq.read_table(package / "typed/decisions.parquet").to_pylist()
assert len(decisions) == 8138
for row in decisions:
    if row["source_view_role"] != "kinetics":
        assert row["status"] == "excluded" and "outside_kinetics_view" in json.loads(row["reasons_json"])
    else:
        prior_row = by_id[row["assignment_id"]]
        assert set(prior_row["reasons"]) <= set(json.loads(row["reasons_json"]))
        assert row["quantitative_value"] == prior_row["raw_intensity"]
        if row["assignment_id"] not in expected and not prior_row["reasons"]:
            if len(row["peptide_sequence"]) == 140:
                assert row["status"] == "normalizer_only"
            else:
                assert "fewer_than_two_fitting_times" in json.loads(row["reasons_json"])
result={"passed":True,"cohort_support":support,"n_decisions":len(decisions),"n_verified_targets":len(rows),
        "max_target_error":maximum_error,"n_train_eligible_sequences":len(eligible),
        "unique_source_origins_verified":True,"source_census_sha256":inventory["results/kinetic-candidate-decisions.jsonl"]["sha256"],
        "source_replay_sha256":inventory["results/independent-replay.json"]["sha256"],
        "recipe_manifest_sha256":hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest(),
        "predictive_outcome_access":False}
(ROOT/'results/independent-admission.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
