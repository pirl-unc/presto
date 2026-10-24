"""Replay observation layers from the unchanged shared recipe's source snapshots."""

import gzip
import hashlib
import json
import math
import resource
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def condition_layers(recipe, condition):
    source = pq.read_table(
        recipe / "source_tables/filtered_vectors.parquet",
        filters=[("condition_id", "=", condition)],
    ).to_pylist()
    original = []
    for row in source:
        for slot, (t, v) in enumerate(zip(row["times_raw"].split(";"), row["intensities_raw"].split(";"), strict=True)):
            original.append(dict(
                peptide=row["peptide_sequence"], bio=row["biological_replicate_raw"],
                technical_original=int(row["technical_replicate_raw"]),
                time=float(t), original_intensity=float(v),
                vector_id=row["source_record_id"], slot=slot,
            ))
    # No mock baselines are permitted in the admitted training conditions.
    assert {r["bio"] for r in original} == {r["bio"] for r in original if r["time"] == 0}
    il = defaultdict(list)
    for r in original:
        il[r["peptide"].replace("I", "L"), r["bio"], r["technical_original"], r["time"]].append(r)
    for r in original:
        group = il[r["peptide"].replace("I", "L"), r["bio"], r["technical_original"], r["time"]]
        r["intensity"] = math.fsum(x["original_intensity"] for x in group)
        r["il_contributors"] = [[x["vector_id"], x["slot"]] for x in group]
        r["technical_remapped"] = {3: 1, 4: 2}.get(r["technical_original"], r["technical_original"])
    zero = defaultdict(list)
    for r in original:
        if r["time"] == 0:
            zero[r["peptide"], r["bio"], r["technical_remapped"]].append(r)
    baselines = {k: max(r["intensity"] for r in rows) for k, rows in zero.items()}
    noise = statistics.median(baselines.values())
    grouped = defaultdict(list)
    for r in original:
        key = r["peptide"], r["bio"], r["technical_remapped"]
        r["baseline"] = max(noise, baselines[key]) if key in baselines else None
        r["baseline_contributors"] = [[x["vector_id"], x["slot"]] for x in zero.get(key, [])]
        r["corrected"] = None if r["baseline"] is None else r["intensity"] - r["baseline"]
        r["retained"] = r["time"] > 0 and r["corrected"] is not None and r["corrected"] > 0
        grouped[r["peptide"], r["bio"], r["time"]].append(r)
    return grouped, noise, len(original)


def main():
    start = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    for path, expected in cfg["input_hashes"].items():
        assert sha(path) == expected, path
    for name, expected in cfg["source_hashes"].items():
        assert sha(FAMILY / name) == expected, name
    recipe = Path(cfg["recipe_directory"])
    examples = [r for r in json.loads((recipe / "examples.json").read_text()) if r["split"] == "train"]
    conditions = sorted({r["condition_id"] for r in examples})
    output = FAMILY / "results/curve-layers.jsonl.gz"
    assert not output.exists()
    records, observations, maxerror = 0, 0, 0.0
    summary = []
    with gzip.open(output, "wt") as stream:
        for condition in conditions:
            grouped, noise, expanded = condition_layers(recipe, condition)
            curves = [r for r in examples if r["condition_id"] == condition]
            rawzeros = {r["source_record_id"]: r for r in pq.read_table(
                recipe / "source_tables/raw_quantities.parquet",
                filters=[("condition_id", "=", condition), ("quantity_time_hours", "=", 0)],
            ).to_pylist()}
            for e in curves:
                for rid in e["actual_zero_raw_quantity_ids"]:
                    assert rawzeros[rid]["metadata_time_hours"] == 0
                    assert rawzeros[rid]["time_status"] == "source_agreement"
                points = []
                for o in e["observations"]:
                    values, allcontributors = [], []
                    for alias in e["peptide_aliases"]:
                        rows = grouped[alias, e["biological_replicate_raw"], o["time_hours"]]
                        surviving = [r for r in rows if r["retained"]]
                        assert rows and surviving
                        current = dict(
                            corrected=statistics.mean(r["corrected"] for r in surviving),
                            restored=statistics.mean(r["intensity"] for r in surviving),
                            unthresholded=statistics.mean(r["intensity"] for r in rows),
                            background=statistics.mean(r["baseline"] for r in surviving),
                        )
                        error = abs(current["corrected"] / o["target_value"] - 1)
                        maxerror = max(maxerror, error)
                        assert error < 1e-10, (e["curve_id"], o["time_hours"], error)
                        assert math.isclose(current["restored"] - current["background"], current["corrected"], rel_tol=1e-10, abs_tol=1e-7)
                        for prior in values:
                            assert all(math.isclose(prior[k], current[k], rel_tol=1e-10, abs_tol=1e-7) for k in current), (e["curve_id"], "alias layers differ")
                        values.append(current)
                        allcontributors.extend(rows)
                    signature = sorted({(r["technical_original"], r["technical_remapped"]) for r in allcontributors if r["retained"]})
                    # Original target bytes/numerical value remain authoritative for layer A.
                    points.append(dict(time=o["time_hours"], **{**values[0], "corrected": o["target_value"]},
                                       source_slots=o["source_slots"], contributors=allcontributors,
                                       retained_source_label_signature=signature))
                    observations += 1
                record = {k: e[k] for k in ("curve_id", "condition_id", "construct_id", "component_id", "split", "biological_replicate_raw", "peptide_IL_sequence", "peptide_sequence", "peptide_aliases", "common_positive_times", "start_0based", "end_0based", "actual_zero_raw_quantity_ids", "filtered_source_vector_ids")}
                record.update(parent_length=len(e["parent_sequence"]), global_noise=noise, points=points)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                records += 1
            summary.append(dict(condition=condition, curves=len(curves), expanded_filtered_slots=expanded, global_noise=noise))
            print(json.dumps(summary[-1]), flush=True)
    dump(FAMILY / "results/layer-replay.json", dict(training_only=True, curves=records, positive_observations=observations,
        maximum_relative_corrected_replay_error=maxerror, conditions=summary, wall_seconds=time.perf_counter()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == "__main__":
    main()
