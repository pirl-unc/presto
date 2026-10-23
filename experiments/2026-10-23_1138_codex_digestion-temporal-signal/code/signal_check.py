"""Frozen training-only temporal repeatability check, no optimizer."""

import gzip
import hashlib
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    recipe = Path(config["recipe_directory"])
    assert (
        hashlib.sha256((recipe / "manifest.json").read_bytes()).hexdigest()
        == config["recipe_manifest_sha256"]
    )
    assert (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        == config["launcher_sha256"]
    )
    assert not (FAMILY / "results/metrics.json").exists()
    # Held-out data are present in the immutable export but are never evaluated.
    curves = [
        r
        for r in json.loads((recipe / "examples.json").read_text())
        if r["split"] == "train"
    ]
    pairs = defaultdict(list)
    for curve in curves:
        pairs[curve["condition_id"], curve["peptide_IL_sequence"]].append(curve)
    predictions, conditions = [], defaultdict(list)
    for group in pairs.values():
        times = group[0]["common_positive_times"]
        assert len(times) >= 3 and len(group) >= 2
        responses, signals = {}, {}
        for curve in group:
            assert curve["common_positive_times"] == times
            values = {v["time_hours"]: v["target_value"] for v in curve["observations"]}
            signals[curve["curve_id"]] = values
            responses[curve["curve_id"]] = np.log10(
                np.array([values[t] for t in times[1:]]) / values[times[0]]
            )
        for curve in group:
            others = [
                c["curve_id"] for c in group if c["curve_id"] != curve["curve_id"]
            ]
            prediction = np.mean([responses[i] for i in others], axis=0)
            target = responses[curve["curve_id"]]
            conditions[curve["construct_id"], curve["condition_id"]].append(
                {
                    "cross_replicate_mse": float(np.mean((prediction - target) ** 2)),
                    "no_change_mse": float(np.mean(target**2)),
                }
            )
            for i, t in enumerate(times[1:]):
                predictions.append(
                    dict(
                        curve_id=curve["curve_id"],
                        construct_id=curve["construct_id"],
                        condition_id=curve["condition_id"],
                        peptide_sequence=curve["peptide_sequence"],
                        biological_replicate_raw=curve["biological_replicate_raw"],
                        split="train",
                        source_other_curve_ids=others,
                        reference_time_hours=times[0],
                        time_hours=t,
                        reference_signal=signals[curve["curve_id"]][times[0]],
                        signal=signals[curve["curve_id"]][t],
                        target_log10_ratio=float(target[i]),
                        predicted_log10_ratio=float(prediction[i]),
                        no_change_prediction=0.0,
                    )
                )
    parent = defaultdict(list)
    for (cid, condition), values in sorted(conditions.items()):
        parent[cid].append(
            dict(
                condition_id=condition,
                curves=len(values),
                cross_replicate_mse=float(
                    np.mean([v["cross_replicate_mse"] for v in values])
                ),
                no_change_mse=float(np.mean([v["no_change_mse"] for v in values])),
            )
        )
    by_parent = [
        dict(
            construct_id=cid,
            conditions=values,
            cross_replicate_mse=float(
                np.mean([v["cross_replicate_mse"] for v in values])
            ),
            no_change_mse=float(np.mean([v["no_change_mse"] for v in values])),
        )
        for cid, values in sorted(parent.items())
    ]
    y = np.array([r["target_log10_ratio"] for r in predictions])
    p = np.array([r["predicted_log10_ratio"] for r in predictions])
    cross = float(np.mean([r["cross_replicate_mse"] for r in by_parent]))
    flat = float(np.mean([r["no_change_mse"] for r in by_parent]))
    improved = sum(r["cross_replicate_mse"] < r["no_change_mse"] for r in by_parent)
    result = dict(
        n_curves=len(curves),
        n_peptide_condition_pairs=len(pairs),
        n_predictions=len(predictions),
        n_parent_constructs=len(parent),
        parents_improved=improved,
        by_parent=by_parent,
        parent_weighted_cross_replicate_rmse=math.sqrt(cross),
        parent_weighted_no_change_rmse=math.sqrt(flat),
        pooled_cross_replicate_rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        pooled_no_change_rmse=float(np.sqrt(np.mean(y**2))),
        pooled_pearson=float(pearsonr(p, y).statistic),
        pooled_spearman=float(spearmanr(p, y).statistic),
        pooled_target_sd=float(y.std()),
        pooled_prediction_sd=float(p.std()),
        descriptive_gate_passed=bool(
            cross < flat and improved >= math.ceil(len(parent) / 2)
        ),
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_validation_or_test_response_evaluation=True,
        training="none; matched-label repeatability only",
    )
    with gzip.open(FAMILY / "results/predictions.jsonl.gz", "wt") as f:
        for r in predictions:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
    (FAMILY / "results/metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
