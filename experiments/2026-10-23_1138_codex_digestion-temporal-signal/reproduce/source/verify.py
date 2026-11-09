"""Independently recompute every training-only response and macro statistic."""

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    recipe = Path(config["recipe_directory"])
    curves = {
        r["curve_id"]: r
        for r in json.loads((recipe / "examples.json").read_text())
        if r["split"] == "train"
    }
    values = {
        k: {o["time_hours"]: o["target_value"] for o in v["observations"]}
        for k, v in curves.items()
    }
    groups = defaultdict(set)
    for k, c in curves.items():
        groups[c["condition_id"], c["peptide_IL_sequence"]].add(k)
    expected = {
        (k, t) for k, c in curves.items() for t in c["common_positive_times"][1:]
    }
    errors, targets, preds, seen = defaultdict(list), [], [], set()
    with gzip.open(FAMILY / "results/predictions.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            k = r["curve_id"]
            c = curves[k]
            t = r["time_hours"]
            ref = min(c["common_positive_times"])
            assert r["split"] == "train" and (k, t) not in seen
            seen.add((k, t))
            others = groups[c["condition_id"], c["peptide_IL_sequence"]] - {k}
            assert set(r["source_other_curve_ids"]) == others
            assert r["reference_time_hours"] == ref
            assert (
                r["reference_signal"] == values[k][ref] and r["signal"] == values[k][t]
            )
            for key in (
                "construct_id",
                "condition_id",
                "peptide_sequence",
                "biological_replicate_raw",
            ):
                assert r[key] == c[key]
            y = math.log10(values[k][t]) - math.log10(values[k][ref])
            p = math.fsum(
                math.log10(values[o][t]) - math.log10(values[o][ref]) for o in others
            ) / len(others)
            assert abs(y - r["target_log10_ratio"]) < 1e-12
            assert abs(p - r["predicted_log10_ratio"]) < 1e-12
            assert r["no_change_prediction"] == 0
            errors[k].append(((y - p) ** 2, y * y))
            targets.append(y)
            preds.append(p)
    assert seen == expected
    condition = defaultdict(list)
    for k, e in errors.items():
        c = curves[k]
        condition[c["construct_id"], c["condition_id"]].append(
            tuple(math.fsum(v[i] for v in e) / len(e) for i in (0, 1))
        )
    parent = defaultdict(list)
    for (cid, _), e in condition.items():
        parent[cid].append(tuple(math.fsum(v[i] for v in e) / len(e) for i in (0, 1)))
    parent = {
        cid: tuple(math.fsum(v[i] for v in e) / len(e) for i in (0, 1))
        for cid, e in parent.items()
    }
    macro = tuple(
        math.sqrt(math.fsum(v[i] for v in parent.values()) / len(parent))
        for i in (0, 1)
    )
    m = json.loads((FAMILY / "results/metrics.json").read_text())
    assert abs(macro[0] - m["parent_weighted_cross_replicate_rmse"]) < 1e-12
    assert abs(macro[1] - m["parent_weighted_no_change_rmse"]) < 1e-12
    assert sum(a < b for a, b in parent.values()) == m["parents_improved"] == 5
    assert len(curves) == m["n_curves"] and len(seen) == m["n_predictions"]
    assert (
        len(groups) == m["n_peptide_condition_pairs"]
        and len(parent) == m["n_parent_constructs"]
    )
    for row in m["by_parent"]:
        a, b = parent[row["construct_id"]]
        assert (
            abs(a - row["cross_replicate_mse"]) < 1e-12
            and abs(b - row["no_change_mse"]) < 1e-12
        )
    ym, pm = math.fsum(targets) / len(targets), math.fsum(preds) / len(preds)
    sy = math.fsum((y - ym) ** 2 for y in targets)
    sp = math.fsum((p - pm) ** 2 for p in preds)
    corr = math.fsum((y - ym) * (p - pm) for y, p in zip(targets, preds)) / math.sqrt(
        sy * sp
    )
    assert abs(corr - m["pooled_pearson"]) < 1e-12
    assert (
        abs(
            math.sqrt(
                math.fsum((y - p) ** 2 for y, p in zip(targets, preds)) / len(preds)
            )
            - m["pooled_cross_replicate_rmse"]
        )
        < 1e-12
    )
    assert (
        bool(
            macro[0] < macro[1]
            and sum(a < b for a, b in parent.values()) >= math.ceil(len(parent) / 2)
        )
        == m["descriptive_gate_passed"]
    )
    report = dict(
        passed=True,
        predictions_verified=len(seen),
        training_curves=len(curves),
        parents=len(parent),
        independent_scalar_log_ratios_and_macro_metrics=True,
        heldout_responses_evaluated=False,
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
