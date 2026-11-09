"""Save all development predictions and metrics from frozen final fits."""

import csv
import gzip
import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch

from common import (
    FAMILY,
    designs,
    dump,
    entry_predict,
    load_data,
    metrics,
    prepare_points,
)


def signature(row, group, variant):
    a, b = row["start"], row["end"]
    seq = group["sequence"]
    if variant.endswith("constant"):
        content = (b - a - 1, int(a > 0) + int(b < len(seq)))
    else:
        inside = tuple(sorted(Counter(seq[i : i + 2] for i in range(a, b - 1)).items()))
        boundaries = tuple(
            sorted(
                ([seq[a - 1 : a + 1]] if a else [])
                + ([seq[b - 1 : b + 1]] if b < len(seq) else [])
            )
        )
        content = (inside, boundaries, group["context"])
    return variant, content, row["time"], row["reference_time"]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config, groups = load_data()
    groups = [prepare_points(g) for g in groups]
    names = [
        "immediate_constant",
        "immediate_sequence",
        "entry_constant",
        "entry_sequence",
        "entry_sampled",
    ]
    assert not (FAMILY / "results/all-predictions.jsonl.gz").exists()
    fits = {
        v: json.loads((FAMILY / f"results/fit-{v}.json").read_text()) for v in names
    }
    rows = []
    classes = defaultdict(list)
    for v, fit in fits.items():
        entry = v.startswith("entry_")
        kind = "constant" if v.endswith("constant") else "sequence_context"
        design, _, _ = designs(groups, kind)
        theta = torch.tensor(fit["coefficients"], dtype=torch.float64)
        with torch.no_grad():
            for g, x in zip(groups, design):
                prediction = entry_predict(
                    x @ (theta[:-1] if entry else theta),
                    theta[-1] if entry else None,
                    g,
                    32,
                )
                for source, p, w in zip(
                    g["rows"], prediction.tolist(), g["weight"].tolist()
                ):
                    row = dict(
                        source,
                        variant=v,
                        prediction=p,
                        weight=w,
                        predicted_log10_signal=math.log10(source["reference_signal"])
                        + p,
                        predicted_signal=source["reference_signal"] * 10**p,
                    )
                    rows.append(row)
                    classes[signature(row, g, v)].append(row)
    max_shift = 0.0
    for values in classes.values():
        value = math.fsum(r["prediction"] for r in values) / len(values)
        for r in values:
            max_shift = max(max_shift, abs(value - r["prediction"]))
            r["canonical_prediction"] = value
    assert max_shift < 1e-10
    raw = {
        v: {
            s: metrics([r for r in rows if r["variant"] == v and r["split"] == s])
            for s in ("train", "val")
        }
        for v in names
    }
    canonical = {
        v: {
            s: metrics(
                [
                    dict(r, prediction=r["canonical_prediction"])
                    for r in rows
                    if r["variant"] == v and r["split"] == s
                ]
            )
            for s in ("train", "val")
        }
        for v in names
    }
    dump(FAMILY / "results/all-metrics-raw.json", raw)
    dump(FAMILY / "results/all-metrics.json", canonical)
    dump(
        FAMILY / "results/rank-ties.json",
        dict(
            policy="Group mathematically identical outputs by internal ordered-pair counts, boundary pairs, context, target/reference time; constant model uses length and boundary count. No value rounding.",
            maximum_shift=max_shift,
        ),
    )
    with gzip.open(FAMILY / "results/all-predictions.jsonl.gz", "wt") as stream:
        for r in rows:
            stream.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
    table = [
        dict(
            variant=v,
            split=s,
            **{
                k: m[k]
                for k in (
                    "rmse",
                    "pearson",
                    "spearman",
                    "prediction_sd",
                    "target_sd",
                    "n_predictions",
                    "n_parents",
                )
            },
        )
        for v in names
        for s, m in canonical[v].items()
    ]
    with (FAMILY / "results/comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = dict(
        parameters={v: len(f["coefficients"]) for v, f in fits.items()},
        entry_rates_per_hour={
            v: math.exp(f["coefficients"][-1])
            for v, f in fits.items()
            if v.startswith("entry_")
        },
        fit_seconds={v: f["fit_wall_seconds"] for v, f in fits.items()},
        all_allocated_parameters_have_gradients={
            v: all(f["gradient_coverage"]) for v, f in fits.items()
        },
        final_gradient_norm={
            v: float(np.linalg.norm(f["final_gradient"])) for v, f in fits.items()
        },
        test_not_evaluated=True,
        promotion=False,
        reason="Development-only; no independent test and no absolute-concentration calibration",
        prediction_count=len(rows),
    )
    dump(FAMILY / "results/summary.json", summary)
    print(json.dumps(dict(summary=summary, metrics=table)), flush=True)


if __name__ == "__main__":
    main()
