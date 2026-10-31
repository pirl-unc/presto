"""Verify completed deterministic fits while the sampled fit remains separate."""

import gzip
import json
import math
import resource
import time
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr

from independent import FAMILY, entry_forward, forward, read_groups


def main():
    tick = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    groups = read_groups(cfg)
    names = [
        "immediate_constant",
        "immediate_sequence",
        "entry_constant",
        "entry_sequence",
    ]
    checks = []
    expected = {}
    metrics = {}
    for name in names:
        fit = json.loads((FAMILY / f"results/fit-{name}.json").read_text())
        theta = np.array(fit["coefficients"])
        gradient = np.zeros_like(theta)
        data = 0.0
        collected = defaultdict(list)
        for g in groups:
            x = g["x"][:, :1] if name.endswith("constant") else g["x"]
            p, grad = (entry_forward if name.startswith("entry_") else forward)(
                theta, g, x, True
            )
            if g["split"] == "train":
                gradient += grad
                data += g["w"] @ ((p - g["y"]) ** 2)
            for source, value, w, y in zip(g["rows"], p, g["w"], g["y"]):
                collected[g["split"]].append((value, y, w))
                if name.startswith("entry_"):
                    expected[name, source[0]["curve_id"], source[1]] = (
                        source,
                        float(value),
                        w,
                        g,
                    )
        reg = 0.01 * theta.copy()
        reg[0] = 0
        if name.startswith("entry_"):
            reg[-1] = 0
        gradient += reg
        row = dict(
            variant=name,
            mse_error=abs(data - fit["data_mse"]),
            gradient_error=float(np.max(np.abs(gradient - fit["final_gradient"]))),
        )
        assert row["mse_error"] < 1e-6 and row["gradient_error"] < 1e-5, row
        checks.append(row)
        metrics[name] = {}
        for split, rows in collected.items():
            p, y, w = np.array(rows).T
            metrics[name][split] = dict(
                rmse=float(np.sqrt(w @ ((p - y) ** 2))),
                pearson=float(pearsonr(p, y).statistic),
                prediction_sd=float(p.std()),
            )
    seen = set()
    maximum = 0.0
    with gzip.open(FAMILY / "results/predictions.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            key = r["variant"], r["curve_id"], r["time"]
            assert key not in seen
            seen.add(key)
            source, p, w, g = expected[key]
            curve, t, ref, signal, reference = source
            maximum = max(maximum, abs(r["prediction"] - p))
            assert abs(r["prediction"] - p) < 1e-6
            assert (
                abs(r["target"] - (math.log10(signal) - math.log10(reference))) < 1e-12
            )
            assert (
                w == r["weight"]
                and signal == r["signal"]
                and reference == r["reference_signal"]
                and ref == r["reference_time"]
            )
            assert r["split"] == g["split"] and r["split"] != "test"
            assert abs(r["predicted_log10_signal"] - math.log10(reference) - p) < 1e-6
    assert seen == set(expected)
    result = dict(
        checks=checks,
        predictions_verified=len(seen),
        maximum_prediction_error=maximum,
        metrics=metrics,
        sampled_not_evaluated=True,
        test_not_evaluated=True,
        wall_seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        runtime_context="Read-only verification overlaps the running sampled fit; do not treat either as an isolated hardware benchmark",
    )
    (FAMILY / "results/independent-exact-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
