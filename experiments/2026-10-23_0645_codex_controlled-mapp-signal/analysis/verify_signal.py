"""Post-run independent scalar recomputation from the published training dumps."""

import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
summary = json.loads((R / "signal-summary.json").read_text())
values = json.loads((R / "training-complete-case-values.json").read_text())
assert len(values) == summary["n_complete_case_peptides"] == 59
assert len({r["component_id"] for r in values}) == summary["n_parent_components"] == 40
assert all(r["split"] == "train" for r in values)
with gzip.open(R / "training-repeatability-predictions.jsonl.gz", "rt") as stream:
    predictions = [json.loads(line) for line in stream]
lookup = {
    (r["scope"], r["peptide_row_id"], r["ut_run_label"], r["epoxo_run_label"]): r
    for r in predictions
}
assert len(lookup) == len(predictions) == 2 * 9 * len(values)
totals = {
    arm: [math.fsum(r[arm][i] for r in values) for i in range(3)]
    for arm in ("ut", "epoxo")
}
assert [totals["ut"], totals["epoxo"]] == summary["complete_case_signal_sum_by_arm_run"]
assert math.isclose(
    math.fsum(totals["epoxo"]) / math.fsum(totals["ut"]),
    summary["complete_case_mean_total_signal_epoxo_over_ut"],
    rel_tol=1e-12,
)
checks = {}
for scope in ("raw_signal", "fixed_support_composition"):
    improvements, errors = defaultdict(list), defaultdict(list)
    for ui in range(3):
        for ei in range(3):
            specific, observed = {}, {}
            for row in values:
                transformed = {
                    arm: [
                        math.log10(v)
                        - (
                            math.log10(totals[arm][i])
                            if scope == "fixed_support_composition"
                            else 0
                        )
                        for i, v in enumerate(row[arm])
                    ]
                    for arm in ("ut", "epoxo")
                }
                specific[row["peptide_row_id"]] = (
                    math.fsum(v for i, v in enumerate(transformed["epoxo"]) if i != ei)
                    / 2
                    - math.fsum(v for i, v in enumerate(transformed["ut"]) if i != ui)
                    / 2
                )
                observed[row["peptide_row_id"]] = (
                    transformed["epoxo"][ei] - transformed["ut"][ui]
                )
            common = math.fsum(specific.values()) / len(values)
            for row in values:
                pid = row["peptide_row_id"]
                saved = lookup[(scope, pid, ui + 1, ei + 1)]
                assert saved["component_id"] == row["component_id"]
                for key, value in (
                    ("specific_prediction", specific[pid]),
                    ("observed_shift", observed[pid]),
                    ("common_prediction", common),
                ):
                    assert math.isclose(saved[key], value, abs_tol=1e-12)
                specific_error, common_error = (
                    (specific[pid] - observed[pid]) ** 2,
                    (common - observed[pid]) ** 2,
                )
                delta = common_error - specific_error
                assert math.isclose(saved["mse_improvement"], delta, abs_tol=1e-12)
                errors["specific"].append(specific_error)
                errors["common"].append(common_error)
                improvements[row["component_id"]].append(delta)
    actual = summary[scope]
    for kind, numbers in errors.items():
        assert math.isclose(
            actual[kind + "_pooled_rmse_log10"],
            math.sqrt(math.fsum(numbers) / len(numbers)),
            rel_tol=1e-12,
        )
    keys = sorted(improvements)
    rng = np.random.default_rng(20261023)
    bootstrap = []
    for _ in range(1000):
        selected = [keys[i] for i in rng.integers(0, len(keys), size=len(keys))]
        bootstrap.append(
            math.fsum(math.fsum(improvements[key]) for key in selected)
            / sum(len(improvements[key]) for key in selected)
        )
    saved_bootstrap = json.loads((R / (scope + "-bootstrap.json")).read_text())["draws"]
    assert np.allclose(bootstrap, saved_bootstrap, rtol=1e-12, atol=1e-12)
    assert np.allclose(
        np.quantile(bootstrap, [0.025, 0.975]),
        actual["component_bootstrap_95pct_interval"],
        rtol=1e-12,
        atol=1e-12,
    )
    checks[scope] = {
        "predictions": 9 * len(values),
        "bootstrap_draws": len(bootstrap),
        "metrics_recomputed": True,
    }
assert math.isclose(
    summary["raw_signal"]["mean_mse_improvement_log10_squared"],
    summary["fixed_support_composition"]["mean_mse_improvement_log10_squared"],
    rel_tol=1e-12,
)
result = {
    "passed": True,
    "method": "Scalar leave-run-out calculations and independent resampled per-comparison errors; no producer imports",
    "checks": checks,
    "normalization_note": "The average MSE improvement is invariant to within-run global offsets when specific predictions are centered against their common mean; composition is not independent confirmation.",
    "source_dump_sha256": hashlib.sha256(
        (R / "training-complete-case-values.json").read_bytes()
    ).hexdigest(),
}
(R / "signal-independent-verification.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, sort_keys=True))
