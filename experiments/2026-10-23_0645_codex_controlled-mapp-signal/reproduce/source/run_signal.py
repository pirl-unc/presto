"""Frozen training-only repeated-acquisition diagnostic; not model evaluation."""

import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
RECIPE = Path(CFG["recipe_output"])


def save(name, value):
    (ROOT / "results" / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def correlations(pred, target):
    if np.ptp(pred) == 0 or np.ptp(target) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(pred, target)[0, 1]),
        "spearman": float(spearmanr(pred, target).statistic),
    }


def compare(logs, components, pids, name):
    # logs[peptide, arm (UT then epoxo), run]. No specimen pairing is asserted.
    folds, losses, predictions = [], [], []
    for ut in range(3):
        for epoxo in range(3):
            target = logs[:, 1, epoxo] - logs[:, 0, ut]
            specific = (logs[:, 1, :].sum(1) - logs[:, 1, epoxo]) / 2 - (
                logs[:, 0, :].sum(1) - logs[:, 0, ut]
            ) / 2
            common = float(specific.mean())
            specific_error = (specific - target) ** 2
            common_error = (common - target) ** 2
            improvement = common_error - specific_error
            losses.append(improvement)
            fold = {
                "ut_run_label": ut + 1,
                "epoxo_run_label": epoxo + 1,
                "n_peptides": len(pids),
                "common_shift_prediction": common,
                "target_mean_shift": float(target.mean()),
                "specific_rmse_log10": float(np.sqrt(specific_error.mean())),
                "common_rmse_log10": float(np.sqrt(common_error.mean())),
                "mse_improvement_log10_squared": float(improvement.mean()),
                "specific_correlations": correlations(specific, target),
                "common_correlations": {"pearson": None, "spearman": None},
            }
            folds.append(fold)
            predictions.extend(
                {
                    "scope": name,
                    "peptide_row_id": pid,
                    "component_id": components[i],
                    "ut_run_label": ut + 1,
                    "epoxo_run_label": epoxo + 1,
                    "observed_shift": float(target[i]),
                    "specific_prediction": float(specific[i]),
                    "common_prediction": common,
                    "mse_improvement": float(improvement[i]),
                }
                for i, pid in enumerate(pids)
            )
    per_peptide = np.array(losses).mean(0)
    sums, sizes = defaultdict(float), Counter()
    for group, value in zip(components, per_peptide):
        sums[group] += float(value)
        sizes[group] += 1
    keys = sorted(sums)
    numerator, denominator = (
        np.array([sums[k] for k in keys]),
        np.array([sizes[k] for k in keys]),
    )
    rng = np.random.default_rng(CFG["signal_contract"]["bootstrap_seed"])
    draws = []
    for _ in range(CFG["signal_contract"]["bootstrap_draws"]):
        indices = rng.integers(0, len(keys), size=len(keys))
        draws.append(float(numerator[indices].sum() / denominator[indices].sum()))
    interval = np.quantile(draws, [0.025, 0.975]).tolist()
    summary = {
        "n_peptides": len(pids),
        "n_components": len(keys),
        "n_leave_two_runs_out_comparisons": 9,
        "specific_pooled_rmse_log10": float(
            np.sqrt(np.mean([x["specific_rmse_log10"] ** 2 for x in folds]))
        ),
        "common_pooled_rmse_log10": float(
            np.sqrt(np.mean([x["common_rmse_log10"] ** 2 for x in folds]))
        ),
        "mean_mse_improvement_log10_squared": float(per_peptide.mean()),
        "component_bootstrap_95pct_interval": interval,
        "component_bootstrap_fraction_positive": float(np.mean(np.array(draws) > 0)),
        "n_folds_improving": sum(x["mse_improvement_log10_squared"] > 0 for x in folds),
        "residual_repeatability_gate": bool(interval[0] > 0),
        "bootstrap_scope": "Descriptive component resampling conditional on these six acquisitions; does not resample studies/specimens/runs and is not biological causal confidence.",
        "folds": folds,
    }
    save(
        name + "-bootstrap.json",
        {
            "seed": CFG["signal_contract"]["bootstrap_seed"],
            "draws": draws,
            "unit": "sequence_parent_component",
            "statistic": "peptide_weighted_mean_of_nine_fold_MSE_improvement",
        },
    )
    return summary, predictions


def main():
    started = time.perf_counter()
    assert json.loads((ROOT / "results/independent-verification.json").read_text())[
        "passed"
    ]
    frozen = json.loads((ROOT / "reproduce/signal-launch.json").read_text())
    assert (
        hashlib.sha256((RECIPE / "manifest.json").read_bytes()).hexdigest()
        == frozen["recipe_manifest_sha256"]
    )
    assert (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        == frozen["script_sha256"]
    )
    # Filter split before constructing any outcome summary or support diagnostic.
    rows = [
        r
        for r in json.loads((RECIPE / "examples.json").read_text())
        if r["split"] == "train" and r["source_dataset"] == "PXD009508"
    ]
    assert all(r["split"] == "train" for r in rows)
    alpha = [r for r in rows if r["context"]["capture_label_raw"] == "alpha6"]
    by_peptide = defaultdict(dict)
    for row in alpha:
        cell = (
            row["context"]["treatment_label_raw"],
            row["context"]["replicate_label"],
        )
        assert cell not in by_peptide[row["peptide_row_id"]]
        by_peptide[row["peptide_row_id"]][cell] = row
    selection, selected, reason_counts = [], [], Counter()
    for pid, cells in sorted(by_peptide.items()):
        reasons = []
        for arm in ("ut", "epoxo"):
            arm_rows = [r for (treatment, _), r in cells.items() if treatment == arm]
            if len(arm_rows) != 3:
                reasons.append("fewer_than_three_positive_runs_" + arm)
            if (
                sum(r["evidence_counts"]["n_direct_msms_evidence"] for r in arm_rows)
                < 1
            ):
                reasons.append("no_direct_msms_" + arm)
        reason_counts.update(reasons)
        selection.append(
            {
                "peptide_row_id": pid,
                "split": "train",
                "component_id": next(iter(cells.values()))["component_id"],
                "status": "excluded" if reasons else "included",
                "reasons": reasons,
                "n_qualified_positive_runs": len(cells),
            }
        )
        if not reasons:
            selected.append(pid)
    assert len(selected) >= 2, (
        "Insufficient complete-case support; preserve selection and do not fit."
    )
    values = np.array(
        [
            [
                [by_peptide[pid][(arm, rep)]["reported_intensity"] for rep in (1, 2, 3)]
                for arm in ("ut", "epoxo")
            ]
            for pid in selected
        ]
    )
    assert (
        values.shape == (len(selected), 2, 3)
        and np.isfinite(values).all()
        and (values > 0).all()
    )
    components = [by_peptide[pid][("ut", 1)]["component_id"] for pid in selected]
    logs = np.log10(values)
    composition_logs = logs - np.log10(values.sum(axis=0))[None, :, :]
    assert np.allclose((10**composition_logs).sum(0), 1)
    absolute, raw_predictions = compare(logs, components, selected, "raw_signal")
    composition, comp_predictions = compare(
        composition_logs, components, selected, "fixed_support_composition"
    )
    totals = defaultdict(list)
    for row in rows:
        totals[row["channel_id"]].append(row)
    run_totals = []
    selected_set = set(selected)
    for channel, items in sorted(totals.items()):
        subset = [
            r
            for r in items
            if r["peptide_row_id"] in selected_set
            and r["context"]["capture_label_raw"] == "alpha6"
        ]
        run_totals.append(
            {
                "channel_id": channel,
                "capture": items[0]["context"]["capture_label_raw"],
                "treatment": items[0]["context"]["treatment_label_raw"],
                "replicate_label": items[0]["context"]["replicate_label"],
                "n_qualified_training_positive_peptides": len(items),
                "qualified_positive_signal_sum": math.fsum(
                    r["reported_intensity"] for r in items
                ),
                "n_complete_case_peptides": len(subset),
                "complete_case_signal_sum": math.fsum(
                    r["reported_intensity"] for r in subset
                )
                if subset
                else None,
            }
        )
    raw_effect = logs[:, 1, :].mean(1) - logs[:, 0, :].mean(1)
    comp_effect = composition_logs[:, 1, :].mean(1) - composition_logs[:, 0, :].mean(1)
    sources = [
        {
            "peptide_row_id": pid,
            "component_id": components[i],
            "peptide_sequence": by_peptide[pid][("ut", 1)]["peptide_sequence"],
            "split": "train",
            "ut": values[i, 0, :].tolist(),
            "epoxo": values[i, 1, :].tolist(),
            "raw_mean_log10_shift": float(raw_effect[i]),
            "composition_mean_log10_shift": float(comp_effect[i]),
            "source_measurement_ids": [
                by_peptide[pid][(arm, rep)]["measurement_id"]
                for arm in ("ut", "epoxo")
                for rep in (1, 2, 3)
            ],
        }
        for i, pid in enumerate(selected)
    ]
    save("training-selection.json", selection)
    save("training-run-totals.json", run_totals)
    save("training-complete-case-values.json", sources)
    with gzip.open(
        ROOT / "results/training-repeatability-predictions.jsonl.gz", "wt"
    ) as stream:
        for row in raw_predictions + comp_predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "scope": "Training-component acquisition repeatability, not sequence-model generalization or causal concentration effects",
        "n_training_alpha6_peptides_with_any_qualified_positive": len(by_peptide),
        "n_complete_case_peptides": len(selected),
        "n_parent_components": len(set(components)),
        "complete_case_retention_fraction": len(selected) / len(by_peptide),
        "selection_reason_counts": dict(reason_counts),
        "mean_raw_log10_shift": float(raw_effect.mean()),
        "median_raw_log10_shift": float(np.median(raw_effect)),
        "sd_raw_peptide_log10_shift": float(raw_effect.std(ddof=1)),
        "mean_composition_log10_shift": float(comp_effect.mean()),
        "sd_composition_peptide_log10_shift": float(comp_effect.std(ddof=1)),
        "complete_case_signal_sum_by_arm_run": values.sum(0).tolist(),
        "complete_case_mean_total_signal_epoxo_over_ut": float(
            values[:, 1, :].sum(0).mean() / values[:, 0, :].sum(0).mean()
        ),
        "raw_signal": absolute,
        "fixed_support_composition": composition,
        "decision": "freeze_parsimonious_exact_sampled_fit"
        if absolute["residual_repeatability_gate"]
        else "move_to_known_substrate_kinetics_no_heldout_tuning",
        "held_out_used": False,
        "neural_training": False,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    save("signal-summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
