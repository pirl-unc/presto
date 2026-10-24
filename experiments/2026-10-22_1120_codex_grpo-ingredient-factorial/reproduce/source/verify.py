"""Recompute the factorial study from saved arrays without importing its trainer."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import logsumexp


FAMILY = Path(__file__).resolve().parents[1]
RUN = FAMILY / "results/run"
SPEC = importlib.util.spec_from_file_location(
    "reference_metrics", Path(__file__).with_name("reference_metrics.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_metrics(rows, prediction, reference, beta, recorded):
    records = [{**row, "prediction": q.tolist()} for row, q in zip(rows, prediction)]
    actual = audit.recompute(records)
    audit.compare(recorded["original"], actual)
    logp = np.log(np.asarray([row["target"] for row in rows]))
    logref = np.log(reference)
    optlog = (logp + beta * logref) / (1 + beta)
    optlog -= logsumexp(optlog, axis=1, keepdims=True)
    adjusted = [{**row, "target": p.tolist()} for row, p in zip(records, np.exp(optlog))]
    optimum = audit.recompute(adjusted)
    audit.compare(recorded["optimum"], optimum)
    q = np.asarray(prediction, dtype=np.float64)
    klref = float(np.sum(q * (np.log(q) - np.log(reference.astype(np.float64))), axis=1).mean())
    assert np.isclose(recorded["kl_reference"], klref, atol=2e-6, rtol=2e-6)
    objective = actual["reverse_kl"] + beta * klref
    assert np.isclose(recorded["objective"], objective, atol=2e-6, rtol=2e-6)
    return actual, optimum


def main():
    summary = load(RUN / "summary.json")
    assert not summary["failures"] and len(summary["conditions"]) == 69
    expected_hashes = load(RUN / "artifacts.json")
    for name, digest in expected_hashes.items():
        assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == digest, name
    launch = load(FAMILY / "reproduce/launch.json")
    execution = load(RUN / "execution.json")
    assert execution["sources"] == launch["source_sha256"]
    for name, digest in execution["sources"].items():
        assert (
            hashlib.sha256((FAMILY / "reproduce/source" / name).read_bytes()).hexdigest() == digest
        )
    plan = FAMILY.parents[1] / launch["plan"]
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == launch["plan_sha256"]
    dataset = audit.read_rows(RUN / "dataset.jsonl")
    assert len(dataset) == len({row["id"] for row in dataset}) == 576
    teacher = load(RUN / "teacher.json")
    parents = {}
    for row in dataset:
        audit.reconstruct(row, teacher)
        parents.setdefault(tuple(row["tokens"]), set()).add(row["split"])
    assert len(parents) == 144 and all(len(values) == 1 for values in parents.values())
    groups = {
        split: [row for row in dataset if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    configs = {row["name"]: row for row in launch["conditions"]}
    references = {seed: np.load(RUN / f"reference-seed{seed}.npz") for seed in (42, 43, 44)}
    by_condition, initial, table, checked = {}, {}, [], 0
    for report in summary["conditions"]:
        config, seed = report["config"], report["seed"]
        assert config == configs[config["name"]]
        name = f"{config['name']}-seed{seed}"
        assert name not in by_condition
        by_condition[name] = report
        initial.setdefault(seed, set()).add(report["initial_hash"])
        assert all(report["gradient_seen"].values())
        assert all(value > 0 for value in report["parameter_max_change"].values())
        assert report["updates"] == config["steps"]
        assert report["collected_batches"] == config["steps"] // config["reuse"]
        assert report["fit_dense_span_scores"] == config["steps"] * 16 * 70
        if config["method"] in ("factorial", "original"):
            assert report["fit_target_terms"] == report["collected_batches"] * 16 * 8
        elif config["method"] == "exact":
            assert report["fit_target_terms"] == report["fit_dense_span_scores"]
        else:
            queries = report["fit_target_terms"]
            draws = report["collected_batches"] * 16 * config["samples"]
            assert draws <= queries <= 5 * draws
        if report["diagnostics_by_reuse_step"]:
            assert report["diagnostics_by_reuse_step"]["0"]["active_clip_fraction"] == 0
            assert report["diagnostics_by_reuse_step"]["0"]["ratio_min"] == 1
            assert report["diagnostics_by_reuse_step"]["0"]["ratio_max"] == 1
        history = load(RUN / name / "history.json")
        assert report["selected_step"] == min(history, key=lambda row: row["objective"])["step"]
        saved = np.load(RUN / name / "predictions.npz")
        assert saved.files == ["train", "validation", "test"]
        for split, rows in groups.items():
            prediction = saved[split]
            assert prediction.shape == (len(rows), 70)
            recorded = report["metrics"][split]
            original, optimum = check_metrics(
                rows, prediction, references[seed][split], config["beta"], recorded
            )
            checked += 1
            for context in range(4):
                mask = np.asarray([row["context_id"] == context for row in rows])
                check_metrics(
                    [row for row in rows if row["context_id"] == context],
                    prediction[mask],
                    references[seed][split][mask],
                    config["beta"],
                    recorded["contexts"][str(context)],
                )
                checked += 1
            table.append(
                {
                    "condition": config["name"],
                    "seed": seed,
                    "split": split,
                    "beta": config["beta"],
                    "reuse": config["reuse"],
                    "budget": config["budget"],
                    "selected_step": report["selected_step"],
                    "fit_seconds": report["seconds_including_validation"],
                    "target_terms": report["fit_target_terms"],
                    "dense_span_scores": report["fit_dense_span_scores"],
                    "objective": recorded["objective"],
                    **{"original_" + key: value for key, value in original.items()},
                    **{"optimum_" + key: value for key, value in optimum.items()},
                }
            )
        screen = (
            report["metrics"]["validation"]["optimum"]["tv"] <= 0.1
            and report["metrics"]["validation"]["optimum"]["noncompatible_mass_mae"] <= 0.05
        )
        assert screen == report["screen"]
    assert all(len(values) == 1 for values in initial.values())
    assert set(by_condition) == {f"{name}-seed{seed}" for name in configs for seed in (42, 43, 44)}
    pairs = []
    for std in (0, 1):
        for beta in ("0", "0.1"):
            for seed in (42, 43, 44):
                first = np.load(RUN / f"factorial-c0-s{std}-r1-b{beta}-seed{seed}/predictions.npz")
                second = np.load(RUN / f"factorial-c1-s{std}-r1-b{beta}-seed{seed}/predictions.npz")
                difference = max(
                    float(np.abs(first[split] - second[split]).max()) for split in groups
                )
                assert difference < 5e-6
                pairs.append(
                    {
                        "std": std,
                        "beta": beta,
                        "seed": seed,
                        "maximum_prediction_difference": difference,
                    }
                )
    with (FAMILY / "results/metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    aggregate = []
    for name, config in configs.items():
        reports = [by_condition[f"{name}-seed{seed}"] for seed in (42, 43, 44)]
        aggregate.append(
            {
                "condition": name,
                "config": config,
                "validation_tv_optimum": float(
                    np.mean([r["metrics"]["validation"]["optimum"]["tv"] for r in reports])
                ),
                "test_tv_optimum": float(
                    np.mean([r["metrics"]["test"]["optimum"]["tv"] for r in reports])
                ),
                "test_tv_original": float(
                    np.mean([r["metrics"]["test"]["original"]["tv"] for r in reports])
                ),
                "test_tv_original_seeds": [r["metrics"]["test"]["original"]["tv"] for r in reports],
                "fit_seconds": float(np.mean([r["seconds_including_validation"] for r in reports])),
                "target_terms": float(np.mean([r["fit_target_terms"] for r in reports])),
                "dense_span_scores": reports[0]["fit_dense_span_scores"],
                "passed_seeds": sum(r["screen"] for r in reports),
            }
        )
    save(FAMILY / "results/condition-means.json", aggregate)
    save(
        FAMILY / "results/independent-verification.json",
        {
            "status": "passed",
            "conditions": 69,
            "teacher_rows": len(dataset),
            "split_and_context_metric_sets": checked,
            "original_hashes": len(expected_hashes),
            "paired_first_update_clipping_checks": pairs,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reference_metrics_sha256": hashlib.sha256(
                Path(__file__).with_name("reference_metrics.py").read_bytes()
            ).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    )
    for name in (
        "summary.json",
        "execution.json",
        "gradient-probe.json",
        "teacher.json",
        "artifacts.json",
    ):
        shutil.copyfile(RUN / name, FAMILY / "results" / name)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    subset = [
        r for r in aggregate if r["condition"].startswith("factorial") and r["config"]["beta"] == 0
    ]
    for i, row in enumerate(subset):
        axes[0].scatter(
            [i] * 3,
            row["test_tv_original_seeds"],
            color="#17629d" if row["config"]["reuse"] == 1 else "#d58319",
            alpha=0.8,
        )
    axes[0].set(
        xticks=range(len(subset)),
        xticklabels=[r["condition"].removeprefix("factorial-").removesuffix("-b0") for r in subset],
        ylabel="Test total variation",
        title="Factorial: beta zero, equal updates",
    )
    axes[0].tick_params(axis="x", rotation=60)
    subset = [
        r
        for r in aggregate
        if r["condition"]
        in (
            "exact-b0",
            "original-reinforce8",
            "hybrid-starts2",
            "hybrid-starts8",
            "query-matched-c0-r4",
            "query-matched-c1-r4",
        )
    ]
    for i, row in enumerate(subset):
        axes[1].scatter(row["target_terms"], row["test_tv_original"], s=40)
        axes[1].annotate(
            row["condition"],
            (row["target_terms"], row["test_tv_original"]),
            xytext=(2, 4 if i % 2 else -12),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set(
        xscale="log",
        xlabel="Logical fitting target terms",
        ylabel="Mean test total variation",
        title="Hybrid and reuse work comparison",
    )
    subset = [r for r in aggregate if r["config"]["beta"] == 0.1]
    for row in subset:
        axes[2].scatter(row["test_tv_optimum"], row["test_tv_original"], color="#249a73", s=30)
    axes[2].set(
        xlabel="TV to regularized optimum",
        ylabel="TV to original target",
        title="Reference KL changes the target",
    )
    fig.savefig(FAMILY / "results/comparison.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                "status": "passed",
                "fits": 69,
                "metric_sets": checked,
                "passed_screens": sum(row["passed_seeds"] for row in aggregate),
            }
        )
    )


if __name__ == "__main__":
    main()
