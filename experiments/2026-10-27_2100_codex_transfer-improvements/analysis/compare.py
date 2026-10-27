"""Paired saved-prediction comparison and publication-ready development plots."""

import csv
import json

import numpy as np
from scipy.stats import rankdata

from verify import CONDITIONS, FAMILY, load, read, atomic

VARIANTS = [
    "pooled-natural-joint",
    "pooled-balanced-joint",
    "interaction-natural-joint",
    "interaction-balanced-joint",
    "interaction-I",
    "interaction-II",
]
SEEDS = [42, 43]
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 4272026


def predictions(name, split, kind, mean=False):
    directory = FAMILY / "results" / name
    own = kind in CONDITIONS[name]["classes"]
    if mean:
        filename = (
            f"mean-{split}-predictions.jsonl.gz"
            if own
            else f"mean-transfer-{split}-predictions.jsonl.gz"
        )
    else:
        filename = (
            f"{split}-predictions.jsonl.gz"
            if own
            else f"identical_weights_expanded_routing-{split}-predictions.jsonl.gz"
        )
    rows = [
        r
        for r in load(directory / filename)
        if r["qualifier"] == "=" and r["mhc_classes"] == [kind]
    ]
    return sorted(rows, key=lambda r: r["observation_id"])


def scores(y, p):
    """One row per model, exact-value scores on a shared resampled population."""
    ry = rankdata(y)
    rp = rankdata(p, axis=1)
    ry -= ry.mean()
    rp -= rp.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(ry**2) * np.sum(rp**2, axis=1))
    rho = np.divide(rp @ ry, denominator, out=np.full(p.shape[0], np.nan), where=denominator > 0)
    return rho, np.sqrt(np.mean((p - y) ** 2, axis=1))


def bootstrap(split, kind):
    names = [f"{v}-s{s}" for v in VARIANTS for s in SEEDS]
    rows = [predictions(n, split, kind) for n in names]
    reference = rows[0]
    ids = [r["observation_id"] for r in reference]
    y = np.array([r["target"] for r in reference])
    for rr in rows:
        assert [r["observation_id"] for r in rr] == ids
        np.testing.assert_array_equal([r["target"] for r in rr], y)
    p = np.array([[r["prediction"] for r in rr] for rr in rows])
    peptide_groups = {}
    for i, row in enumerate(reference):
        peptide_groups.setdefault(row["peptide"], []).append(i)
    groups = [np.array(v) for _, v in sorted(peptide_groups.items())]
    own = "interaction-I" if kind == "I" else "interaction-II"
    contrasts = [
        ("pooled-balanced-joint", "pooled-natural-joint"),
        ("interaction-natural-joint", "pooled-natural-joint"),
        ("interaction-balanced-joint", "pooled-balanced-joint"),
        ("interaction-balanced-joint", "interaction-natural-joint"),
        ("interaction-balanced-joint", own),
    ]
    rho, rmse = scores(y, p)
    rho = rho.reshape(len(VARIANTS), 2).mean(axis=1)
    rmse = rmse.reshape(len(VARIANTS), 2).mean(axis=1)
    draws = np.empty((BOOTSTRAP_DRAWS, 2, len(VARIANTS)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for i in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        a, b = scores(y[index], p[:, index])
        draws[i, 0] = a.reshape(len(VARIANTS), 2).mean(axis=1)
        draws[i, 1] = b.reshape(len(VARIANTS), 2).mean(axis=1)
    result = []
    for a, b in contrasts:
        ia, ib = VARIANTS.index(a), VARIANTS.index(b)
        for metric, axis, values in (("spearman", 0, rho), ("rmse", 1, rmse)):
            delta = draws[:, axis, ia] - draws[:, axis, ib]
            lo, hi = np.nanquantile(delta, [0.025, 0.975])
            result.append(
                dict(
                    split=split,
                    evaluation_class=kind,
                    candidate=a,
                    reference=b,
                    metric=metric,
                    delta=float(values[ia] - values[ib]),
                    ci95_low=float(lo),
                    ci95_high=float(hi),
                    peptide_clusters=len(groups),
                    exact_rows=len(y),
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED,
                )
            )
    return result


def table_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    results = FAMILY / "results"
    assert read(results / "validation-decision.json")["test_used_for_decision"] is False
    all_rows = []
    runtime = []
    baselines = []
    for name in CONDITIONS:
        v = read(results / f"{name}-independent-verification.json")
        assert v["status"] == "passed"
        all_rows.extend(v["table"])
        report = read(results / name / "report.json")
        progress = read(results / name / "progress.json")
        runtime.append(
            dict(
                condition=name,
                parameters=report["parameters"],
                selected_epoch=report["summary"]["selected_epoch"],
                fit_seconds=report["summary"]["elapsed_seconds"],
                worker_seconds=report["resources"]["wall_seconds"],
                optimizer_seconds=progress["optimizer_seconds"],
                peak_rss_bytes=report["resources"]["peak_rss"],
                peak_gpu_allocated=report["resources"]["peak_gpu_allocated"],
                peak_gpu_reserved=report["resources"]["peak_gpu_reserved"],
                hardware=report["runtime"]["actual_gpu"],
            )
        )
        for key, value in v["mean_baselines"].items():
            baselines.append(
                dict(
                    condition=name,
                    scope=key,
                    **{k: value[k] for k in ("n_all", "n", "loss", "rmse")},
                )
            )
    table_csv(results / "metrics-independent.csv", all_rows)
    table_csv(results / "runtime.csv", runtime)
    table_csv(results / "source-mean-baselines.csv", baselines)
    aggregate = []
    for variant in VARIANTS:
        for split in ("validation", "test"):
            for kind in ("I", "II"):
                rows = [
                    r
                    for r in all_rows
                    if r["condition"].rsplit("-s", 1)[0] == variant
                    and r["split"] == split
                    and r["evaluation_class"] == kind
                ]
                assert len(rows) == 2
                aggregate.append(
                    dict(
                        variant=variant,
                        split=split,
                        evaluation_class=kind,
                        scope=rows[0]["scope"],
                        n=rows[0]["n"],
                        **{
                            f"{metric}_{stat}": float(fn([r[metric] for r in rows]))
                            for metric in (
                                "spearman",
                                "pearson",
                                "rmse",
                                "loss",
                                "accuracy",
                                "balanced_accuracy",
                                "precision",
                                "recall",
                                "f1",
                                "auroc",
                                "auprc",
                            )
                            for stat, fn in (("mean", np.mean), ("min", np.min), ("max", np.max))
                        },
                    )
                )
    table_csv(results / "seed-summary.csv", aggregate)
    atomic(results / "seed-summary.json", aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    labels = [
        "Pooled / natural",
        "Pooled / balanced",
        "Interaction / natural",
        "Interaction / balanced",
        "Interaction / I only",
        "Interaction / II only",
    ]
    for ax, kind in zip(axes, ("I", "II")):
        for i, variant in enumerate(VARIANTS):
            values = [
                r["spearman"]
                for r in all_rows
                if r["condition"].rsplit("-s", 1)[0] == variant
                and r["split"] == "test"
                and r["evaluation_class"] == kind
            ]
            zero = kind not in CONDITIONS[f"{variant}-s42"]["classes"]
            ax.bar(i, np.mean(values), color="#d1903e" if zero else "#287b89", alpha=0.7)
            ax.scatter([i - 0.07, i + 0.07], values, s=24, c="#152f3d", zorder=3)
            ax.text(
                i,
                max(values) + 0.02,
                f"{np.mean(values):.3f}" + ("\nzero-shot" if zero else ""),
                ha="center",
                fontsize=9,
            )
        ax.set(title=f"MHC-{kind}", ylim=(0, 0.8), xticks=range(len(VARIANTS)), xticklabels=labels)
        ax.tick_params(axis="x", labelrotation=28)
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Test Spearman ρ")
    fig.suptitle("Joint: 2,000 updates; source-only: 1,000 · H100 · seeds 42 / 43")
    fig.tight_layout()
    fig.savefig(results / "test-comparison.png", dpi=180)
    fig.savefig(results / "test-comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    colors = dict(zip(VARIANTS, ("#2d6f98", "#ba7a22", "#22866b", "#9462a9", "#af5557", "#708443")))
    for name, spec in CONDITIONS.items():
        directory = results / name
        history = read(directory / "history.json")
        progress = [
            json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()
        ]
        variant = name.rsplit("-s", 1)[0]
        steps = np.array([p["steps"] for p in progress])
        times = np.array([p["elapsed_seconds"] for p in progress])
        for col, kind in enumerate(("I", "II")):
            points = [
                (h["steps"], h["validation"]["ic50"]["by_class"][kind]["spearman"])
                for h in history
                if kind in h["validation"]["ic50"]["by_class"]
            ]
            if not points:
                continue
            x, y = map(np.array, zip(*points))
            for row, xx in ((0, x), (1, np.interp(x, steps, times))):
                axes[row, col].plot(
                    xx,
                    y,
                    color=colors[variant],
                    linestyle="-" if spec["seed"] == 42 else "--",
                    label=variant if spec["seed"] == 42 else None,
                )
    for row in range(2):
        for col, kind in enumerate(("I", "II")):
            axes[row, col].set(
                title=f"MHC-{kind} validation",
                xlabel="Optimizer updates" if row == 0 else "Elapsed worker seconds",
                ylabel="Spearman ρ",
                ylim=(0, 0.8),
            )
            axes[row, col].grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[0, 1].get_legend_handles_labels()
    merged = dict(zip(labels + labels2, handles + handles2))
    fig.legend(merged.values(), merged.keys(), loc="lower center", ncol=3)
    fig.suptitle("Source-class validation only · solid seed 42, dashed seed 43")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(results / "validation-curves.png", dpi=180)
    plt.close(fig)

    comparisons = []
    for split in ("validation", "test"):
        for kind in ("I", "II"):
            comparisons.extend(bootstrap(split, kind))
            print(f"bootstrap {split} MHC-{kind} complete", flush=True)
    table_csv(results / "paired-bootstrap.csv", comparisons)
    atomic(
        results / "paired-bootstrap.json",
        dict(
            contract=(
                "paired peptide-cluster percentile bootstrap; seed-mean scores, not ensemble; "
                "development holdouts; no multiplicity correction"
            ),
            comparisons=comparisons,
        ),
    )


if __name__ == "__main__":
    main()
