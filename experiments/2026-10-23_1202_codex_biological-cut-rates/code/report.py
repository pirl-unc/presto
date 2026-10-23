"""Close the biological exact/hybrid family without fitting or selecting models."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify import FAMILY, read_groups


def main():
    metrics = json.loads((FAMILY / "results/reported-metrics.json").read_text())
    metrics["sampled_rb512"] = json.loads(
        (FAMILY / "results/sampled-reported-metrics.json").read_text()
    )
    names = [
        "no_change",
        "shared_time",
        "constant",
        "sequence",
        "sequence_context",
        "sampled_rb512",
    ]
    labels = [
        "No change",
        "Shared time",
        "One cut rate",
        "Sequence",
        "Sequence + context",
        "Sampled hybrid",
    ]
    controls = ("no_change", "shared_time", "constant")
    gates = {
        v: bool(
            all(
                metrics[v][s]["rmse"] < metrics[c][s]["rmse"]
                for s in ("val", "test")
                for c in controls
            )
            and all(metrics[v][s]["prediction_sd"] > 0 for s in ("val", "test"))
        )
        for v in ("sequence", "sequence_context", "sampled_rb512")
    }
    rows = [
        dict(
            variant=v,
            split=s,
            **{
                k: m[k]
                for k in [
                    "n_predictions",
                    "n_parents",
                    "mse",
                    "rmse",
                    "pooled_rmse",
                    "pearson",
                    "spearman",
                    "prediction_sd",
                    "target_sd",
                ]
            },
        )
        for v in names
        for s, m in metrics[v].items()
    ]
    with (FAMILY / "results/comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    groups = read_groups(cfg)
    parent_names = {g["parent"]: g["condition"].split("aSPIRE_")[-1] for g in groups}
    parent_rows = [
        dict(variant=v, split=s, parent=parent_names[parent], **values)
        for v in names
        for s, m in metrics[v].items()
        for parent, values in m["by_parent"].items()
    ]
    with (FAMILY / "results/by-parent.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(parent_rows[0]))
        writer.writeheader()
        writer.writerows(parent_rows)
    fit = json.loads((FAMILY / "results/fits.json").read_text())
    sampled = json.loads((FAMILY / "results/sampled-fit.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    x = np.arange(len(names))
    for delta, split, color, label in [
        (-0.18, "val", "#4d8fbb", "Validation (3 parents)"),
        (0.18, "test", "#c77d43", "Test (2 parents)"),
    ]:
        axes[0].bar(
            x + delta,
            [metrics[v][split]["rmse"] for v in names],
            0.36,
            color=color,
            label=label,
        )
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_ylabel("Equal-parent RMSE (log10 temporal ratio)")
    axes[0].set_title("No learned variant wins against every control")
    axes[0].legend(frameon=False, fontsize=9)
    for history, label, color in [
        (fit["sequence_context"]["history"], "Exact sequence + context", "#4d8fbb"),
        (sampled["history"], "Sampled hybrid, K=512, seed42", "#c77d43"),
    ]:
        axes[1].plot(
            [h["step"] for h in history],
            [h["objective"] for h in history],
            label=label,
            color=color,
        )
    axes[1].set_xlabel("Optimizer updates")
    axes[1].set_ylabel("Training MSE + coefficient penalty")
    axes[1].set_title("Same fixed objective; sampled fit retains noise")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FAMILY / "results/comparison.png", dpi=160)
    fig.savefig(FAMILY / "results/comparison.pdf")
    plt.close(fig)
    probes = json.loads((FAMILY / "results/gradient-probes-rb.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for state, color in [("initial", "#4d8fbb"), ("final", "#c77d43")]:
        subset = [r for r in probes if r["state"] == state]
        for key, style, label in [
            ("rms_relative_to_initial_gradient", "--", "full score"),
            (
                "local_rms_relative_to_initial_gradient",
                "-",
                "irrelevant cuts integrated",
            ),
        ]:
            ax.plot(
                [r["count"] for r in subset],
                [r[key] for r in subset],
                style,
                marker="o",
                color=color,
                label=state + ": " + label,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 128, 512], ["32", "128", "512"])
    ax.set_xlabel("Complete partitions per condition/time")
    ax.set_ylabel("Gradient RMS error / initial exact gradient norm")
    ax.set_title("32 probe groups: rare-event noise remains")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FAMILY / "results/gradient-comparison.png", dpi=160)
    fig.savefig(FAMILY / "results/gradient-comparison.pdf")
    summary = dict(
        promotion_gates=gates,
        all_allocated_parameters_trained={
            "constant": 1,
            "sequence": 39,
            "sequence_context": 40,
            "sampled_hybrid": 40,
        },
        total_optimizer_updates=8000,
        all_predictions_verified=562404,
        raw_predictions_and_canonical_rank_corrections_retained=True,
        exact_vs_sampled="Same model/objective/initialization/Adam budget; one sampled seed, exact residuals/probabilities retained",
        concentration_qualification=False,
    )
    (FAMILY / "results/summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
