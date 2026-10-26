"""Descriptive reporting from frozen predictions; no additional model fitting."""

import csv
import os
from pathlib import Path
import statistics

from common import ROOT, read, request, write


def main():
    req = request()
    fitted = Path(req["fit_directory"])
    out = ROOT / "results" / req["output_subdir"]
    out.mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    predictions = read(fitted / "outer-predictions.json")
    summary = read(fitted / "summary.json")
    models = read(fitted / "models.json")
    sequences = sorted({r["sequence"] for r in predictions})
    fixed = []
    arms = ("unit", "slope", "sequence")
    labels = ("Fixed dose slope", "Learned dose slope", "Dose + sequence")
    colors = ("#8c8c8c", "#3274a1", "#c46a2b")
    for seq in sequences:
        row = {"sequence": seq}
        for arm in arms:
            records = [
                r
                for r in predictions
                if r["sequence"] == seq and r["arm"] == arm and r["dose_fmol"] == 1
            ]
            assert len(records) == 3
            row["gene"] = records[0]["gene"]
            row["observed"] = statistics.mean(r["y"] for r in records)
            row[arm] = statistics.mean(r["prediction"] for r in records)
        fixed.append(row)
    diagnostics = {"scope": "posthoc_descriptive_fixed_1fmol_no_refit_no_gate_change", "arms": {}}
    target = [r["observed"] for r in fixed]
    for arm in arms:
        values = [r[arm] for r in fixed]
        slopes = [m["dose_slope"] for m in models if m["level"] == "outer" and m["arm"] == arm]
        diagnostics["arms"][arm] = {
            "target_sd": statistics.pstdev(target),
            "prediction_sd": statistics.pstdev(values),
            "pearson": float(pearsonr(target, values).statistic),
            "spearman": float(spearmanr(target, values).statistic),
            "dose_slope_min": min(slopes),
            "dose_slope_max": max(slopes),
        }
    write(out / "fixed-dose-diagnostics.json", diagnostics)
    with (out / "fixed-dose-peptides.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fixed[0]))
        writer.writeheader()
        writer.writerows(fixed)
    metrics = [
        {
            "arm": arm,
            **{
                k: summary["arms"][arm][k]
                for k in (
                    "equal_peptide_rmse",
                    "pooled_rmse",
                    "spearman",
                    "pearson",
                    "prediction_sd",
                    "target_sd",
                )
            },
        }
        for arm in arms
    ]
    with (out / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    values = [summary["arms"][arm]["equal_peptide_rmse"] for arm in arms]
    axes[0].bar(range(3), values, color=colors, width=0.65)
    axes[0].set_xticks(range(3), ["Fixed\ndose slope", "Learned\ndose slope", "Dose +\nsequence"])
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("Equal-peptide RMSE, log10 signal")
    axes[0].set_title("No gain from sequence features")
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.018, f"{value:.3f}", ha="center")
    for arm, label, color in zip(arms[1:], labels[1:], colors[1:], strict=True):
        axes[1].scatter(target, [r[arm] for r in fixed], label=label, color=color, alpha=0.8, s=45)
    limits = (min(target) - 0.1, max(target) + 0.1)
    axes[1].plot(limits, limits, "--", color="#777777", linewidth=1, label="Perfect agreement")
    axes[1].set_xlim(limits)
    axes[1].set_ylim(limits)
    axes[1].set_xlabel("Observed mean log10 signal at 1 fmol")
    axes[1].set_ylabel("Predicted mean log10 signal at 1 fmol")
    axes[1].set_title("At fixed dose, predictions vary little")
    axes[1].legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "18 selected standards · 13 precursor-gene folds · development estimates", fontsize=13
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / ("sequence-response-comparison." + ext), dpi=180)
    plt.close(fig)
    print(diagnostics, flush=True)


if __name__ == "__main__":
    main()
