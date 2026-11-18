"""Paired AUROC differences against the previous default, with cohort scope."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    root = E / "results/analysis"
    reports = {
        cluster: json.loads((root / f"paired-bootstrap-{args.model}-{cluster}.json").read_text())
        for cluster in ("protein", "condition")
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.26, top=0.78, wspace=0.65)
    labels = {
        "expanded": "New protected cohort",
        "apc": "Earlier APC cohort",
        "broad": "Earlier broad cohort*",
        "matched": "Affinity-matched cohort",
    }
    records = []
    for ax, cls in zip(axes, ("I", "II"), strict=True):
        pops = ["expanded", "apc", *(["broad"] if cls == "I" else []), "matched"]
        for i, pop in enumerate(pops):
            cluster = "condition" if pop == "broad" else "protein"
            matches = [
                r for r in reports[cluster]
                if r["population"] == pop and r["class"] == cls
                and r["comparator"] == "previous-default" and r["metric"] == "auroc"
            ]
            assert len(matches) == 1
            row = matches[0]
            low, high = row["ci95"]
            delta = row["difference"]
            color = "#1e6594" if delta >= 0 else "#b65b25"
            ax.plot([low, high], [i, i], color=color, linewidth=2)
            ax.scatter([delta], [i], color=color, s=50, zorder=3)
            ax.annotate(f"{delta:+.3f}", (delta, i), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=10)
            records.append({"class": cls, "population": pop, "cluster": cluster, **row})
        ax.axvline(0, color="#666666", linestyle=":", linewidth=1.3)
        ax.set_yticks(range(len(pops)), [labels[p] for p in pops], fontsize=10)
        ax.set_ylim(len(pops) - 0.6, -0.8)
        ax.set_title(f"Class {cls}", fontsize=13)
        ax.set_xlabel("AUROC change versus previous default", fontsize=10)
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Presentation validation: gains and tradeoffs", fontsize=16, y=0.97)
    fig.text(0.5, 0.87, args.model, ha="center", fontsize=10)
    fig.text(0.5, 0.08,
             "95% paired bootstrap intervals: protein families; *sample/condition clusters.\n"
             "Adaptive development validation; intervals exclude selection uncertainty. TEST remains sealed.",
             ha="center", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(root / f"comparison-{args.model}.{suffix}", dpi=180)
    (root / f"comparison-{args.model}.json").write_text(json.dumps(records, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
