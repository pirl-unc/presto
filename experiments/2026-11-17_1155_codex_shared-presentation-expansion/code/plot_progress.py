"""Plot committed development histories; selected markers follow declared BCE."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("original", "sqrt"), default="original")
    args = parser.parse_args()
    histories = {}
    for folder in sorted((E / "results/interim").iterdir()):
        for path in folder.glob("*-n6000.json"):
            histories[path.stem] = json.loads(path.read_text())
    for path in sorted((E / "results/fits").glob("*-n6000/raw/history.json")):
        histories[path.parents[1].name] = json.loads(path.read_text())
    histories = {
        run: history
        for run, history in histories.items()
        if ("-sqrt-" in run) == (args.family == "sqrt")
    }
    anchors = json.loads((E / "results/interim-1000-s42.json").read_text())[
        "same_population_anchors"
    ]
    colors = {
        "pilot-balanced": "#777777",
        "expanded-none": "#24824b",
        "expanded-state": "#266bc0",
        "expanded-ii-transfer": "#bb7020",
        "expanded-pathway-state": "#a13a93",
    }
    labels = {
        "pilot-balanced": "Prior data",
        "expanded-none": "Broad data, no APC",
        "expanded-state": "Broad data + APC",
        "expanded-ii-transfer": "Broad II, prior I + APC",
        "expanded-pathway-state": "Broad data + APC + pathway",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3))
    fig.subplots_adjust(bottom=0.27, top=0.84, wspace=0.28)
    for ax, cls, count in zip(axes, ("I", "II"), (7470, 1566), strict=True):
        for run, h in sorted(histories.items()):
            arm, seed, _ = run.rsplit("-", 2)
            arm = arm.removesuffix("-sqrt")
            xs = [p["steps"] for p in h]
            ys = [p["validation"]["elution_observation"]["by_class"][cls]["auroc"] for p in h]
            ax.plot(
                xs,
                ys,
                color=colors[arm],
                linestyle="-" if seed == "s42" else "--",
                linewidth=2,
                marker=".",
                label=f"{labels[arm]} ({seed[1:]})",
            )
            selected = [i for i, p in enumerate(h) if p["selected"]][-1]
            ax.scatter(
                [xs[selected]],
                [ys[selected]],
                color=colors[arm],
                marker="o",
                s=85,
                facecolors="none",
                linewidths=1.7,
                zorder=5,
            )
        ax.axhline(
            anchors["previous-default"][cls],
            color="#222222",
            linestyle=":",
            linewidth=1.8,
            label="Previous default",
        )
        ax.set(
            title=f"Class {cls} · {count:,} validation examples",
            xlabel="Optimizer updates",
            ylabel="AUROC",
        )
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
    )
    title = "Square-root study sampling" if args.family == "sqrt" else "Equal-study sampling"
    fig.suptitle(f"{title}: presentation learning curves", fontsize=16)
    fig.text(
        0.5,
        0.02,
        "Circles mark current selection by equal-class/equal-study BCE. TEST remains sealed.",
        fontsize=10,
        ha="center",
    )
    out = E / f"results/interim-presentation-curves-{args.family}"
    for suffix in (".png", ".svg"):
        fig.savefig(out.with_suffix(suffix), dpi=170)
    plt.close(fig)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
