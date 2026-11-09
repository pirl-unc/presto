"""Descriptive closure only: tables, plots, overlap and prediction variation."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def main():
    result = FAMILY / "results"
    summary = read(result / "evaluation/summary.json")
    assert read(result / "independent-prediction-verification.json")["status"] == "passed"
    fields = [
        "variant",
        "split",
        "data_loss",
        "exact_rmse",
        "exact_mae",
        "exact_r2",
        "exact_pearson_pooled",
        "exact_spearman_pooled",
        "exact_direction_accuracy",
        "prediction_sd_pooled",
        "exact_target_sd_pooled",
    ]
    with (result / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in summary["metrics"])
    variants = [r["variant"] for r in summary["metrics"] if r["split"] == "test"]
    names = [
        "No change",
        "Condition only",
        "Precursor only",
        "Condition + precursor",
        "Uniform processing + precursor",
        "Sequence processing + precursor",
        "Sampled sequence + precursor",
        "Sequence without precursor",
        "Frozen native prior + precursor",
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.6), layout="constrained")
    y = np.arange(len(variants))
    for split, offset, color in (("val", -0.18, "#3A789B"), ("test", 0.18, "#E39C45")):
        scores = [
            next(
                r["exact_rmse"]
                for r in summary["metrics"]
                if r["variant"] == v and r["split"] == split
            )
            for v in variants
        ]
        ax.barh(
            y + offset,
            scores,
            height=0.34,
            label="Validation" if split == "val" else "Test",
            color=color,
        )
        for yi, value in zip(y + offset, scores):
            ax.text(value + 0.001, yi, f"{value:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.19)
    ax.set_xlabel("Weighted RMSE of log10(treatment/control); lower is better")
    ax.set_title(
        "Macrophage MHC-I response: processing does not beat context\nPXD057588 — fixed cohort and nine predeclared models",
        loc="left",
    )
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(result / "comparison.png", dpi=170)
    fig.savefig(result / "comparison.pdf")
    plt.close(fig)
    spread = []
    for variant in variants:
        with gzip.open(result / "evaluation" / (variant + "-predictions.jsonl.gz"), "rt") as stream:
            rows = [json.loads(line) for line in stream]
        for split in ("train", "val", "test"):
            for condition in ("CT2A", "GL261"):
                chosen = [r for r in rows if r["split"] == split and r["condition_id"] == condition]
                w = np.array([r["weight"] for r in chosen])
                w /= w.sum()
                entry = {"variant": variant, "split": split, "condition": condition}
                for field, key in (
                    ("prediction_log10_ratio", "prediction"),
                    ("target_log10_ratio", "target"),
                ):
                    a = np.array([r[field] for r in chosen])
                    entry[key + "_weighted_sd"] = float(np.sqrt(w @ (a - w @ a) ** 2))
                    entry[key + "_range"] = float(np.ptp(a))
                spread.append(entry)
    (result / "within-condition-spread.json").write_text(json.dumps(spread, indent=2) + "\n")
    cfg = read(FAMILY / "reproduce/launch.json")
    prior_cfg = read(Path(cfg["prior_family"]) / "reproduce/launch.json")
    recipe = Path(prior_cfg["recipe_directory"])
    native = [
        g
        for g in read(recipe / "examples.json.gz")
        if g["split"] == "train" and g["full_parent_model_supported"]
    ]
    peptides = {g["peptide_sequence"] for g in native}
    parent_hashes = {
        o["reference_sequence_sha256"]
        for o in read(recipe / "occurrences.json.gz")
        if o["peptide_sequence"] in peptides
    }
    cohort = read(result / "cohort.json")
    groups = read(Path(cohort["model_input_directory"]) / "groups.json.gz")
    overlaps = []
    for split in ("train", "val", "test"):
        selected = [g for g in groups if g["split"] == split]
        apc_peptides = {g["peptide_sequence"] for g in selected}
        apc_parents = {o["parent_sequence_sha256"] for g in selected for o in g["origins"]}
        overlaps.append(
            {
                "split": split,
                "n_unique_peptides": len(apc_peptides),
                "n_exact_peptide_overlaps_with_prior_training": len(apc_peptides & peptides),
                "n_parent_sequences": len(apc_parents),
                "n_exact_parent_overlaps_with_prior_training": len(apc_parents & parent_hashes),
            }
        )
    (result / "prior-training-overlap.json").write_text(
        json.dumps(
            {
                "scope": "exact sequence overlap only; no homology-independent claim or posthoc filtering",
                "prior_recipe_manifest_sha256": hashlib.sha256(
                    (recipe / "manifest.json").read_bytes()
                ).hexdigest(),
                "splits": overlaps,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"tables_written": True, "plots_written": True, "prior_overlap": overlaps}))


if __name__ == "__main__":
    main()
