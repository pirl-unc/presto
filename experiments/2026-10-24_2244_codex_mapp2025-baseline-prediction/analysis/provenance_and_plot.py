"""Descriptive prior-source overlap and static result figure; no fitting or selection."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    current = json.loads((ROOT / "results/covariates.json").read_text())
    result = {}
    paths = {
        "prior_MAPP": REPO
        / "data/availability_recipes/2026-10-23-mapp-complete-run-composition-v1",
        "HAP1": REPO / "data/hap1_response_recipes/2026-10-24_0730_v1",
        "lymph_DC": REPO / "data/lymph_composition_recipes/2026-10-24_1938_v1",
    }
    for name, base in paths.items():
        manifest = json.loads((base / "manifest.json").read_text())
        if name == "prior_MAPP":
            sources = ["examples.jsonl", "parents.jsonl"]
            with (base / sources[0]).open() as f:
                old = [json.loads(line) for line in f]
            sequences = {c["sequence"] for r in old for c in r["candidates"]}
            with (base / sources[1]).open() as f:
                accessions = {json.loads(line)["reference_id"] for line in f}
        elif name == "HAP1":
            sources = ["derived/responses.parquet", "source/references.parquet"]
            sequences = set(
                pq.read_table(base / sources[0], columns=["peptide_sequence"]).column(0).to_pylist()
            )
            accessions = set(
                pq.read_table(base / sources[1], columns=["accession_raw"]).column(0).to_pylist()
            )
        else:
            sources = ["examples.json", "source_tables/references.parquet"]
            sequences = {r["sequence"] for r in json.loads((base / sources[0]).read_text())}
            accessions = set(
                pq.read_table(base / sources[1], columns=["accession_raw"]).column(0).to_pylist()
            )
        for relative in sources:
            assert sha(base / relative) == manifest["files"][relative]["sha256"]
        splits = {}
        for split in ("train", "val", "test"):
            rs = [r for r in current if r["split"] == split]
            splits[split] = {
                "n_observations_with_previous_peptide": sum(
                    r["peptide_sequence"] in sequences for r in rs
                ),
                "n_unique_peptides_previously_observed": len(
                    {r["peptide_sequence"] for r in rs} & sequences
                ),
                "n_supplied_accessions_in_previous_inventory": len(
                    {r["parent_accession"] for r in rs} & accessions
                ),
            }
        result[name] = {
            "manifest_sha256": sha(base / "manifest.json"),
            "sources": sources,
            "overlap": splits,
        }
    (ROOT / "results/prior-source-overlap.json").write_text(
        json.dumps(
            {
                "timing": "postfit_covariate_audit_no_refit_or_exclusion",
                "scope": "literal_sequence_and_accession_matches; inventories_not_verified_origins_or_teacher_lineage",
                "studies": result,
            },
            indent=2,
        )
        + "\n"
    )
    records = json.loads((ROOT / "results/metrics.json").read_text())
    lookup = {(r["split"], r["model"]): r for r in records}
    models = [
        "uniform",
        "length_ridge_0.001",
        "sequence_ridge_1",
        "exact",
        "reinforce_42",
        "reinforce_43",
        "reinforce_44",
    ]
    names = [
        "Uniform",
        "Length ridge",
        "Sequence ridge",
        "Exact KL",
        "RL seed 42",
        "RL seed 43",
        "RL seed 44",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    positions = np.arange(len(models))
    for ax, metric, title in zip(
        axes, ["rmse", "reverse_kl"], ["Centered log10 signal RMSE", "Observed-set reverse KL"]
    ):
        for j, (split, color) in enumerate([("val", "#3987A3"), ("test", "#D57940")]):
            ax.bar(
                positions + (j - 0.5) * 0.35,
                [lookup[split, m][metric] for m in models],
                width=0.35,
                color=color,
                label=split,
            )
        ax.set_xticks(positions, names, rotation=35, ha="right")
        ax.set_title(title + " (lower is better)")
        ax.set_ylim(0, None)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Baseline MAPP: modest sequence signal; no clear RL advantage\nSequence-vs-length primary test gate fails (4.33% gain; 5% required)",
        fontsize=12,
    )
    fig.savefig(ROOT / "results/heldout-comparison.png", dpi=180)


if __name__ == "__main__":
    main()
