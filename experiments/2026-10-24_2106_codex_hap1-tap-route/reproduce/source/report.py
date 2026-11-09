"""Export the fixed biological comparison and resource summaries."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ORDER = [
    "sequence_scores",
    "expanded_ridge",
    "route_only_exact",
    "route_scores_exact",
    "route_scores_sampled",
]
LABELS = [
    "Sequence + MHC scores",
    "Expanded feature ridge",
    "Shared route alone",
    "Shared route + scores · exact",
    "Shared route + scores · sampled",
]


def main():
    verified = json.loads((EXP / "results/independent-verification.json").read_text())
    assert verified["status"] == "passed"
    summary = json.loads((EXP / "results/fits/summary.json").read_text())
    strata = json.loads((EXP / "results/fits/strata.json").read_text())
    predictions = pd.read_parquet(EXP / "results/fits/predictions.parquet")
    records = []
    for variant in ORDER:
        m = summary["metrics"][variant]
        records.append(
            {
                "variant": variant,
                "train_rmse": m["train"]["rmse"],
                **{f"cv_{k}": v for k, v in m["inner_cv"].items()},
                "seconds_full_plus_folds": m["seconds"],
            }
        )
    with (EXP / "results/comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    counts = {
        "observations": len(predictions),
        "peptides": predictions.peptide_sequence.nunique(),
        "components": predictions.component_id.nunique(),
        "exposure": {},
    }
    for exposed in [False, True]:
        selected = predictions[predictions.frozen_predictor_training_exposure == exposed]
        counts["exposure"][str(exposed)] = {
            "observations": len(selected),
            "peptides": selected.peptide_sequence.nunique(),
        }
    fit_index = json.loads((EXP / "results/fits/fit-index.json").read_text())
    audit = {
        "counts": counts,
        "sampled_conditional_route_draws": sum(
            r["sampled_route_draws"] for r in fit_index["route_scores_sampled"].values()
        ),
        "sampled_to_exact_runtime": records[4]["seconds_full_plus_folds"]
        / records[3]["seconds_full_plus_folds"],
        "expanded_gain_over_sequence_scores": 1 - records[1]["cv_rmse"] / records[0]["cv_rmse"],
        "all_route_fits_below_1_percent_last_50_step_improvement": all(
            not r["still_improving_over_1_percent"]
            for v, fs in fit_index.items()
            if v.startswith("route")
            for r in fs.values()
        ),
        "source_scope": "training-only positive HAP1 TAP1/TAP2 KO/WT elution responses",
    }
    (EXP / "results/report-summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    yy = np.arange(5)
    axes[0].scatter(
        [r["train_rmse"] for r in records], yy, marker="x", color="#8998a6", label="Training"
    )
    axes[0].scatter([r["cv_rmse"] for r in records], yy, color="#167c8e", label="Reused inner CV")
    axes[0].set_yticks(yy, LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE · log10 KO/WT elution response")
    axes[0].set_title("Shared-route hypothesis misses the matched-feature control")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    axes[0].grid(axis="x", alpha=0.18)
    lookup = {
        (s["variant"], s["label"]): s["rmse"] for s in strata if s["kind"] == "inner_cv_condition"
    }
    for i, (variant, label) in enumerate(zip(ORDER[1:], LABELS[1:], strict=True)):
        gains = [
            100 * (1 - lookup[variant, str(k)] / lookup["sequence_scores", str(k)])
            for k in range(2)
        ]
        axes[1].plot(gains, [i - 0.13, i + 0.13], linestyle="none", marker="o", color="#167c8e")
        axes[1].scatter(gains[1], i + 0.13, color="#df8b28", zorder=3)
    axes[1].scatter([], [], color="#167c8e", label="TAP1")
    axes[1].scatter([], [], color="#df8b28", label="TAP2")
    axes[1].set_yticks(np.arange(4), LABELS[1:])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#999999", linewidth=1)
    axes[1].set_xlabel("CV RMSE improvement over original score regression (%)")
    axes[1].set_title("Condition-specific differences remain")
    axes[1].legend(loc="lower left")
    axes[1].grid(axis="x", alpha=0.18)
    fig.suptitle("HAP1 shared TAP-route test · development only · no concentration validation")
    fig.savefig(EXP / "results/comparison.png", dpi=170, bbox_inches="tight")
    fig.savefig(EXP / "results/comparison.pdf", bbox_inches="tight")
    shutil.copyfile(__file__, EXP / "reproduce/source/report.py")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
