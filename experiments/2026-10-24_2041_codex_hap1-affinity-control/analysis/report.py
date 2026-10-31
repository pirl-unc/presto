"""Verify all strata and report the frozen MHC-score development comparison."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from verify import metric

EXP = Path(__file__).resolve().parents[1]
VARIANTS = [
    "calibrated_precursor",
    "sequence",
    "length_affinity",
    "sequence_affinity",
    "sequence_shuffled_affinity",
    "sequence_affinity_geometry",
]
LABELS = [
    "Precursor + condition",
    "Original sequence",
    "Length + MHC scores",
    "Sequence + MHC scores",
    "Sequence + shuffled scores",
    "Sequence + scores + geometry",
]


def main():
    assert (
        json.loads((EXP / "results/independent-verification.json").read_text())["status"]
        == "passed"
    )
    shutil.copyfile(__file__, EXP / "reproduce/source/report.py")
    out = EXP / "results"
    summary = json.loads((out / "fits/summary.json").read_text())
    strata = json.loads((out / "fits/strata.json").read_text())
    checkpoints = json.loads((out / "fits/checkpoints.json").read_text())
    dump = pq.read_table(out / "fits/predictions.parquet")
    y, w = dump["target_log10_ratio"].to_numpy(), dump["weight"].to_numpy()
    conditions, folds, seen = (
        dump["condition_id"].to_numpy(),
        dump["fold"].to_numpy(),
        dump["frozen_predictor_training_exposure"].to_numpy(),
    )
    worst = 0.0
    for row in strata:
        population = "train" if row["kind"].startswith("train_") else "inner_cv"
        p = dump[f"{row['variant']}_{population}"].to_numpy()
        if row["kind"] == "fold":
            mask = folds == int(row["label"])
        elif row["kind"].endswith("_exposure"):
            mask = seen == (row["label"] == "True")
        else:
            mask = conditions == row["label"]
        assert int(mask.sum()) == row["n_observations"]
        for name, value in metric(y[mask], p[mask], w[mask]).items():
            err = abs(value - row[name])
            assert err < 1e-10, (row["variant"], row["kind"], row["label"], name, err)
            worst = max(worst, float(err))
    records = []
    for variant in VARIANTS:
        cp = checkpoints[variant]["full"]
        records.append(
            {
                "variant": variant,
                "raw_features": len(cp["mean"]),
                "supported_condition_coordinates": sum(
                    c["supported"] for c in cp["coefficients"].values()
                ),
                "nonzero_coefficients": sum(c["nonzero"] for c in cp["coefficients"].values()),
                "train_rmse": summary["metrics"][variant]["train"]["rmse"],
                **{f"cv_{k}": v for k, v in summary["metrics"][variant]["inner_cv"].items()},
                "seconds": summary["metrics"][variant]["seconds"],
            }
        )
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    rmse = lambda v: summary["metrics"][v]["inner_cv"]["rmse"]
    gain = 1 - rmse("sequence_affinity") / rmse("sequence")
    shuffle_gain = 1 - rmse("sequence_affinity") / rmse("sequence_shuffled_affinity")
    by_fold = {(r["variant"], r["label"]): r["rmse"] for r in strata if r["kind"] == "fold"}
    improved = sum(
        by_fold["sequence_affinity", str(k)] < by_fold["sequence", str(k)] for k in range(5)
    )
    assert summary["development_gate_pass"] == (
        gain >= 0.02 and shuffle_gain >= 0.01 and improved >= 4
    )
    by_exposure = {
        (r["variant"], r["label"]): r for r in strata if r["kind"] == "inner_cv_exposure"
    }
    by_cond = {(r["variant"], r["label"]): r for r in strata if r["kind"] == "inner_cv_condition"}
    condition_gains = {
        c: 100 * (1 - by_cond["sequence_affinity", c]["rmse"] / by_cond["sequence", c]["rmse"])
        for c in sorted(set(conditions))
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    ax = axes[0]
    ax.scatter(
        [r["train_rmse"] for r in records],
        np.arange(6),
        marker="x",
        color="#8b98a7",
        label="Full training",
    )
    ax.scatter(
        [r["cv_rmse"] for r in records], np.arange(6), color="#167c8e", label="Reused inner CV"
    )
    ax.set_yticks(np.arange(6), LABELS)
    ax.invert_yaxis()
    ax.axvline(rmse("sequence"), color="#c78339", ls="--", lw=1)
    ax.set_xlabel("RMSE in log10 KO/WT elution response")
    ax.set_title("Frozen scores add 2.12% CV improvement")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax = axes[1]
    exposure_gains = {}
    for i, (v, label) in enumerate(
        [("sequence_affinity", "MHC scores"), ("sequence_affinity_geometry", "Scores + geometry")]
    ):
        values = [
            100 * (1 - by_exposure[v, k]["rmse"] / by_exposure["sequence", k]["rmse"])
            for k in ["True", "False"]
        ]
        exposure_gains[v] = values
        ax.scatter(values, np.arange(2) + (i - 0.5) * 0.15, label=label, s=55)
    ax.set_yticks(
        [0, 1],
        [
            "Present in bundled training table\n4,925 peptides / 70,963 response cells",
            "Not found in bundled training table\n1,392 peptides / 16,297 response cells",
        ],
    )
    ax.set_ylim(-0.55, 1.55)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.2)
    ax.set_xlabel("CV RMSE improvement over original sequence (%)")
    ax.set_title("Both exposure strata improve\nFull sample lineage remains unresolved")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "HAP1 development control · frozen MHCflurry features · no new outer-test evaluation",
        fontsize=13,
    )
    fig.savefig(out / "comparison.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "comparison.pdf", bbox_inches="tight")
    receipt = {
        "status": "passed",
        "n_strata": len(strata),
        "max_metric_error": worst,
        "development_gate_pass": summary["development_gate_pass"],
        "relative_gain": gain,
        "gain_vs_shuffled": shuffle_gain,
        "folds_improved": improved,
        "condition_gains_percent": condition_gains,
        "exposure_gains_percent": exposure_gains,
    }
    (out / "report-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
