"""Close the fixed development comparison after independent verification."""

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from verify import metric

EXP = Path(__file__).resolve().parents[1]


def main():
    out = EXP / "results"
    verification = json.loads((out / "independent-verification.json").read_text())
    assert verification["status"] == "passed"
    shutil.copyfile(__file__, EXP / "reproduce/source/report.py")
    summary = json.loads((out / "fits/summary.json").read_text())
    strata = json.loads((out / "fits/strata.json").read_text())
    checkpoint = json.loads((out / "fits/checkpoints.json").read_text())
    dump = pq.read_table(out / "fits/predictions.parquet")
    y, w = dump["target_log10_ratio"].to_numpy(), dump["weight"].to_numpy()
    conditions = dump["condition_id"].to_numpy()
    folds = dump["fold"].to_numpy()
    worst = 0.0
    for s in strata:
        population = "train" if s["kind"] == "train_condition" else "inner_cv"
        p = dump[f"{s['variant']}_{population}"].to_numpy()
        mask = folds == int(s["label"]) if s["kind"] == "fold" else conditions == s["label"]
        for name, v in metric(y[mask], p[mask], w[mask]).items():
            delta = abs(v - s[name])
            assert delta < 1e-10
            worst = max(worst, float(delta))
    variants = [
        "calibrated_precursor",
        "length",
        "position_p2_c",
        "position_p1_p2_c",
        "sequence",
        "sequence_geometry",
        "sequence_flanks",
        "sequence_context",
        "sequence_shuffled_context",
    ]
    labels = [
        "Precursor + condition",
        "Length",
        "Length + P2/C",
        "Length + P1/P2/C",
        "Original sequence",
        "+ parent geometry",
        "+ external flanks",
        "+ both",
        "+ shuffled context",
    ]
    records = []
    for variant in variants:
        cp = checkpoint[variant]["full"]
        records.append(
            {
                "variant": variant,
                "raw_features": len(cp["mean"]),
                "supported_condition_coordinates": sum(
                    p["supported"] for p in cp["coefficients"].values()
                ),
                "nonzero_coefficients": sum(p["nonzero"] for p in cp["coefficients"].values()),
                "train_rmse": summary["metrics"][variant]["train"]["rmse"],
                **{f"cv_{k}": v for k, v in summary["metrics"][variant]["inner_cv"].items()},
                "seconds": summary["metrics"][variant]["seconds"],
            }
        )
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    conds = sorted(set(conditions))
    by_cond = {(s["variant"], s["label"]): s for s in strata if s["kind"] == "inner_cv_condition"}
    gains = {
        v: [100 * (1 - by_cond[v, c]["rmse"] / by_cond["sequence", c]["rmse"]) for c in conds]
        for v in ["sequence_geometry", "sequence_flanks", "sequence_context"]
    }
    folds_better = sum(
        s["rmse"]
        < next(
            x["rmse"]
            for x in strata
            if x["variant"] == "sequence" and x["kind"] == "fold" and x["label"] == s["label"]
        )
        for s in strata
        if s["variant"] == "sequence_context" and s["kind"] == "fold"
    )
    assert folds_better == summary["folds_improved"]
    gain = (
        1
        - summary["metrics"]["sequence_context"]["inner_cv"]["rmse"]
        / summary["metrics"]["sequence"]["inner_cv"]["rmse"]
    )
    shuffled_gain = (
        1
        - summary["metrics"]["sequence_context"]["inner_cv"]["rmse"]
        / summary["metrics"]["sequence_shuffled_context"]["inner_cv"]["rmse"]
    )
    assert summary["development_gate_pass"] == (
        gain >= 0.02 and shuffled_gain >= 0.01 and folds_better >= 4
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    ax = axes[0]
    iy = np.arange(len(variants))
    ax.scatter(
        [r["train_rmse"] for r in records], iy, label="Full training", color="#8696a7", marker="x"
    )
    ax.scatter([r["cv_rmse"] for r in records], iy, label="Reused inner CV", color="#177c8e")
    ax.set_yticks(iy, labels)
    ax.invert_yaxis()
    ax.axvline(summary["metrics"]["sequence"]["inner_cv"]["rmse"], ls="--", color="#c77b30", lw=1)
    ax.set_xlabel("RMSE in log10 KO/WT elution response (lower is better)")
    ax.set_title("Parent context misses the 2% development gate")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax = axes[1]
    for i, (v, title) in enumerate(zip(gains, ["Geometry", "Flanks", "Both"], strict=True)):
        ax.scatter(gains[v], np.arange(len(conds)) + (i - 1) * 0.18, label=title)
    ax.set_yticks(
        np.arange(len(conds)), [c.removeprefix("hap1_").removesuffix("_ko").upper() for c in conds]
    )
    ax.invert_yaxis()
    ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel("CV RMSE improvement over original sequence (%)")
    ax.set_title("Condition-specific changes are descriptive")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "HAP1 training-only diagnostic · no new outer validation/test evaluation", fontsize=14
    )
    fig.savefig(out / "comparison.png", dpi=160)
    fig.savefig(out / "comparison.pdf")
    receipt = {
        "status": "passed",
        "n_strata": len(strata),
        "max_metric_error": worst,
        "context_relative_gain": gain,
        "context_gain_vs_shuffled": shuffled_gain,
        "folds_improved": folds_better,
        "development_gate_pass": summary["development_gate_pass"],
        "condition_gains_percent": dict(zip(conds, zip(*gains.values()), strict=True)),
        "argv": sys.argv,
    }
    (out / "report-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
