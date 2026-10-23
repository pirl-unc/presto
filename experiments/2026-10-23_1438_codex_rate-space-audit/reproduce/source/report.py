"""Descriptive paired diagnostics and figures; no fitting or held-out access."""

import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direct_rates import FAMILY, LN10, dump, load_blocks


def main():
    _, blocks, _ = load_blocks()
    saved = json.loads((FAMILY / "results/direct-fits.json").read_text())
    audit = json.loads((FAMILY / "results/audit.json").read_text())
    rows, initial_mass, final_mass, weights = [], [], [], []
    with gzip.open(FAMILY / "results/paired-reference-masses.jsonl.gz", "wt") as stream:
        for block, fit, before in zip(blocks, saved["fits"], audit["blocks"]):
            old = block.reference_log_mass(block.initial) / LN10
            new = block.reference_log_mass(np.asarray(fit["rates"])) / LN10
            initial_mass.extend(old)
            final_mass.extend(new)
            weights.extend(block.g["w"])
            rows.append(
                dict(
                    condition=fit["condition"],
                    parameters=fit["parameter_count"],
                    previous_rmse=before["condition_rmse"],
                    direct_rmse=fit["condition_rmse"],
                    nfev=fit["nfev"],
                    status=fit["status"],
                    active_bounds=fit["active_lower_bound_count"],
                    near_zero_rates=fit["rate_below_1e_10_count"],
                    projected_gradient_mapping=fit[
                        "projected_mse_gradient_mapping_max"
                    ],
                    seconds=fit["seconds"],
                )
            )
            for source, a, b, w in zip(block.g["rows"], old, new, block.g["w"]):
                stream.write(
                    json.dumps(
                        dict(
                            condition_id=fit["condition"],
                            curve_id=source[0]["curve_id"],
                            time=source[1],
                            reference_time=source[2],
                            weight=w,
                            previous_log10_reference_amount=float(a),
                            direct_log10_reference_amount=float(b),
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
    with (FAMILY / "results/comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    initial_mass = np.asarray(initial_mass)
    final_mass = np.asarray(final_mass)
    weights = np.asarray(weights)
    assert abs(weights.sum() - 1) < 1e-12
    summary = dict(
        previous_fraction_below_1e_20=float(np.mean(initial_mass < -20)),
        final_fraction_below_1e_20=float(np.mean(final_mass < -20)),
        previous_weight_below_1e_20=float(weights[initial_mass < -20].sum()),
        final_weight_below_1e_20=float(weights[final_mass < -20].sum()),
        median_log10_mass_change=float(np.median(final_mass - initial_mass)),
        active_bounds=sum(r["active_bounds"] for r in rows),
        near_zero_rates=sum(r["near_zero_rates"] for r in rows),
        nfev=sum(r["nfev"] for r in rows),
        relative_mse_reduction=(audit["global_mse"] - saved["global_mse"])
        / audit["global_mse"],
    )
    dump(FAMILY / "results/paired-mass-summary.json", summary)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    names = [
        "Shared rates\nregularized",
        "Shared rates\nno penalty",
        "Free rates\nlog coordinates",
        "Free rates\ndirect coordinates",
    ]
    values = [0.902565672826, 0.883147115, audit["global_rmse"], saved["global_rmse"]]
    axes[0].bar(range(4), values, color=["#8f9aa6", "#668bb3", "#cc9944", "#398b78"])
    axes[0].set_xticks(range(4), names, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Training log10-ratio RMSE")
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.025, f"{v:.6f}", ha="center", fontsize=9)
    axes[0].set_title("Negligible solver improvement")
    old_rates = np.concatenate([b.initial for b in blocks])
    new_rates = np.concatenate([np.asarray(r["rates"]) for r in saved["fits"]])
    for value, label, color in (
        (old_rates, "Previous log-rate fit", "#cc9944"),
        (new_rates, "Direct-rate fit", "#398b78"),
    ):
        axes[1].plot(
            np.sort(np.log10(value)),
            np.arange(1, len(value) + 1) / len(value),
            label=label,
            color=color,
        )
    axes[1].set_xlabel("log10(rate per hour)")
    axes[1].set_ylabel("Fraction of rate groups")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Rates approach zero boundary")
    for value, label, color in (
        (initial_mass, "Previous log-rate fit", "#cc9944"),
        (final_mass, "Direct-rate fit", "#398b78"),
    ):
        order = np.argsort(value)
        axes[2].plot(value[order], weights[order].cumsum(), label=label, color=color)
    axes[2].axvline(-20, color="#777777", linestyle=":", linewidth=1)
    axes[2].set_xlabel("log10(reference copies per parent)")
    axes[2].set_ylabel("Cumulative training weight")
    axes[2].set_title("Ratios leave abundance weakly constrained")
    fig.suptitle(
        "Training-only diagnostic: 6 substrates, 7 conditions; no biological model promotion",
        fontsize=12,
    )
    fig.savefig(FAMILY / "results/comparison.png", dpi=160)
    fig.savefig(FAMILY / "results/comparison.pdf")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
