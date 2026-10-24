"""Close two frozen proposal families; read saved outcomes without fitting."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parent
UNIFORM = ROOT / "2026-10-23_1909_codex_boundary-importance"


def read(p):
    return json.loads(p.read_text())


def csv_write(p, rows):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def qualification(p):
    data = read(p / "results/qualification.json")
    rows = []
    for r in data:
        s = r["state"]
        row = dict(
            state_index=r["state_index"],
            rare=r["rare"],
            name=s.get("name", "ordinary"),
            eta=s["eta"],
            time=s["time"],
            target_start=s["target"][0],
            target_end=s["target"][1],
            count=r["count"],
            primary=r["primary"],
            gate_pass=r["gate_pass"],
        )
        for k in [
            "exact_log_probability",
            "mean_relative_probability_error",
            "mean_log_probability_bias",
            "mean_log_gradient_relative_error",
            "rms_log_gradient_relative_error",
            "plain_mean_log_gradient_relative_error",
            "plain_rms_log_gradient_relative_error",
            "controlled_numerator_mean_relative_error",
            "minimum_ess",
            "mean_ess",
            "seconds",
        ]:
            row[k] = r.get(k)
        rows.append(row)
    csv_write(p / "results/comparison.csv", rows)
    primary = [r for r in rows if r["primary"]]
    groups = [r for r in primary if not r["rare"]]
    summary = dict(
        ordinary_max_abs_mean_probability_relative_error=max(
            abs(r["mean_relative_probability_error"]) for r in groups
        ),
        ordinary_max_abs_mean_log_bias=max(
            abs(r["mean_log_probability_bias"]) for r in groups
        ),
        ordinary_max_mean_gradient_relative_error=max(
            r["mean_log_gradient_relative_error"] for r in groups
        ),
        ordinary_max_rms_gradient_relative_error=max(
            r["rms_log_gradient_relative_error"] for r in groups
        ),
        ordinary_min_mean_ess=min(r["mean_ess"] for r in groups),
        rare_primary=[r for r in primary if r["rare"]],
    )
    (p / "results/report-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rows, summary


def main():
    q0, s0 = qualification(UNIFORM)
    q1, s1 = qualification(FAMILY)
    actual = read(FAMILY / "results/actual-summary.json")
    verification = read(FAMILY / "results/actual-independent-verification.json")
    assert verification["all_replay_checks_pass"]
    reviews = {r["case_id"]: r for r in verification["cases"]}
    rows = []
    for case in actual["cases"]:
        review = reviews[case["case_id"]]
        for original, checked in zip(
            read(FAMILY / f"results/actual-{case['case_id']:02d}.json")["rows"],
            review["rows"],
        ):
            rows.append(
                dict(
                    case_id=case["case_id"],
                    condition=case["condition"].rsplit(":", 1)[-1],
                    eta=case["eta"],
                    n_residues=case["n_residues"],
                    count=checked["count"],
                    primary=checked["primary"],
                    reference_eligible=case["reference"]["eligible"],
                    original_worker_gate=checked["original_worker_gate"],
                    verified_gate=checked["verified_gate"],
                    max_abs_mean_probability_relative_error=max(
                        map(abs, checked["mean_relative_probability_error"])
                    ),
                    max_abs_mean_log_bias=max(
                        map(abs, checked["mean_log_probability_bias"])
                    ),
                    max_mean_absolute_log_error=max(
                        checked["mean_absolute_log_probability_error"]
                    ),
                    mean_gradient_relative_error=checked[
                        "mean_gradient_relative_error"
                    ],
                    rms_gradient_relative_error=checked["rms_gradient_relative_error"],
                    plain_mean_gradient_relative_error=original[
                        "plain_mean_gradient_relative_error"
                    ],
                    plain_rms_gradient_relative_error=original[
                        "plain_rms_gradient_relative_error"
                    ],
                    mean_ess=original["mean_ess"],
                    minimum_ess=original["minimum_ess"],
                    worker_seconds=case["seconds"],
                    peak_rss_bytes=case["peak_rss_bytes"],
                )
            )
    csv_write(FAMILY / "results/actual-comparison.csv", rows)
    primary = [r for r in rows if r["primary"]]
    agg = []
    for eta in [-1.0, 0.0, 1.0]:
        g = [r for r in primary if r["eta"] == eta]
        agg.append(
            dict(
                eta=eta,
                pass_count=sum(r["verified_gate"] for r in g),
                total=len(g),
                max_probability_relative_error=max(
                    r["max_abs_mean_probability_relative_error"] for r in g
                ),
                max_mean_absolute_log_error=max(
                    r["max_mean_absolute_log_error"] for r in g
                ),
                max_mean_gradient_error=max(
                    r["mean_gradient_relative_error"] for r in g
                ),
                max_rms_gradient_error=max(r["rms_gradient_relative_error"] for r in g),
                minimum_ess=min(r["minimum_ess"] for r in g),
            )
        )
    changed_primary = sum(
        r["original_worker_gate"] != r["verified_gate"] for r in primary
    )
    out = dict(
        by_eta=agg,
        primary_pass=sum(r["verified_gate"] for r in primary),
        primary_total=len(primary),
        primary_gate_decisions_changed=changed_primary,
        all_count_gate_decisions_changed=verification[
            "gate_decisions_changed_by_registered_mean_absolute_criterion"
        ],
        total_sampled_paths=21 * 3 * 8 * (64 + 512),
        suite_seconds=actual["seconds"],
        peak_worker_rss_bytes=max(r["peak_rss_bytes"] for r in primary),
        max_fd_error=max(
            x["error"]
            for c in verification["cases"]
            for x in c["checks"]["gradient_directions"]
        ),
    )
    (FAMILY / "results/actual-report-summary.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for q, label, color in [
        (q0, "Uniform boundary times", "#9c5a26"),
        (q1, "Truncated exponential + control", "#167c80"),
    ]:
        z = [r for r in q if r["primary"]]
        ax[0].plot(
            np.arange(48),
            [r["rms_log_gradient_relative_error"] for r in z],
            ".",
            label=label,
            color=color,
        )
    ax[0].axhline(0.25, color="gray", ls="--", lw=1)
    ax[0].set(
        xlabel="Fixed small-system state (45 ordinary + 3 rare)",
        ylabel="Relative RMS log-gradient error",
        title="Small systems; K = 16,384",
    )
    ax[0].legend(fontsize=8)
    for eta, color in [(-1, "#a95330"), (0, "#167c80"), (1, "#6955a4")]:
        g = [r for r in primary if r["eta"] == eta]
        ax[1].plot(
            np.arange(7),
            [r["max_mean_absolute_log_error"] for r in g],
            ".-",
            label=f"eta = {eta}",
            color=color,
        )
    ax[1].axhline(0.05, color="gray", ls="--", lw=1, label="Registered limit")
    ax[1].set_xticks(
        np.arange(7),
        ["Ffh12.5", "Ffh6.25", "H2A", "H2B", "IL37b", "LEDGF", "hTau"],
        rotation=40,
        ha="right",
    )
    ax[1].set(
        ylabel="Maximum mean absolute log error\nacross three target spans",
        title="Full parents; K = 512",
    )
    ax[1].legend(fontsize=8)
    fig.savefig(FAMILY / "results/comparison.png", dpi=180)
    plt.close(fig)
    rare_tables = []
    for s in [s0, s1]:
        rare_tables.append(
            table(
                [
                    "Stress case",
                    "True log probability",
                    "Mean probability relative error",
                    "Mean log bias",
                    "RMS gradient error",
                    "Mean ESS",
                    "Pass",
                ],
                [
                    [
                        r["name"],
                        f"{r['exact_log_probability']:.3f}",
                        f"{r['mean_relative_probability_error']:.5g}",
                        f"{r['mean_log_probability_bias']:.5g}",
                        f"{r['rms_log_gradient_relative_error']:.5g}",
                        f"{r['mean_ess']:.2f}",
                        r["gate_pass"],
                    ]
                    for r in s["rare_primary"]
                ],
            )
        )
    actual_table = table(
        [
            "Length effect eta",
            "Passed / 7",
            "Worst mean absolute log error",
            "Worst mean gradient error",
            "Worst RMS gradient error",
            "Minimum ESS / 512",
        ],
        [
            [
                r["eta"],
                f"{r['pass_count']}/7",
                f"{r['max_mean_absolute_log_error']:.4f}",
                f"{r['max_mean_gradient_error']:.4f}",
                f"{r['max_rms_gradient_error']:.4f}",
                f"{r['minimum_ess']:.2f}",
            ]
            for r in agg
        ],
    )
    case_table = table(
        [
            "Condition",
            "eta",
            "Pass",
            "Max mean absolute log error",
            "Mean gradient error",
            "RMS gradient error",
            "Mean ESS",
        ],
        [
            [
                r["condition"],
                r["eta"],
                r["verified_gate"],
                f"{r['max_mean_absolute_log_error']:.4f}",
                f"{r['mean_gradient_relative_error']:.4f}",
                f"{r['rms_gradient_relative_error']:.4f}",
                f"{r['mean_ess']:.1f}",
            ]
            for r in primary
        ],
    )
    (UNIFORM / "results/comparison.md").write_text(rare_tables[0] + "\n")
    (FAMILY / "results/comparison.md").write_text(
        rare_tables[1] + "\n\n" + actual_table + "\n\n" + case_table + "\n"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
