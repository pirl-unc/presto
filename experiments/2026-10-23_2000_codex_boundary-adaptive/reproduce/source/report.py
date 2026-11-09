"""Report the frozen adaptive proposal versus published fixed references."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
OLD = FAMILY.parent / "2026-10-23_1920_codex_boundary-exponential"


def read(p):
    return json.loads(p.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def actual_rows(family):
    summary = read(family / "results/actual-summary.json")
    verified = read(family / "results/actual-independent-verification.json")
    assert summary["complete"] and verified["all_replay_checks_pass"]
    review = {r["case_id"]: r for r in verified["cases"]}
    result = []
    for c in summary["cases"]:
        primary = next(
            (r for r in review[c["case_id"]].get("rows", []) if r["primary"]), None
        )
        r = dict(
            case_id=c["case_id"],
            condition=c["condition"].rsplit(":", 1)[-1],
            eta=c["eta"],
            status=c["status"],
            gate_pass=None,
            max_absolute_mean_probability_error=None,
            max_mean_absolute_log_error=None,
            mean_gradient_relative_error=None,
            rms_gradient_relative_error=None,
            mean_ess=None,
            minimum_ess=None,
            worker_seconds=c.get("seconds"),
            conditional_sampling_seconds=c.get("conditional_sampling_seconds"),
            reference_load_seconds=c.get("reference", {}).get("seconds"),
            peak_rss_bytes=c.get("peak_rss_bytes"),
        )
        if primary:
            raw = next(r for r in c["primary"] if r["primary"])
            r.update(
                gate_pass=primary["verified_gate"],
                max_absolute_mean_probability_error=max(
                    map(abs, primary["mean_relative_probability_error"])
                ),
                max_mean_absolute_log_error=max(
                    primary["mean_absolute_log_probability_error"]
                ),
                mean_gradient_relative_error=primary["mean_gradient_relative_error"],
                rms_gradient_relative_error=primary["rms_gradient_relative_error"],
                mean_ess=raw["mean_ess"],
                minimum_ess=raw["minimum_ess"],
            )
        result.append(r)
    return result


def main():
    assert read(FAMILY / "results/independent-verification.json")["all_checks_pass"]
    numerical = read(FAMILY / "results/qualification.json")
    nrows = []
    for r in numerical:
        s = r["state"]
        nrows.append(
            dict(
                rare=r["rare"],
                state_index=r["state_index"],
                name=s.get("name", "ordinary"),
                eta=s["eta"],
                time=s["time"],
                target_start=s["target"][0],
                target_end=s["target"][1],
                count=r["count"],
                primary=r["primary"],
                gate_pass=r["gate_pass"],
                mean_probability_relative_error=r["mean_relative_probability_error"],
                mean_log_bias=r["mean_log_probability_bias"],
                mean_gradient_relative_error=r["mean_log_gradient_relative_error"],
                rms_gradient_relative_error=r["rms_log_gradient_relative_error"],
                plain_rms_gradient_relative_error=r[
                    "plain_rms_log_gradient_relative_error"
                ],
                mean_ess=r["mean_ess"],
                exact_log_probability=r["exact_log_probability"],
            )
        )
    write_csv(FAMILY / "results/numerical-comparison.csv", nrows)
    current = actual_rows(FAMILY)
    previous = actual_rows(OLD)
    write_csv(FAMILY / "results/actual-comparison.csv", current)
    by_eta = []
    for eta in [-1.0, 0.0, 1.0]:
        c = [r for r in current if r["eta"] == eta]
        o = [r for r in previous if r["eta"] == eta]
        complete = [r for r in c if r["status"] == "completed"]
        by_eta.append(
            dict(
                eta=eta,
                previous_pass=sum(r["gate_pass"] is True for r in o),
                adaptive_pass=sum(r["gate_pass"] is True for r in c),
                total=len(c),
                completed=len(complete),
                max_mean_absolute_log_error=max(
                    r["max_mean_absolute_log_error"] for r in complete
                ),
                max_mean_gradient_error=max(
                    r["mean_gradient_relative_error"] for r in complete
                ),
                max_rms_gradient_error=max(
                    r["rms_gradient_relative_error"] for r in complete
                ),
                minimum_ess=min(r["minimum_ess"] for r in complete),
            )
        )
    primary = [r for r in nrows if r["primary"]]
    ordinary = [r for r in primary if not r["rare"]]
    report = dict(
        by_eta=by_eta,
        primary_pass=sum(r["gate_pass"] is True for r in current),
        primary_total=len(current),
        completed=sum(r["status"] == "completed" for r in current),
        max_ordinary_mean_probability_relative_error=max(
            abs(r["mean_probability_relative_error"]) for r in ordinary
        ),
        max_ordinary_abs_mean_log_bias=max(abs(r["mean_log_bias"]) for r in ordinary),
        max_ordinary_mean_gradient_error=max(
            r["mean_gradient_relative_error"] for r in ordinary
        ),
        max_ordinary_rms_gradient_error=max(
            r["rms_gradient_relative_error"] for r in ordinary
        ),
        rare_primary=[r for r in primary if r["rare"]],
        suite_seconds=read(FAMILY / "results/actual-summary.json")["seconds"],
        sum_conditional_sampling_seconds=sum(
            r["conditional_sampling_seconds"] or 0 for r in current
        ),
        peak_worker_rss_bytes=max(r["peak_rss_bytes"] or 0 for r in current),
        reference_policy="Previously verified exact references reused by hash; old worker time included construction, no direct speedup claim",
    )
    (FAMILY / "results/report-summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    rows = [
        [
            r["eta"],
            f"{r['previous_pass']}/7",
            f"{r['adaptive_pass']}/7",
            f"{r['max_mean_absolute_log_error']:.5f}",
            f"{r['max_mean_gradient_error']:.5f}",
            f"{r['max_rms_gradient_error']:.5f}",
            f"{r['minimum_ess']:.2f}",
        ]
        for r in by_eta
    ]
    head = table(
        [
            "eta",
            "Initial-rate proposal passes",
            "Adaptive passes",
            "Worst mean absolute log error",
            "Worst mean gradient error",
            "Worst RMS gradient error",
            "Minimum ESS / 512",
        ],
        rows,
    )
    detail = table(
        [
            "Case",
            "eta",
            "Status",
            "Pass",
            "Max mean absolute log error",
            "RMS gradient error",
        ],
        [
            [
                r["condition"],
                r["eta"],
                r["status"],
                r["gate_pass"],
                r["max_mean_absolute_log_error"],
                r["rms_gradient_relative_error"],
            ]
            for r in current
        ],
    )
    (FAMILY / "results/comparison.md").write_text(head + "\n\n" + detail + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True, layout="constrained")
    for ax, eta in zip(axes, [-1.0, 0.0, 1.0]):
        for data, label, color in [
            (previous, "Initial-rate proposal", "#a55c36"),
            (current, "Adaptive proposal", "#197e86"),
        ]:
            rows = [r for r in data if r["eta"] == eta]
            values = [
                r["max_mean_absolute_log_error"]
                if r["max_mean_absolute_log_error"] is not None
                else np.nan
                for r in rows
            ]
            ax.plot(range(7), values, ".-", label=label, color=color)
        ax.axhline(0.05, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(
            range(7),
            ["Ffh12.5", "Ffh6.25", "H2A", "H2B", "IL37b", "LEDGF", "hTau"],
            rotation=45,
            ha="right",
        )
        ax.set_title(f"eta = {eta:g}; K = 512")
    axes[0].set_ylabel("Maximum mean absolute log error\nacross three target spans")
    axes[1].legend(fontsize=8)
    fig.savefig(FAMILY / "results/comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
