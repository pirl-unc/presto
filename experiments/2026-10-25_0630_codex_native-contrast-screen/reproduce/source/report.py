"""Close the frozen screen from existing predictions; no additional fitting."""

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
metrics = json.loads((R / "heldout-metrics.json").read_text())
decision = json.loads((R / "decision.json").read_text())
replay = json.loads((R / "independent-replay.json").read_text())
assert replay["passed"]
models = json.loads((R / "models.json").read_text())
selection = json.loads((R / "inner-selection.json").read_text())
fit = json.loads((R / "fit-summary.json").read_text())
records = [
    dict(split=part, condition=kind, **values)
    for part, arms in metrics.items()
    for kind, values in arms.items()
]
with (R / "heldout-metrics.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
with (R / "inner-losses.csv").open("x", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["condition", "alpha", "mean_inner_weighted_mse"])
    for kind, selected in selection.items():
        writer.writerows(
            [kind, alpha, loss] for alpha, loss in selected["mean_losses"].items()
        )

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
for axis, part in zip(axes[:2], ("val", "test"), strict=True):
    rows = pq.read_table(R / (part + "-predictions.parquet")).to_pylist()
    extent = max(abs(r["target_log10_ratio_change"]) for r in rows) * 1.08
    axis.plot(
        [-extent, extent],
        [-extent, extent],
        color="0.75",
        linestyle="--",
        linewidth=1,
        label="Perfect prediction",
    )
    axis.axhline(0, color="0.35", linewidth=0.8, label="Zero reference")
    for kind, color in [("length", "#c56e1a"), ("sequence", "#176b91")]:
        selected = [r for r in rows if r["condition"] == kind]
        axis.scatter(
            [r["target_log10_ratio_change"] for r in selected],
            [r["prediction"] for r in selected],
            s=[18 + 260 * r["weight"] for r in selected],
            alpha=0.75,
            color=color,
            label=kind.title(),
        )
    axis.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        xlabel="Observed log10 ratio change",
        ylabel="Predicted log10 ratio change",
        title=("Validation: 8 groups" if part == "val" else "Test: 18 groups"),
    )
axes[0].legend(fontsize=8, loc="upper left")
for kind, color in [("length", "#c56e1a"), ("sequence", "#176b91")]:
    values = selection[kind]["mean_losses"]
    axes[2].plot(
        [float(a) for a in values],
        list(values.values()),
        marker="o",
        color=color,
        label=kind.title(),
    )
axes[2].set(
    xscale="log",
    xlabel="Ridge penalty alpha",
    ylabel="Mean inner validation MSE",
    title="Selection uses training groups only",
)
axes[2].legend(fontsize=8)
fig.suptitle(
    "Native PSMD2 response: sequence model fails the prespecified signal gate",
    fontsize=12,
)
fig.savefig(R / "screen-summary.png", dpi=180)
fig.savefig(R / "screen-summary.svg")
plt.close(fig)

table = [
    "| Split | Model | RMSE | Weighted Pearson | Weighted Spearman | Predicted/target SD |",
    "|---|---|---:|---:|---:|---:|",
]
for record in records:
    f = lambda value: "undefined" if value is None else f"{value:.5f}"
    table.append(
        "| "
        + " | ".join(
            [
                record["split"],
                record["condition"],
                *[
                    f(record[key])
                    for key in (
                        "rmse",
                        "pearson",
                        "spearman",
                        "prediction_target_sd_ratio",
                    )
                ],
            ]
        )
        + " |"
    )
runtime = []
for stage in ("qualify", "prepare", "fit", "evaluate", "replay"):
    row = json.loads((R / (stage + "-runtime.json")).read_text())
    assert row["exit_code"] == 0 and row["resource_violation"] is None
    peak = row.get("peak_child_process_rss_bytes")
    runtime.append(
        f"| {stage} | {row['elapsed_seconds']:.3f} | {peak / 2**20:.1f}" + " |"
        if peak
        else f"| {stage} | {row['elapsed_seconds']:.3f} | Not reliably sampled |"
    )
text = f"""

## Outcome: no demonstrated transferable signal from this representation

The fixed screen failed. Validation sequence RMSE is0.317328 versus0.317216 for
zero change (0.036% worse). Test sequence RMSE is0.423308 versus0.424407 for zero,
only0.259% better. The paired test gain interval is[-0.247%,0.449%], spanning no
improvement. Predicted variation is only0.95%/0.89% of target SD on validation/test.
Both the5% gain and25% spread requirements fail. Do not advance a physical exact
versus RL comparison on the strength of this result, and do not tune this test set.

{chr(10).join(table)}

All metrics are in dimensionless log10 signal-ratio-change space. Pearson on test
is0.2745 despite the nearly zero output scale; this is not calibrated predictive
quality. Weighted Spearman is0.0904. The CSV includes MSE/loss, MAE, ordinary
correlations, signed-change accuracy and denominators, target/prediction means
and spread. Every validation/test prediction is retained in Parquet;500nM binding
metrics are inapplicable. Bootstrap intervals against length also include zero.

### What the near-constant output means here

Training-only selection chooses alpha1 for length and alpha100 for sequence.
Sequence inner validation MSE improves monotonically across the registered grid:
0.75394,0.37587,0.22623,0.19381,0.18891 as alpha increases from0.01 to100. Weakening
the penalty is already harmful in the training-group comparisons. Do not extend
the grid after viewing outer results.

The sequence model has61 raw features,59 solved coordinates and rank56; N-terminal
cysteine and C-terminal histidine have zero feature-difference RMS in training
and are explicitly omitted from the solve. All59 active coefficients are nonzero;
the maximum ridge stationarity residual is6.94e-17. The separate SVD solution
reproduces all32 coefficient sets and1,580 inner predictions. Across684 training,
validation and test predictions, the maximum discrepancy is3.30e-17. All metrics,
4,000 component-bootstrap draws and the frozen gate decision reproduce.

Thus this screen's near-constant output follows cross-validated shrinkage of this
limited representation; it is not explained by a missing gradient path or an
unfinished optimizer. It does not resolve the older IC50 model's underfitting,
prove that native processing has no sequence signal, or identify the biological
cause. The observations remain conditional on six positive source signals, with
shared acquisitions and only8 validation components.

### Runtime and reproducibility

| Stage | Wall seconds including monitor | OS peak child-process RSS, MiB |
|---|---:|---:|
{chr(10).join(runtime)}

Training-core work, including inner selection and final solve, took
{fit["fit_seconds"]["length"]:.4f}s for length and{fit["fit_seconds"]["sequence"]:.4f}s
for sequence. Local CPU, one BLAS thread; no GPU/Modal or shared-environment changes.
All fitting stages passed the unchanged20min/3GiB/1GiBdisk+32MiBstartup limits.
The original subsecond qualification's one-second-polled RSS cannot estimate a
peak; the prospective local-launch amendment adds50ms polling and OS peak
accounting before biological fitting. Environment versions are recorded after
execution, explicitly labeled as a post-run snapshot rather than a historical
startup capture; this agent performed no package modifications.

The execution-order amendment was recorded before fitting: exact merged/full-CI
code and byte-identical source bindings enabled local fitting while serialized
PyPI releases proceed. The original release-first launcher/config/prepare source
and all amendment evidence remain. Upstream publication is outstanding work;
it is not silently redefined as complete. The monitor-template registration
failure occurred before any stage and is documented separately. All five actual
model stages succeeded without retry or criterion changes.

### Next scientific decision

The next experiment should use training data only to distinguish poor sequence
representation from unstable acquisition-level response. Quantify within-arm
ratio variability and leave-one-acquisition sensitivity without calling the
numbered runs independent biological specimens. Then test a preregistered
source-context representation only within training components if justified.
The inspected outer partitions cannot become a fresh confirmation test. A new
independent biological cohort is needed for a revised model's transfer claim.
This result does not complete precursor-relative APC availability, latent-pool
recovery, concentration/observation calibration or the exact-versus-RL objective.

![Frozen screen summary](results/screen-summary.png)
"""
with (ROOT / "README.md").open("a") as stream:
    stream.write(text)
environment = dict(
    snapshot_timing="after completed stages; no package changes by this agent",
    python=platform.python_version(),
    platform=platform.platform(),
    versions={
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "pyarrow", "psutil", "matplotlib")
    },
)
(ROOT / "reproduce/environment.json").write_text(
    json.dumps(environment, indent=2) + "\n"
)
summary = dict(
    analysis="existing frozen outputs only; no further fit",
    decision=decision,
    condition_metrics=records,
    replay_verified=True,
)
(R / "report-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for name in (
    "core.py",
    "features.py",
    "qualify.py",
    "fit.py",
    "evaluate.py",
    "replay.py",
):
    assert (ROOT / "code" / name).read_bytes() == (
        ROOT / "reproduce/source" / name
    ).read_bytes()
print(
    json.dumps(
        dict(
            report_created=True,
            numerical_code_unchanged=True,
            figure_sha256=hashlib.sha256(
                (R / "screen-summary.png").read_bytes()
            ).hexdigest(),
        )
    )
)
