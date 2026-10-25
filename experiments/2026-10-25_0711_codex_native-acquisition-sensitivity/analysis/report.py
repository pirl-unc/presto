"""Render the closed training-only diagnostic without refitting or outer access."""

from pathlib import Path
import csv
import json
import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parents[1]
s = json.loads((P / "results/summary.json").read_text())
v = json.loads((P / "results/replay.json").read_text())
components = json.loads((P / "results/components.json").read_text())
config = json.loads((P / "reproduce/launch.json").read_text())
assert v["passed"]
runtime = [
    json.loads((P / "results" / f"{name}-runtime.json").read_text())
    for name in config["stages"]
]
assert all(r["exit_code"] == 0 and r["resource_violation"] is None for r in runtime)
with (P / "results/partitions.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        [
            "control",
            "kd",
            "pearson",
            "spearman",
            "sign_agreement",
            "selected_sd",
            "remaining_sd",
        ]
    )
    for r in s["partitions"]:
        writer.writerow(
            [
                r["control"],
                r["kd"],
                r["pearson"],
                r["spearman"],
                r["sign"]["same_sign_fraction"],
                r["selected"]["sd"],
                r["remaining"]["sd"],
            ]
        )
with (P / "results/omissions.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        [
            "omitted_acquisition",
            "change_rms",
            "mean_absolute_change",
            "sign_flip_fraction",
            "eligible_weight",
        ]
    )
    for r in s["omissions"]:
        writer.writerow(
            [
                r["omitted_acquisition"],
                r["change"]["rms"],
                r["mean_absolute_change"],
                r["sign"]["opposite_sign_fraction"],
                r["sign"]["eligible_weight"],
            ]
        )
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
m = np.array([r["pearson"] for r in s["partitions"]]).reshape(3, 3)
im = axes[0].imshow(m, vmin=-1, vmax=1, cmap="RdBu", aspect="equal")
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", color="white")
axes[0].set(
    xticks=range(3),
    xticklabels=["KD1", "KD2", "KD3"],
    yticks=range(3),
    yticklabels=["Cont1", "Cont2", "Cont3"],
    title="Selected vs remaining acquisitions\nWeighted Pearson (training only)",
)
fig.colorbar(im, ax=axes[0], shrink=0.8)
labels = [r["omitted_acquisition"] for r in s["omissions"]]
axes[1].barh(
    labels,
    [r["change"]["rms"] for r in s["omissions"]],
    color=["#3176a1"] * 3 + ["#aa6438"] * 3,
)
axes[1].axvline(
    s["response"]["rms"], color="black", linestyle="--", label="Original response RMS"
)
axes[1].set(
    xlabel="RMS log10 ratio change",
    title="Sensitivity to omitting one acquisition",
    xlim=(0, 0.48),
)
axes[1].legend(fontsize=8)
for key, name in [
    ("response_squared_contribution", "Response"),
    ("within_squared_contribution", "Within-arm residual"),
]:
    values = np.array(sorted([r[key] for r in components], reverse=True))
    axes[2].plot(
        np.arange(1, len(values) + 1), np.cumsum(values) / values.sum(), label=name
    )
axes[2].plot([0, 55], [0, 1], ls=":", color="gray", label="Equal contributions")
axes[2].set(
    xlabel="Components ranked separately by contribution",
    ylabel="Cumulative fraction of weighted squared value",
    title="Contribution concentration (55 groups)",
    xlim=(0, 55),
    ylim=(0, 1),
)
axes[2].legend(fontsize=8)
fig.savefig(P / "results/acquisition-sensitivity.png", dpi=160)
fig.savefig(P / "results/acquisition-sensitivity.svg")
plt.close(fig)
partition_table = "\n".join(
    f"| {r['control']} + {r['kd']} | {r['pearson']:.4f} | {r['spearman']:.4f} | {100 * r['sign']['same_sign_fraction']:.1f}% |"
    for r in s["partitions"]
)
omit_table = "\n".join(
    f"| {r['omitted_acquisition']} | {r['change']['rms']:.5f} | {r['mean_absolute_change']:.5f} | {100 * r['sign']['opposite_sign_fraction']:.2f}% |"
    for r in s["omissions"]
)
runtime_table = "\n".join(
    f"| {r['stage']} | {r['elapsed_seconds']:.3f} | {r['peak_child_process_rss_bytes'] / 2**20:.1f} |"
    for r in runtime
)
concentration = {}
for key in ("response_squared_contribution", "within_squared_contribution"):
    a = sorted([r[key] for r in components], reverse=True)
    concentration[key] = {
        "top1_fraction": a[0] / sum(a),
        "top5_fraction": sum(a[:5]) / sum(a),
    }
(P / "results/concentration.json").write_text(
    json.dumps(concentration, indent=2) + "\n"
)
text = f"""# Native acquisition sensitivity: concordant training responses, weak sequence predictor

**Result:** complementary acquisition responses are concordant on the frozen training
cohort (median weighted Pearson **{s["median_partition_pearson"]:.4f}**, range
{min(r["pearson"] for r in s["partitions"]):.4f}–{max(r["pearson"] for r in s["partitions"]):.4f}).
This weakens acquisition instability as the dominant explanation for screen0630's
nearly constant predictions. It does not establish a biological noise ceiling,
causal processing effects, transfer performance or absolute concentration.

The prespecified decision selects a **new training-only parent-boundary-context
comparison**, with no retuning of the inspected outer splits. This diagnostic
fits no model and demonstrates no improvement over uniform availability.

![Training acquisition sensitivity](results/acquisition-sensitivity.png)

## Reproducibility and contract

- Date: 2026-10-25 UTC. Agent: Codex; model identity: GPT-6 as supplied by system.
- Base commit: `{config["git_commit"]}`; dirty worktree with pre-existing tasks/todo.md
  changes and untracked historical evidence. All five executed numerical/launcher
  sources were frozen before qualification in `reproduce/source/`, with hashes,
  stage commands, configuration and preregistered plan in `reproduce/`.
- Shared input package: `{config["package"]}`.
  Manifest SHA256: `{config["manifest_sha256"]}`.
  Input file hashes checked against that manifest; no duplicate raw dataset created.
- Exactly 158 original training contrasts, 56 complete human parent sequences,
  55 dependency components. Original source-wide dependency graph, hash references,
  complete-case selection and weights unchanged. Equal component, then parent,
  then contrast weight; weights normalize to one.
- PXD018722 native HEK293T control versus PSMD2 knockdown; three acquisitions per
  arm, all six intensities positive per endpoint. Original admission: peptide PEP
  <=0.01, reverse/contaminant exclusion, 8–25 residues, literal nonengineered unique
  complete-parent sequence with every declared parent resolved and agreeing.
- Assay: non-MHC native peptidomics raw sequence intensity only. LFQ, MHC-elution,
  IC50, stability and TCR labels excluded. No censor substitution, missing-intensity
  imputation, acquisition relabeling or extra eligibility filtering.
- Target: mean KD log10(peptide/reference intensity) minus mean control, using an
  unchanged same-parent reference. This is a conditional relative response, not
  a molar concentration or the full intracellular peptide distribution. These
  HEK293T measurements are not an APC calibration dataset.
- Acquisitions are `Cont1`, `Cont2`, `Cont3`, `KD1`, `KD2`, `KD3`. Physical specimen,
  aliquot, biological replication and pairing are unresolved. No paired-sample or
  independent-biological-replicate assertion is made.
- Training/pretraining, losses, optimization and model parameters: **none**.
  Synthetic data appear only in software qualification, never biological fitting.
- No validation/test predictions or held-out loss: this is explicitly a training-only
  diagnostic after outer inspection in screen0630. No held-out examples are materialized
  by the diagnostic or replay. Parquet predicates select training IDs; whole-file
  integrity hashing reads storage bytes without evaluating other splits. See
  `results/read-scope.json`. IC50 regression and 500 nM metrics are inapplicable.
- Local CPU only; no Modal GPU. One BLAS thread; exact environment overrides in
  `reproduce/launch.json`. Versions captured at each stage start. Shared Python
  environment unchanged. Tested Hitlist2.0.38 release independently published and
  byte-verified; receipt in `results/recipe-release.json`.

Run the frozen commands from a fresh copy of this experiment directory, retaining
source/config but with a new empty results/ and launch_logs/ directory. Existing
results are evidence and must not be overwritten:

```sh
reproduce/launch.sh qualify
reproduce/launch.sh diagnose
reproduce/launch.sh replay
```

The launcher uses an absolute original path; when relocating a fresh replay,
update the copied launcher path explicitly and retain the original invocation.
Report-only regeneration: `reproduce/report.sh` (no biological fits or outer access).

## Within-arm dispersion and omission sensitivity

Weighted response RMS is **{s["response"]["rms"]:.5f}** log10 units; centered SD
{s["response"]["sd"]:.5f}. Within-arm RMS is **{s["within_arm_rms"]:.5f}**
(control {s["within_control_rms"]:.5f}, KD {s["within_kd_rms"]:.5f}), or
{s["within_to_response_rms_ratio"]:.1%} of response RMS. This is a descriptive ratio,
not an error variance or reliability coefficient. No division by sqrt(3) is used.
Per-acquisition residual means/RMS/SD are preserved in `results/summary.json`.

| Omitted acquisition | Response change RMS | Mean absolute change | Sign flips |
|---|---:|---:|---:|
{omit_table}

**{s["stable_nonzero_sign_fraction"]:.1%}** of the original weighted mass keeps the
same nonzero response sign under all six omissions. Original zero-response exclusions:
{s["zero_original_response_count"]}; all comparisons report their eligible weight and
zero exclusions. Each omission uses two runs in one arm and three in the other;
these are sensitivity views, not independent observations or new training labels.

## Complementary acquisition comparisons

One control + one KD acquisition define one response. The other two acquisitions
per arm define its complementary response. Pearson uses weighted covariance;
Spearman applies weighted Pearson to ordinary average-tie ranks. Constant arrays
have undefined correlation. These nine views are dependent and are not pooled or
bootstrapped as nine independent replicates.

| Selected acquisitions | Weighted Pearson | Weighted Spearman | Same nonzero sign |
|---|---:|---:|---:|
{partition_table}

All nine correlations are defined and positive, selecting the prespecified
representation investigation. Concordance can also reflect shared-reference,
acquisition or observation effects; it cannot establish biological replication.

## Component contributions and relation to screen0630

The largest component supplies {concentration["response_squared_contribution"]["top1_fraction"]:.1%}
of weighted squared response; the largest five supply
{concentration["response_squared_contribution"]["top5_fraction"]:.1%}. For within-arm
squared residual the corresponding fractions are
{concentration["within_squared_contribution"]["top1_fraction"]:.1%} and
{concentration["within_squared_contribution"]["top5_fraction"]:.1%}. Full 55-component
contributions are preserved in `results/components.json`; no component is discarded,
no acquisition subset is selected, and no new exclusion rule is introduced.

Screen0630's fixed 61-feature sequence model selected ridge alpha100 in training CV
and failed both outer gates, with prediction spread below 1% of target SD. The exact
solution was independently reproduced, so unfinished optimization was not the
explanation for that small model. The present result suggests investigating missing
sequence context before increasing optimizer complexity or adding RL on this cohort.
It does **not** resolve the older full-model IC50 underfit or erase the failed screen.

## Verification and runtime

Data-free qualification checks identical acquisitions, omission-residual and
complementary-response identities, duplicate-weight and sign-reversal invariance,
constant/zero conventions and both decision branches. The diagnostic code was
unchanged after qualification.

Independent scalar replay imports no production core, uses `math.fsum`, manual
average ranks and log10 of original intensity ratios, and checks all 1,896
contributors against 1,284 distinct admitted source quantity cells. Every response,
residual, omission, partition, component contribution and summary/decision replays:
**{v["scalar_values_checked"]:,} scalar checks**, maximum absolute difference
**{v["max_absolute_difference"]:.3g}**, within frozen atol2e-12/rtol2e-13.
Training diagnostics are retained per contrast in Parquet; CSV tables and JSON
summaries permit recomputation.

| Stage | Wall seconds | OS child peak MiB |
|---|---:|---:|
{runtime_table}

All stages respect 20min, 3GiB RSS, 1GiB disk reserve plus32MiB startup margin.
Process-tree RSS is polled at50ms; OS child-process peak is also recorded. The first
sandboxed preflight was denied process enumeration before a numerical child was
launched. Its environment and failure are preserved under
`reproduce/attempts/sandbox-preflight/`; the unchanged command then passed with
process-monitoring permission. No numerical stage failed or was retried.

## Handoff

Status: closed; positive training acquisition concordance, no new predictor.
Next step: preregister nested training-component CV for parent-boundary context
against the frozen 61-feature sequence reference. Keep outer validation/test closed;
a changed representation needs new independent biological confirmation.
Open questions: missing physical-replicate metadata (Hitlist#837), selection of only
complete detected peptides, source processing versus clearance, APC transfer,
MS observation calibration and precursor-relative absolute abundance remain unresolved.
"""
(P / "README.md").write_text(text)
shutil.copy2(Path(__file__), P / "reproduce/source/report.py")
print("Report, CSV tables and figures generated from closed diagnostic outputs.")
