"""Render the completed fixed development comparison; no fitting or outer reads."""

from pathlib import Path
import csv, json, shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

P = Path(__file__).resolve().parents[1]
load = lambda name: json.loads((P / "results" / name).read_text())
m, d, v, models, prep = map(
    load,
    ["metrics.json", "decision.json", "replay.json", "models.json", "preparation.json"],
)
config = json.loads((P / "reproduce/launch.json").read_text())
assert v["passed"]
runtimes = [load(stage + "-runtime.json") for stage in config["stages"]]
assert all(r["exit_code"] == 0 and r["resource_violation"] is None for r in runtimes)
with (P / "results/metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["scope", "kind", *m["pooled"]["zero"]], lineterminator="\n"
    )
    writer.writeheader()
    for scope, scores in [
        ("pooled", m["pooled"]),
        *[(f"fold_{k}", val) for k, val in m["folds"].items()],
    ]:
        for kind, record in scores.items():
            writer.writerow(dict(scope=scope, kind=kind, **record))
rows = pq.read_table(P / "results/development-oof-predictions.parquet").to_pylist()
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
for kind, color in [("sequence", "#c87b2b"), ("boundary", "#2776a4")]:
    rr = [r for r in rows if r["kind"] == kind]
    axes[0].scatter(
        [r["target"] for r in rr],
        [r["prediction"] for r in rr],
        s=18,
        alpha=0.65,
        label=kind,
        color=color,
    )
lim = max(abs(r["target"]) for r in rows) * 1.05
axes[0].plot([-lim, lim], [-lim, lim], ls="--", color="gray", lw=1)
axes[0].set(
    xlim=(-lim, lim),
    ylim=(-lim, lim),
    xlabel="Observed log10 ratio change",
    ylabel="Development OOF prediction",
    title="Both feature models remain nearly constant",
)
axes[0].legend(fontsize=8)
for record in models:
    if record["kind"] == "boundary":
        means = record["selection_mean_mse"]
        axes[1].plot(
            [float(x) for x in means],
            list(means.values()),
            marker="o",
            label="Fold " + str(record["outer_fold"]),
        )
axes[1].set(
    xscale="log",
    xlabel="Ridge penalty alpha",
    ylabel="Mean inner-component validation MSE",
    title="Every boundary fold selects alpha 100",
)
axes[1].legend(fontsize=8)
kinds = list(m["pooled"])
axes[2].bar(
    kinds,
    [m["pooled"][k]["rmse"] for k in kinds],
    color=["#999999", "#798c50", "#c87b2b", "#2776a4"],
)
axes[2].set(
    ylim=(0, 0.48),
    ylabel="Weighted development OOF RMSE",
    title="No material gain over controls",
)
for i, k in enumerate(kinds):
    axes[2].text(
        i,
        m["pooled"][k]["rmse"] + 0.006,
        f"{m['pooled'][k]['rmse']:.4f}",
        ha="center",
        fontsize=9,
    )
fig.savefig(P / "results/boundary-context.png", dpi=160)
fig.savefig(P / "results/boundary-context.svg")
plt.close(fig)
metric_table = "\n".join(
    f"| {k} | {r['rmse']:.6f} | {r['pearson'] if r['pearson'] is None else round(r['pearson'], 4)} | {r['spearman'] if r['spearman'] is None else round(r['spearman'], 4)} | {100 * r['prediction_target_sd_ratio']:.3f}% |"
    for k, r in m["pooled"].items()
).replace("None", "undefined")
fold_table = "\n".join(
    f"| {fold} | {len(prep['folds'][fold]['development_components'])} | "
    + " | ".join(f"{scores[k]['rmse']:.6f}" for k in kinds)
    + " |"
    for fold, scores in m["folds"].items()
)
model_table = "\n".join(
    f"| {x['outer_fold']} | {x['kind']} | {x['fitted']['alpha']:g} | {x['fitted']['n_solved_coordinates']} | {x['fitted']['design_rank']} | {x['fitted']['n_nonzero_coefficients']} |"
    for x in models
    if x["kind"] != "zero"
)
interval_table = "\n".join(
    f"| {k} | {100 * g:.4f}% | [{100 * d['paired_gain_intervals'][k][0]:.4f}%, {100 * d['paired_gain_intervals'][k][1]:.4f}%] |"
    for k, g in d["gains"].items()
)
runtime_table = "\n".join(
    f"| {r['stage']} | {r['elapsed_seconds']:.3f} | {r['peak_child_process_rss_bytes'] / 2**20:.1f} |"
    for r in runtimes
)
text = f"""# Native boundary context: fixed development comparison fails

Adding two external parent residues on each side of the peptide does **not**
produce a useful predictor of the native control/PSMD2 response. Boundary RMSE
is **{m["pooled"]["boundary"]["rmse"]:.6f}**, versus peptide-only
**{m["pooled"]["sequence"]["rmse"]:.6f}** and length-only
**{m["pooled"]["length"]["rmse"]:.6f}**. Prediction SD is only
**{100 * d["prediction_target_sd_ratio"]:.3f}%** of observed SD. All four fixed
materiality gates fail. No model is promoted and no independent confirmation
priority is earned.

![Boundary comparison](results/boundary-context.png)

## Contract and reproduction

- Date:2026-10-25 UTC; agent Codex, model GPT-6 (system identity).
- Base commit `{config["git_commit"]}`; dirty with preserved unrelated
  tasks/todo.md changes and historical untracked evidence. Ten executed sources,
  original peptide-feature checksum, launch commands, environment overrides and
  the detailed plan were frozen before qualification in `reproduce/`.
- Shared input: `{config["package"]}`.
  Manifest SHA256 `{config["manifest_sha256"]}`. Input files checked against that
  manifest. No large raw dataset copied into this experiment.
- Same158original training contrasts,56complete human parent sequences and55
  dependency components. No eligibility change, new reference, acquisition
  selection, target weighting from reliability, or outlier removal. Source-wide
  dependencies precede eligibility; groups remain intact in every fold.
- PXD018722 native HEK293T control/PSMD2 knockdown, six positive raw sequence
  intensities per endpoint, PEP<=.01, no reverse/contaminants,8–25residues, literal
  nonengineered unique complete parent with all declared alternatives resolved.
  Physical biological replication and control/KD pairing remain unknown.
- Only native non-MHC raw sequence intensities are labels. LFQ, affinity, stability,
  TCR and MHC-elution families are excluded. Source zeros/missing quantities are
  not imputed; this fixed cohort consists of complete detected endpoints.
- Target is the KD-minus-control difference of three-acquisition mean log10
  peptide/reference intensity ratios within one parent. It is not absolute
  abundance, a full peptide distribution or APC calibration. Shared precursor/run
  scale and fixed MS response cancel only under the stated contrast assumptions.
- All weights retain equal component, then parent, then contrast weighting and
  normalize within each fit/evaluation subset. No intercept, identity features,
  measured intensity, acquisition label or reliability feature enters a predictor.
- No pretraining. Software qualification uses60synthetic known-context rows only;
  these do not enter biological fitting. Loss is weighted MSE plus alpha times
  coefficient squared norm; no division by rank/feature count.
- Local CPU, oneBLASthread, noModalGPU or installs. Exact overrides and stage-start
  library versions are preserved. Shared Python environment unchanged.
- Original validation/test groups remain closed. This is **training-only nested
  development CV**, not a fresh blind test. Thus no original held-out loss/test
  dump or IC50/500nMclassification applies. Every development OOF and inner
  selection prediction is retained with target/ID/weight/fold. Parquet filters
  precede row materialization; whole-file hash reads check storage integrity,
  without materializing or evaluating other splits.

Frozen invocation (execute only in a fresh registered copy; existing evidence
must not be overwritten):

```sh
reproduce/launch.sh qualify
reproduce/launch.sh prepare
reproduce/launch.sh fit
reproduce/launch.sh evaluate
reproduce/launch.sh replay
```

The original launcher contains the original absolute path; a relocated replay
must explicitly adjust its copied path and retain the original invocation.
`reproduce/report.sh` regenerates this report/plots without fitting or outer access.

## Inputs, representation and information flow

The baseline uses ln(peptide length),20amino-acid fractions,20first-residue and
20last-residue indicators. Its pure61feature function is byte-identical to
screen0630. The new model adds88features: four external positions (two preceding
start, two following end) ×20canonical amino acids plus OTHER and BOUNDARY.
OTHER is a literal noncanonical residue; BOUNDARY is outside the complete parent.

Every peptide and reference endpoint is mapped by literal sequence to all exact,
overlapping parent occurrences. Features average over occurrences; none is
selected using labels. All214endpoints in this actual cohort have one occurrence,
so averaging is not activated here. The general overlapping/unknown/terminal
behavior is exercised in software qualification. All source-origin spans and
parent hashes reconcile before fitting and independently in replay.

Each model predicts `g(peptide,parent context) - g(reference,parent context)`.
The endpoint features are subtracted, then each surviving coordinate is divided
by its training-only weighted RMS. Exactly zero-RMS coordinates are dropped;
there is no feature centering or intercept. Weighted ridge solves all remaining
coefficients. The model has no neural encoder or auxiliary branch that can remain
untrained. All original endpoint contexts,149feature names and contrast vectors
are retained in results.

## Fixed nested selection and outcomes

Five development folds hash complete dependency IDs using
`pxd018722-boundary-dev-v1`; three inner selection folds use
`pxd018722-boundary-inner-v1`. Minimum five components in every development and
inner selection fold passes. Inner support ranges7–21components. All scaling is
refit inside each inner fit. Alpha[.01,.1,1,10,100] minimizes the unweighted mean of
three weighted validation MSEs, with larger alpha winning exact ties. The selected
model is refit on that development-training partition and predicts omitted
training components once. No final all-training deployment fit is made.

| Condition | Weighted OOF RMSE | Pearson | Spearman | Prediction SD / target SD |
|---|---:|---:|---:|---:|
{metric_table}

Pearson uses weighted covariance. Spearman uses ordinary average-tie ranks
followed by weighted Pearson. Mathematical constants have undefined correlations.
Full weighted/unweighted metrics, MAE, signed-change metrics and per-component
errors are in JSON/CSV/Parquet. The boundary Spearman0.0326 is near zero; the small
pooled improvement over61features is not meaningful success.

| Development fold | Components | Zero RMSE | Length RMSE | Sequence RMSE | Boundary RMSE |
|---|---:|---:|---:|---:|---:|
{fold_table}

Boundary improves versus61features in only3of5folds. Paired component bootstrap:
2,000PCG64draws, seed202610250711, sorted55component IDs, resampling whole groups
with replacement while retaining within-group weights; linear2.5/97.5percentiles.
These intervals describe development uncertainty with overlapping training sets,
not prospective transfer confidence.

| Boundary gain versus | RMSE gain | Paired95%interval |
|---|---:|---:|
{interval_table}

Fixed promotion rule required >=5%gain over all three controls, predictionSD>=25%
of targetSD, improvement in at least4of5folds, and positive bootstrap lower bound
versus61features. **Every requirement fails.** Neither alpha grid nor folds were
changed after observing the result.

## Which parameters trained

| Fold | Representation | Selected alpha | Solved coordinates | Design rank | Nonzero coefficients |
|---|---|---:|---:|---:|---:|
{model_table}

All boundary folds choose alpha100. Across inner folds, maximal stationarity
residual is2.78e-16. SVD reconstruction agrees, so this small model's shrinkage is
not unfinished numerical optimization. The model learns a strong, known boundary
signal in the data-free synthetic qualification (RMSE0.009901), but that does not
establish learnable signal on this biological cohort. It also does not diagnose
the older full neural model's IC50 underfit.

## Verification, runtime and decision

Independent replay imports no production core. It reconstructs contexts via a
separate literal scan, checks all source spans, fits augmented ridge systems with
SVD instead of normal equations, reconstructs inner selection and OOF predictions,
uses manual average ranks, and regenerates all2,000bootstrap draws individually.
All240nonzero fits,9,480inner predictions,632OOF predictions and all decisions pass:
**{v["numerical_values_checked"]:,}numerical checks**, maximum difference
**{v["max_absolute_difference"]:.3g}**, within frozen rtol2e-9/atol2e-10.

| Stage | Wall seconds | OS child peak MiB |
|---|---:|---:|
{runtime_table}

All five stages passed on the first numerical attempt within unchanged20minute,
3GiB RSS and1GiB diskreserve+32MiBstartup limits. Process-tree polling50ms and OS
child peak are both recorded. Process monitoring required the existing authorized
sandbox escalation; no guard was disabled. This is experiment runtime evidence,
not a general CPU/GPU benchmark.

Acquisition diagnostic0711 found high complementary-response concordance, but
this does not make those responses predictable from sequence. Immediate boundary
features fail the current component-generalization screen. A useful next question
is whether the arbitrary reference peptide and sparse per-parent support distort
this training objective. That requires a separate fixed reference-invariant
comparison, not feature expansion, alpha retuning or relabeling the original test.

## Handoff

Status: closed, negative development result; no promoted predictor or RL advantage.
Next: audit endpoint/parent support and compare the same fixed features under a
reference-invariant all-pairs loss, keeping dependent pairs inside their original
components. Preserve this failed result and the original outer closures.
Open: full precursor-relative APC availability, undetected/sub-ligand products,
source-observation calibration, machinery/clearance transfer and independent
biological confirmation remain required for the overall goal.
"""
(P / "README.md").write_text(text)
shutil.copy2(Path(__file__), P / "reproduce/source/report.py")
print("Closed boundary context report, metric table and figure generated.")
