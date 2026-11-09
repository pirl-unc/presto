"""Render the closed objective comparison; no new fits or outer access."""

from pathlib import Path
import csv, json, shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

P = Path(__file__).resolve().parents[1]
load = lambda name: json.loads((P / "results" / name).read_text())
support, metrics, decision, replay, fit = map(
    load,
    [
        "support.json",
        "metrics.json",
        "decision.json",
        "replay.json",
        "fit-summary.json",
    ],
)
config = json.loads((P / "reproduce/launch.json").read_text())
assert replay["passed"]
runtimes = [load(s + "-runtime.json") for s in config["stages"]]
assert all(x["exit_code"] == 0 and x["resource_violation"] is None for x in runtimes)
with (P / "results/metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "evaluation_space",
            "scope",
            "condition",
            *metrics["all_pairs"]["pooled"]["all_pairs:zero"],
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for space, data in metrics.items():
        for scope, records in [
            ("pooled", data["pooled"]),
            *[(f"fold_{key}", val) for key, val in data["folds"].items()],
        ]:
            for condition, record in records.items():
                writer.writerow(
                    dict(
                        evaluation_space=space,
                        scope=scope,
                        condition=condition,
                        **record,
                    )
                )
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
kinds = ["zero", "length", "sequence", "boundary"]
x = np.arange(4)
for objective, shift, color in [
    ("reference", -0.18, "#a47b41"),
    ("all_pairs", 0.18, "#287da1"),
]:
    values = [
        metrics["all_pairs"]["pooled"][objective + ":" + kind]["rmse"] for kind in kinds
    ]
    axes[0].bar(x + shift, values, width=0.35, label=objective + " loss", color=color)
axes[0].set(
    xticks=x,
    xticklabels=kinds,
    ylim=(0, 0.53),
    ylabel="Weighted RMSE on all pairs",
    title="Same evaluation space, no useful gain",
)
axes[0].legend(fontsize=8)
rows = pq.read_table(P / "results/development-oof-predictions.parquet").to_pylist()
r = [
    r
    for r in rows
    if r["training_objective"] == r["evaluation_space"] == "all_pairs"
    and r["kind"] == "boundary"
]
limit = max(abs(row["target"]) for row in r) * 1.05
axes[1].scatter(
    [row["target"] for row in r],
    [row["prediction"] for row in r],
    s=12,
    alpha=0.65,
    color="#287da1",
)
axes[1].plot([-limit, limit], [-limit, limit], ls="--", lw=1, color="gray")
axes[1].set(
    xlim=(-limit, limit),
    ylim=(-limit, limit),
    xlabel="Observed pair response",
    ylabel="Development OOF prediction",
    title="All-pair boundary outputs remain narrow",
)
labels = ["2 endpoints", "3 endpoints", "4+ endpoints"]
mass = [
    support["strata"][key]["component_parent_weight"] for key in ("n2", "n3", "n4plus")
]
axes[2].bar(labels, mass, color=["#999999", "#b4aa73", "#527d56"])
axes[2].set(
    ylim=(0, 0.6),
    ylabel="Fraction of component/parent weight",
    title="Sparse support limits the objective contrast",
)
for i, v in enumerate(mass):
    axes[2].text(i, v + 0.012, f"{100 * v:.1f}%", ha="center")
fig.savefig(P / "results/reference-invariant.png", dpi=160)
fig.savefig(P / "results/reference-invariant.svg")
plt.close(fig)


def table(space):
    lines = []
    for key, value in metrics[space]["pooled"].items():
        objective, kind = key.split(":")
        corr = "undefined" if value["pearson"] is None else f"{value['pearson']:.4f}"
        rank = "undefined" if value["spearman"] is None else f"{value['spearman']:.4f}"
        lines.append(
            f"| {objective} | {kind} | {value['rmse']:.6f} | {corr} | {rank} | {100 * value['prediction_target_sd_ratio']:.3f}% |"
        )
    return "\n".join(lines)


support_table = "\n".join(
    f"| {key} | {v['n_parents']} | {v['n_endpoints']} | {v['n_pairs']} | {100 * v['component_parent_weight']:.2f}% |"
    for key, v in support["strata"].items()
)
interval_table = "\n".join(
    f"| {key} | {100 * g:.4f}% | [{100 * decision['intervals'][key][0]:.4f}%, {100 * decision['intervals'][key][1]:.4f}%] |"
    for key, g in decision["gains"].items()
)
runtime_table = "\n".join(
    f"| {x['stage']} | {x['elapsed_seconds']:.3f} | {x['peak_child_process_rss_bytes'] / 2**20:.1f} |"
    for x in runtimes
)
models = load("models.json")
model_table = "\n".join(
    f"| {r['outer_fold']} | {r['kind']} | {r['fitted']['alpha']:g} | {r['fitted']['n_solved_coordinates']} | {r['fitted']['design_rank']} |"
    for r in models
    if r["kind"] != "zero"
)
text = f"""# Native reference-invariant objective: negative result, stopping rule triggered

The all-pairs loss does not rescue sequence-based response prediction. On the
**same all-pair evaluation space**, the new boundary model scores RMSE
**{metrics["all_pairs"]["pooled"]["all_pairs:boundary"]["rmse"]:.6f}**, versus
**{metrics["all_pairs"]["pooled"]["reference:boundary"]["rmse"]:.6f}** for the frozen
reference-trained boundary model and **{metrics["all_pairs"]["pooled"]["all_pairs:zero"]["rmse"]:.6f}**
for zero change. Prediction SD is only **{100 * decision["prediction_target_sd_ratio"]:.3f}%**
of observed SD. All four preregistered gates fail.

**Stop further feature/optimizer expansion on this native response-only cohort.**
The next scientific step is native baseline product amounts and the MS/precursor
observation contract. This stops one unsuccessful development path; it does not
complete or pause the broader APC availability goal. The cohort can remain a
constraint on a later joint model. No predictor or RL advantage is promoted.

![Matched objective comparison](results/reference-invariant.png)

## Reproducibility and original data

- Date: 2026-10-25 UTC. Agent: Codex; model GPT-6 (system identity).
- Base commit: `{config["git_commit"]}`; dirty with unrelated tasks/todo.md edits
  and untracked historical evidence preserved. Thirteen executed sources,
  configuration, exact stage commands and prospective plan were frozen before
  qualification in `reproduce/`. Original fitting/selection/metric functions are
  AST-identical to0726; original feature/context modules are byte-identical.
- Shared package: `{config["package"]}`; manifestSHA256
  `{config["manifest_sha256"]}`. Raw inputs remain once underdata/.
- Frozen controls and endpoint features are reused from `{config["reference_family"]}`.
  Exact per-file hashes are in `reproduce/launch.json` and checked before every
  stage. The20original final controls are not refit; their632original development
  predictions reproduce with **maximum difference0.0**.
- Fixed nativePXD018722 HEK293T control/PSMD2 panel: same158training reference
  contrasts,214endpoints,56complete human parents and55dependency components.
  PEP<=.01, no reverse/contaminants,8–25residues, literal nonengineered unique
  complete parent with all declared alternatives resolved; all six endpoint
  intensities positive. No added source peptides, changed reference, acquisition
  choice, imputation, censor substitution or outlier/reliability filtering.
- Labels use raw native non-MHC sequence intensities only. LFQ, affinity/stability,
  MHC-elution andTCRlabels are excluded. Target is the KD-minus-control mean
  log10 intensity ratio between two peptides in one parent. It is a relative
  response, not molar abundance, complete-pool availability or APC calibration.
- Control/KD biological specimen, clone, aliquot and pairing metadata remain
  unresolved. Numbered acquisitions are not asserted independent biological
  replicates. Matching-only identifications and modified-form aggregation retain
  their original admission limitations.
- All original validation/test groups remain closed. These are training-component
  development results, with overlapping nested fit sets. No original held-out
  validation/test dump, IC50regression or500nMclassification applies. All inner
  and development predictions are retained. Predicate filters precede row
  materialization; whole-file hashes check bytes without evaluating other splits.
- No pretraining or synthetic biological augmentation. Synthetic arrays and
  known-context examples exercise actual pair algebra/fitting/statistics only.
  LocalCPU/oneBLASthread, noModalGPU or environment installations. Environment
  versions were recorded at the start of every stage.

Frozen stages (run in a newly registered copy, never overwrite existing evidence):

```sh
reproduce/launch.sh qualify
reproduce/launch.sh support
reproduce/launch.sh prepare
reproduce/launch.sh fit
reproduce/launch.sh evaluate
reproduce/launch.sh replay
```

The launch script records the original absolute path. A relocated copy must
explicitly update that path and preserve the original command. Report-only
regeneration: `reproduce/report.sh`.

## Support and the exact objective change

The target-free support stage precedes effect construction. It finds449unordered
within-parent pairs from the same214endpoints; these are dependent records, not
449independent biological measurements. There remain158within-parent response
degrees before cross-parent dependencies. Lexicographic endpoint IDs choose one
orientation; each feature and target difference uses that same orientation.

| Parent support | Parents | Endpoints | Pairs | Component/parent weight |
|---|---:|---:|---:|---:|
{support_table}

Hierarchically weighted endpoint fractions equal the parent-weight fractions
because endpoints split their parent's mass. Ordinary endpoint fractions are
also retained in `results/support.json`. For the42.73%weighted mass with only two
endpoints, reference loss and all-pair loss are identical. No such group is dropped.

The original objective gives equal weight to each of n-1contrasts against one
hash-selected reference. The new objective gives equal weight to all n(n-1)/2
unordered pairs. Both retain equal component and then equal parent weights.
Weights normalize within every fit/evaluation subset.

For within-parent endpoint residuals e, the checked identities are:

`mean_{{i<j}}(e_i-e_j)^2 = 2n/(n-1) * mean_i(e_i-mean(e))^2`

`mean_{{i!=a}}(e_i-e_a)^2 = n/(n-1) * [Var(e)+(mean(e)-e_a)^2]`

The n-dependent factor is retained. All-pair loss is independent of arbitrary
anchor choice and common parent shifts. Production qualification checks these
quadratic matrices and gradients on synthetic cases, with common shifts and
permutations; preparation checks every actual parent before any biological fit.
Independent replay separately reconstructs the pair/centered quadratic equality.

All449targets are built from original log10 intensity ratios and reconciled to
differences of original reference targets. All1,896contributors map to1,284distinct
source quantity cells, with acquisition labels independently verified. Complete
endpoint quantities, measurement IDs and pair lineage are preserved in results.

## Models, fitting and information flow

The four fixed conditions are zero response, ln(length),61peptide features and
149features including two external residues on each side of a peptide. The61map
contains length,20AAfractions,20first-residue and20last-residue indicators. Four
external positions each use20AA+OTHER+BOUNDARY. Exact parent contexts and their
uncertainty policy remain unchanged from0726; all214actual endpoints have one
literal occurrence. No biological identity, measured intensity or acquisition
metadata is a predictor input.

Endpoint features are subtracted. Each fitting objective computes its own
training-only weighted RMS scales, drops exactly zero coordinates and solves
weightedMSE+alpha||beta||², with no intercept, centering or rank normalization.
Thus regularization retains the registered scaling convention; this is not a
claim that models using different losses must have identical coefficient priors.
All surviving coefficients are solved. There are no untrained neural branches.

Five development and three inner component folds reuse the exact0726hashes.
Alpha[.01,.1,1,10,100] minimizes mean three-fold weightedMSE, with larger alpha on
exact ties. Only all-pair models are fitted. Both objective families predict both
spaces in their omitted training components; all models remain sealed before
aggregate outcome evaluation. The comparison makes no new blind-test claim.

## Primary: both objectives evaluated on all pairs

| Training loss | Features | Weighted RMSE | Pearson | Spearman | PredSD/targetSD |
|---|---|---:|---:|---:|---:|
{table("all_pairs")}

The new boundary model is worse than every preregistered matched control. It
improves over the original-objective boundary model in only1of5development folds.
The best displayed control is the reused length model, but its small gain over
zero is not qualification of an availability model.

| New boundary gain versus | Gain | Paired95%interval |
|---|---:|---:|
{interval_table}

The decision required >=5%gain over its three new-objective controls and the
original boundary control, predictionSD>=25%targetSD, improvements in4of5folds,
and a positive paired95%lower bound versus the original boundary model. All fail.
Bootstrap resamples the55whole components2,000times usingPCG64seed202610250726,
retains within-component weights and uses linear percentile intervals. Pairs are
never resampled as independent observations. Intervals describe development
uncertainty with overlapping fitted models, not prospective biological transfer.

## Secondary: both objectives evaluated on original reference contrasts

| Training loss | Features | Weighted RMSE | Pearson | Spearman | PredSD/targetSD |
|---|---|---:|---:|---:|---:|
{table("reference")}

These scores use a different target/weight space and must not be compared to
all-pair scores as if their scale were identical. Pearson is weighted covariance;
Spearman is weighted Pearson of ordinary average-tie ranks. Constant predictions
have undefined correlations. Full weighted/unweighted metrics and per-fold and
per-component errors are preserved in JSON/CSV, along with every prediction.

## Parameter support and verification

| Development fold | New model | Selected alpha | Solved coordinates | Design rank |
|---|---|---:|---:|---:|
{model_table}

Both sequence and boundary models choose alpha100 in all five folds. The small
spread persists after changing reference weighting; it is not fixed by this
objective. The result does not prove that the source biology lacks signal, and
it does not diagnose the older neural IC50underfit.

Independent replay imports no production core. It reconstructs raw-source targets
and pair membership, checks features with a separate literal scan, solves all240
new fits with augmentedSVD, verifies selection and both prediction spaces, uses
manual ranks and regenerates every bootstrap draw. It checks26,940inner and4,856
development predictions plus the632reused-control predictions. All
**{replay["numeric_values_checked"]:,}numerical checks** pass; maximum difference
**{replay["max_absolute_difference"]:.3g}**, within frozen2e-9relative/2e-10absolute
tolerances. Pair-target arithmetic also passes the tighter2e-12absolute/2e-13relative
pre-fit checks. Every numerical stage passes on its first attempt.

| Stage | Wall seconds | OS child peak MiB |
|---|---:|---:|
{runtime_table}

Unchanged resource limits:20min/stage,3GiB RSS,1GiB free reserve+32MiBstartup margin;
50ms process-tree monitoring plusOSchild peak. All pass. This is recorded local
experiment runtime, not a general hardware benchmark.

## Handoff

Status: closed negative result; stopping rule triggered. Do not extend this
response-only cohort's feature or optimizer search.
Next: qualify native baseline amounts and the observation/calibration information
needed to distinguish availability from MS response, using shared data and a
registered direct baseline before any new exact/discrete generative comparison.
Open: precursor-relative APC quantities, undetected/sub-ligand products,
machinery/clearance conditioning, observation calibration and independently
confirmed biological transfer. The full goal remains active and unachieved.
"""
(P / "README.md").write_text(text)
shutil.copy2(Path(__file__), P / "reproduce/source/report.py")
print("Objective report, complete metrics table and figure generated.")
