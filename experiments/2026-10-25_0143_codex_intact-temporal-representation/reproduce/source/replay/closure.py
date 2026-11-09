"""Close the registered development comparison without a blind-test claim."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


config = read("reproduce/launch.json")
resources = {stage: read(f"results/{stage}-resource.json")
             for stage in ("prepare", "qualify", "fit", "evaluate", "replay")}
assert all(record["passed"] for record in resources.values())
replay = read("results/independent-replay.json")
assert replay["passed"]
evaluation = read("results/evaluation.json")
preparation = read("results/preparation.json")
fitting = read("results/fit-summary.json")
physical = read("results/physical-fits.json")
direct = read("results/direct-fits.json")
metrics = evaluation["all_metrics"]
pooled = [r for r in metrics if r["scope"] == "pooled_outer_development"]
assert len(pooled) == 7 and len(physical) == 45

with (ROOT / "results/comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(metrics)

selected = []
for fold in range(5):
    for arm in ("physical_uniform", "physical_residue_penalized", "physical_residue_unpenalized"):
        choices = [r for r in physical if r["outer_fold"] == fold and r["arm"] == arm]
        selected.append(min(choices, key=lambda r: r["objective"]))

selection = {"physical_selected": [{k: r[k] for k in
             ("outer_fold", "arm", "start_number", "success", "objective", "mse", "tau_minutes",
              "all_parameters_changed", "gradient_absolute_totals", "raw_boundary_coordinates")} for r in selected],
             "ridge_selected": [{"outer_fold": r["outer_fold"], "arm": r["arm"],
                                  "alpha": r["alpha"], "rank": r["transform"]["rank"]}
                                 for r in direct if r.get("role") == "selected_outer"]}
(ROOT / "results/selected-models.json").write_text(json.dumps(selection, indent=2) + "\n")

def number(value):
    return "undefined" if value is None else f"{value:.6f}"


table = ["| Model | Development CV RMSE | R² | Pearson | Spearman | Prediction SD | Target SD |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for record in pooled:
    table.append("| " + " | ".join([record["arm"]] + [number(record[k]) for k in
                 ("rmse", "r2", "pearson", "spearman", "prediction_sd", "target_sd")]) + " |")
runtime = ["| Stage | Wall seconds | Peak polled RSS bytes |", "|---|---:|---:|"]
for stage, record in resources.items():
    runtime.append(f"| {stage} | {record['wall_seconds']:.3f} | {record['peak_polled_rss_bytes']} |")
sequence_decision = "passes" if evaluation["sequence_gate_passed"] else "fails"
physical_decision = "passes" if evaluation["unpenalized_improvement_gate_passed"] else "fails"
optimizer_failures = sum(not r["success"] for r in selected)
report = f"""# Temporal representations: measured changes in development CV

Codex / GPT-6. Registered {config['registered_utc']} before execution at Presto
`{config['presto_commit']}` with recorded task dirt and experiment-local code.
[Prospective specification](reproduce/plan.md), [exact launch/environment/input hashes](reproduce/launch.json).

The sequence-versus-length comparison **{sequence_decision}** its prespecified
development screen: RMSE gain **{evaluation['sequence_gain_percent']:.3f}%**.
Removing the physical rate penalty **{physical_decision}** the separate 5% improvement
gate: gain **{evaluation['unpenalized_gain_percent']:.3f}%**. There are
**{optimizer_failures} unsuccessful selected physical optimizers** among 15;
failed convergence remains reported and is not promotion-qualified.

These results concern nested development CV on previously inspected training
measurements. They are **not a new blind biological test**, independent-protein
transfer, an RL comparison, or evidence of calibrated APC concentrations.

## Data, conditioning and split

Exactly the original 0046 training cohort: 626 positive measured product/intact
substrate log10 intensity ratios, 96 peptides, nominal times 0–80 minutes.
PXD058121 F2 measures human alpha-synuclein (140 residues) with purified mouse-liver
20S; one source animal, unresolved replicate independence and unknown MS scaling.
Canonical recipe `pxd058121_product_substrate_forecast_v1`, manifest
`fb04bd482f5902d776bcd853d5280d86df7fd8b74424574c1bf793e3d2332f0c`, unchanged
source and exclusions. Shared data remain in the pinned prior-family paths.
No new curator, imputation, negatives, censor bounds, synthetic biological labels
or label-dependent filtering. Other assay families and IC50/binder metrics do not apply.

The first eligible time's replicate mean is supplied as a measured baseline for
each peptide. Predict later **change** in log product/intact ratio. Baseline cells
are not predicted targets; their exact IDs/values and reuse dependencies are saved
in [baselines](results/baselines.json) and [all changes](results/changes.parquet).
Equal peptide/time/replicate weights are recomputed in each split. Baseline noise
is shared across a peptide's later changes. No peptide identity or earlier fitted
slope is an input feature.

Five peptide folds use SHA256(`intact-temporal-representation-v1|` + sequence)
modulo 5; three inner folds use `intact-temporal-inner-v1|` modulo 3. Repeated cells
of a sequence stay together. Outer validation requires >=10 peptides, inner >=5;
all support decisions are in [preparation](results/preparation.json). Peptides
overlap in one parent: this is not a homology-disjoint evaluation. Original 100/120
minute outcomes are not read. Earlier PXD044463 test descriptive outcomes were
historically inspected, so that partition is not treated as a fresh external test.

## Model and fitting contract

Seven arms: zero change, shared elapsed-time slope, length-conditioned slope,
sequence-conditioned slope, uniform physical rates, residue rates with penalty
0.01, and residue rates without penalty. Length uses log length; sequence adds
20 amino-acid fractions and 20 indicators per terminus. Training-only unique
peptide means/scales and supported covariance eigenvectors remove constant/null
directions at the fixed relative 1e-10 cutoff. Prediction is elapsed time/80 times
a shared intercept plus the projected features' coefficients. The intercept is
unpenalized; ridge penalizes alpha times mean squared other coefficients. Inner
peptide CV selects alpha .001/.01/.1/1, with ascending-alpha ties and equal peptide
weight across inner folds. All 120 candidate fits and inner-validation predictions
are retained. No pretraining or sequence encoder is used.

Physical changes subtract the frozen production log product/parent prediction
at baseline time from its prediction at the later time, cancelling fixed peptide
offsets. Uniform has two parameters; residue has 17 category rates plus effective
pre-exposure tau. Bounds remain rates 1e-7–1/min and tau .001–120 minutes. Each of
15 fold/arm combinations uses three data-free starts (.001/min,tau10),
(1e-5/min,tau60), (1e-4/min,tau30). No initializer sees validation-peptide labels.
L-BFGS-B uses raw bounds [-30,30], maxiter1500, maxfun3000, maxls40, gtol1e-9,
ftol1e-13; choose the lowest training objective. The objective is balanced change
MSE plus alpha times mean squared centered log category rates. All 45 starts,
objective/gradient histories, final predictions, parameters and convergence
failures are retained. [Selection and parameter coverage](results/selected-models.json).
Gradient coverage does not imply biological identifiability.

This isolates representation/regularization after 0046 found matched exact and
conditional-score predictions without sampled-gradient benefit. No new RL claim
is made here. Affinity, stability, transport, MHC, TCR, APC machinery and detector
heads are not trained. A promising candidate still needs a frozen prospective
biological exact-versus-discrete comparison before advancing the full goal.

## Outcomes

Pooled outer-development errors and SDs are equal-peptide/time/replicate weighted;
Pearson and Spearman are unweighted. These are held-out peptide folds within an
already inspected development cohort. [All metrics](results/evaluation.json),
[fold training/validation CSV](results/comparison.csv), [peptide strata](results/strata.parquet),
[outer predictions](results/predictions.parquet), [inner predictions](results/inner-predictions.parquet).

{chr(10).join(table)}

Sequence promotion requires >=5% RMSE gain over length, positive Pearson and
Spearman, and prediction SD >=0.1 target SD. Physical regularization change requires
>=5% gain. Paired 1000-peptide bootstrap (seed31415) intervals and exact draws are
saved; they describe uncertainty within this parent, not independent specimens.
No threshold, split, features or tuning budget changes after observing outcomes.
If sequence fails, stop expanding this single-parent representation and seek
broader measured substrates/contexts. Either result requires fresh evaluation;
the full precursor-relative availability goal remains unachieved.

## Verification and resources

Independent source reconstruction checks baseline IDs/values, fold assignments,
prediction coverage and weights. Independent SVD projectors/augmented least
squares check direct fits and selection; NumPy physical formulas check all starts,
stored objective evaluations, final finite-difference gradients and selected
predictions. All overall/stratified metrics, paired bootstrap and fixed gates are
replayed. [Exact verification scope and errors](results/independent-replay.json).
L-BFGS internal line-search/update implementation is not independently replayed.

{chr(10).join(runtime)}

Local float64 CPU, one numerical thread; no Modal/GPU/installations. Fixed caps:
3GiB RSS,20minutes/stage,32MiB logical artifacts,1GiB free disk plus32MiB startup
margin. Launch each stage listed in launch.json using `sh reproduce/launch.sh STAGE`;
stage receipts prevent overwriting. Source snapshots are under `reproduce/source/`.
All stages were initially deferred below the disk reserve; no guard was weakened.
The artifact manifest hashes this report and retained outputs after closure.
"""
(ROOT / "README.md").write_text(report)
files = {}
excluded = {"artifact-manifest.json", "results/closure.json", "results/closure-resource.json",
            "results/closure-started.json", "launch_logs/closure.log"}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts:
        continue
    name = str(path.relative_to(ROOT))
    if name not in excluded:
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
(ROOT / "artifact-manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
result = {"passed": True, "scope": "development_only", "n_physical_starts": len(physical),
          "n_prediction_rows": fitting["n_prediction_rows"], "n_hashed_files": len(files),
          "logical_bytes": sum(r["size_bytes"] for r in files.values()),
          "new_blind_test": False, "original_outer_outcomes_read": False,
          "test_absence_reason": "Original outcomes already inspected; representation qualification uses original training only.",
          "sequence_gate_passed": evaluation["sequence_gate_passed"],
          "physical_regularization_gate_passed": evaluation["unpenalized_improvement_gate_passed"],
          "selected_physical_optimizer_failures": optimizer_failures, "goal_achieved": False}
(ROOT / "results/closure.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
