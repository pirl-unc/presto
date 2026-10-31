"""Render the complete frozen evaluation; no predictive choices or fitting."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results/evaluation"
summary = json.loads((out / "summary.json").read_text())
preparation = json.loads((ROOT / "results/preparation.json").read_text())
verification = json.loads((ROOT / "results/independent-verification.json").read_text())
metrics = summary["metrics"]
order = [
    "zero",
    "condition",
    "fixed_precursor",
    "calibrated_precursor",
    "sequence_0.001",
    "sequence_0.01",
    "sequence_0.1",
]
physical = [
    "0810_uniform_exact",
    "0810_residue_exact",
    "1007_length_exact",
    "1007_residue_length_exact",
    "0810_augmented_exact",
    "0810_augmented_sampled",
    "1007_augmented_exact",
    "1007_augmented_sampled",
]
strata = pq.read_table(out / "strata-qualified.parquet").to_pylist()
condition_metrics = {
    (r["split"], r["model"], r["stratum"]): r
    for r in strata
    if r["scope"] == "direct" and r["stratum_type"] == "condition"
}
conditions = sorted({k[2] for k in condition_metrics})
condition_rows = []
for split in ["validation", "test"]:
    for c in conditions:
        base = condition_metrics[split, "calibrated_precursor", c]
        seq = condition_metrics[split, "sequence_0.1", c]
        condition_rows.append(
            {
                "split": split,
                "condition": c,
                "baseline_rmse": base["rmse"],
                "sequence_rmse": seq["rmse"],
                "rmse_gain_percent": 100 * (1 - seq["rmse"] / base["rmse"]),
                "sequence_spearman": seq["spearman"],
                "prediction_sd": seq["prediction_sd"],
            }
        )
comparison = [
    {"population": population, "model": name, **m}
    for population, models in metrics.items()
    for name, m in models.items()
]
for name, rows in [("comparison.csv", comparison), ("conditions.csv", condition_rows)]:
    with (out / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), layout="constrained")
test_conditions = [r for r in condition_rows if r["split"] == "test"]
axes[0].barh(
    [r["condition"].removeprefix("hap1_").removesuffix("_ko").upper() for r in test_conditions],
    [r["rmse_gain_percent"] for r in test_conditions],
    color="#246f9e",
)
axes[0].axvline(0, color="#333333", linewidth=0.8)
axes[0].set_xlabel("Test RMSE reduction vs precursor/context (%)")
axes[0].set_title("Sequence model: all ten HAP1 knockouts")
names = [
    "sequence_0.1",
    "0810_residue_exact",
    "1007_residue_length_exact",
    "0810_augmented_exact",
    "1007_augmented_exact",
    "1007_augmented_sampled",
]
labels = [
    "Sequence baseline",
    "Residue trimming",
    "Residue + length trimming",
    "Sequence + old exact",
    "Sequence + new exact",
    "Sequence + new sampled",
]
axes[1].barh(
    labels,
    [metrics["test:erap1"][n]["rmse"] for n in names],
    color=["#246f9e", "#a5a9b0", "#a5a9b0", "#698c76", "#698c76", "#698c76"],
)
axes[1].axvline(metrics["test:erap1"]["sequence_0.1"]["rmse"], color="#246f9e", linestyle="--")
axes[1].set_xlabel("ERAP1 test RMSE (log10 elution ratio; lower is better)")
axes[1].set_title("Processing adds no useful predictive gain")
for ax in axes:
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Frozen held-out evaluation • same-study molecular-component splits", fontsize=13)
fig.savefig(out / "comparison.png", dpi=170)
fig.savefig(out / "comparison.pdf")


def number(value):
    return "undefined" if value is None else f"{value:.6f}"


def table(scope, names):
    lines = [
        "| Model | Validation RMSE | Test RMSE | Test R² | Test Pearson | Test Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        v, t = metrics["validation:" + scope][name], metrics["test:" + scope][name]
        lines.append(
            f"| {name} | {v['rmse']:.6f} | {t['rmse']:.6f} | {t['r2']:.6f} | {number(t['pearson'])} | {number(t['spearman'])} |"
        )
    return "\n".join(lines)


cohort_counts = {}
for split in ["validation", "test"]:
    rows = pq.read_table(out / split / "cohort.parquet").to_pylist()
    cohort_counts[split] = {
        "observations": len(rows),
        "peptides": len({r["peptide_sequence"] for r in rows}),
        "components": len({r["component_id"] for r in rows}),
    }
(out / "cohort-counts.json").write_text(json.dumps(cohort_counts, indent=2) + "\n")
condition_table = [
    "| Knockout | Test control RMSE | Test sequence RMSE | Reduction | Test Spearman |",
    "|---|---:|---:|---:|---:|",
]
for row in test_conditions:
    condition_table.append(
        f"| {row['condition']} | {row['baseline_rmse']:.6f} | {row['sequence_rmse']:.6f} | {row['rmse_gain_percent']:.2f}% | {number(row['sequence_spearman'])} |"
    )

text = f"""# Frozen HAP1 outer evaluation: sequence signal survives, processing adds no gain

Codex / GPT-6, October 24, 2026. Registered before held-out outcome access at
Presto `9a8742db807d805f276f67658ccae9486faab5d7`, with the recorded unrelated task
dirt and new frozen experiment code. [Prospective plan](../agents/codex/plans/hap1-frozen-outer-evaluation.md).
No new training, calibration, model selection or source curation occurred.

The preselected sequence_0.1 model reduces test RMSE **9.167%** against the
preselected calibrated-precursor comparator (validation **10.897%**). Test R² is
**0.24986** and Spearman **0.46623**. Predictions vary within all ten knockouts;
all ten have lower test RMSE. Both prespecified direct-model gates pass.

The preselected ERAP1 processing augmentation does not help: test RMSE
**0.585337** versus **0.585327** without it, a **0.001744% worsening**. Validation
improves only0.1471%, below the2% requirement. Both physical-model gates fail.
Sampled-trained parameters perform essentially identically. This is evidence for
same-study MHC-elution response prediction, not APC peptide-concentration prediction.

## Biological and source contract

Shapiro2025 PXD056426 human HAP1 MHC-I elution and separate global MaxLFQ,
unchanged canonical Hitlist source `88cad10d7bc2a638305cb0d37ad96ac2826c3badaf7b3e0aad7928820df5c4b8`
and registered recipe `a1eec392c06485966e131663f85e7599dcd9e5ff621841eb0d44eeaafe779309`.
HLA-A*02:01/B*40:01/C*03:04 is fixed genotype context, not peptide-specific restriction.
The assay label and sole predicted output are **log10 positive KO replicate intensity
minus mean log10 positive WT intensity**. WT requires at least two positive cells;
the original520B2M-positive sequence exclusions remain. All candidate parent
origins and source-cell/sample/control lineage are retained. Complete precursor
support requires the original protein-group/positive replicate policy. MaxLFQ
provides a condition-level parent-response surrogate, not paired molar concentration.
Missing, zero and filter-NA source states stay distinct; no imputation, negative
generation, synthetic biological labels or inferred censoring bounds. Other assay
families, affinity/nM and500nM binder metrics are not applicable.

The models were fitted previously on87,260outer-training cells,6,317peptides and
1,798molecular components. Frozen validation contains{cohort_counts["validation"]["observations"]:,}cells,
{cohort_counts["validation"]["peptides"]:,}peptides and{cohort_counts["validation"]["components"]:,}components;
test contains{cohort_counts["test"]["observations"]:,}cells,{cohort_counts["test"]["peptides"]:,}peptides and
{cohort_counts["test"]["components"]:,}components. Components do not overlap across splits.
These counts do not establish independent biological specimens or studies.
ERAP1 contributes1,970validation cells/774peptides/795origins/425,963paths and
2,297test cells/888peptides/929origins/554,402paths. All relevant parent alternatives
enter deterministic exact inference.

## Frozen models and information flow

- Seven direct models from0730: zero; condition offset; unit precursor coefficient;
  nonnegative calibrated precursor coefficient; three condition-specific sequence
  ridge penalties.001/.01/.1. Features are peptide length,20amino-acid fractions,
 20N-terminal and20C-terminal indicators, supplied KO identity and precursor ratio.
  Archived training-only feature means, scales, masks, condition centers, coefficients,
  offsets and gamma0.1895535345 are reused unchanged. Original fits use weighted
  closed-form regression and ridge penalty; no pretrained sequence encoder.
- Four0810physical models: shared trimming rate or residue-specific rates,
  physical-only or added to the frozen sequence baseline; exact- and sampled-trained
  augmentation. Two or21numeric coordinates plus the original profiled offset.
- Four1007physical models add two remaining-length modifiers (1–8,9–16reference,
  >=17): four or23numeric coordinates. The original full-training150Adam steps,
  lr.03,seed42 and.01rate/reduction penalties (plus.01length penalty for1007) are
  historical training contracts, not new training in this experiment.
- Physical inference starts with unit parent supply/uniform cleavage source,
  then N-terminal trimming and clearance. Sum all query origins and birth paths;
  apply frozen precursor gamma and effective KO rate scale. Physical-only output
  is log10 KO/WT amount ratio plus archived offset. Augmentation adds that ratio
  minus the direct model's precursor term to the direct prediction, plus archived
  offset. No offset is estimated from validation or test.
- Sampled-trained models use exact forward inference here. Their prior training
  used exact-forward/conditional-score hybrid gradients, not simulator-onlyRL/GRPO.
  No new sampler, clipping, standardization, sample reuse or KL penalty is tested.
  No affinity, stability, TAP, detector, TCR, species or MHC-class branch is trained.

## Held-out outcomes

Within each split, weights give equal mass to conditions, components within
condition, peptides within component and replicate cells within peptide. ERAP1
weights are renormalized. MSE, RMSE, MAE, R² and SD are weighted; Pearson/Spearman
are unweighted. All metrics and condition/component/length strata are preserved.
No confidence intervals or independent-cohort replication were performed.

{table("direct", order)}

All ten conditions improve against the prespecified precursor comparator:

{chr(10).join(condition_table)}

The ERAP1-only comparison uses the same frozen sequence model and all eight
physical parameter sets. Other sequence penalties are shown descriptively;
their ERAP1 test outcomes do not select a replacement for sequence_0.1.

{table("erap1", order + physical)}

![Held-out comparisons](results/evaluation/comparison.png)

## Verification, numerical qualification and runtime

Before outcome access, all source/recipe and upstream closure hashes were checked;
all15models reproduced677,068archived training prediction values. Independent
affine and scalar-recurrence inference agreed within1.14e-13. Target mutation
did not change predictions. Preparation took{preparation["seconds"]:.2f}s,
peak{preparation["peak_gib"]:.3f}GiB. Both sets of covariate-only predictions were
saved and hashed before either outcome set was read. Outcome access is timestamped;
both outer sets are now inspected and closed to further tuning.

Evaluation including all metrics took{summary["seconds"]:.2f}s,peak{summary["peak_gib"]:.3f}GiB.
Independent post-evaluation reconstruction verified{verification["n_prediction_values"]:,}prediction
values (maxerror{verification["max_prediction_error"]:.3g}), every source origin/input,
source rows, hierarchical weights, split disjointness, gates and
{verification["n_stratum_metric_records"]:,}stratum records;{verification["seconds"]:.2f}s,
peak{verification["peak_gib"]:.3f}GiB. Original-source cell-level curation audits are
reused by hash; this is not new experimental biological validation. Local float64
CPU, one numerical thread; no Modal/GPU/installations. All resource caps pass.
Inference time does not compare exact versus sampled *training* costs.

Five validation component strata for uniform/length-only physical controls emitted
near-constant correlation warnings: prediction ranges3.3e-14to5.9e-14. Their Pearson
and Spearman values are uninterpretable. Original scores/predictions are preserved;
`strata-qualified.parquet` marks those ten correlation cells undefined under
SciPy's numerical warning criterion. Primary metrics/gates and every prediction
are unchanged; the qualification receipt verifies that these are the only edits.
This post-evaluation numerical qualification must not be interpreted as another
biological fit. An equivalent endswith lint amendment was frozen before preparation;
its original source/receipt is retained under`reproduce/pre-qualification-lint`.

## Reproduction, artifacts and limits

- `reproduce/launch.sh`, `launch.json`, `checkpoints.json`, copied plan and
  `reproduce/source/` pin the actual invocation, dependencies, code and full-model
  parameters. The launcher intentionally refuses to overwrite an existing
  evaluation. Use a new output copy to reproduce, retaining its frozen paths.
- `results/preparation.json`, `results/independent-verification.json` and
  `launch_logs/` contain qualification and closure. Frozen input directories are
  listed inlaunch.json; original recipe/source versions are not relabeled by later releases.
- `results/evaluation/{{validation,test}}/` contains every covariate, source-linked
  outcome, per-example prediction, parent/origin record and physical correction.
  Prediction-onlyNPZs and the pre-outcome receipt preserve separation from scoring.
- `summary.json`, `comparison.csv`, `conditions.csv`, `strata-qualified.parquet`,
  correlation qualification and PNG/PDF provide all reported results. Original
  `strata.parquet` remains the raw reduction. Posthoc analysis invocation is recorded
  separately; no inference code/checkpoints changed after opening outcomes.

**Decision:** retain sequence_0.1 as the HAP1 observed-positive response reference.
Do not promote the processing augmentation or retune either inspected outer split.
The strongest demonstrated signal is a parsimonious condition-specific sequence
regression. Elution conflates processing, MHC selection and detection; these data
do not separately identify those factors. HAP1 is not a professional APC; seen
conditions, shared WT controls/study and unresolved clone/specimen/homology
independence limit generalization. No unseen-KO, independent-cohort, free-peptide,
short-product or precursor-relative concentration validation is established.

**Status:** biological evaluation and independent closure complete. **Next step:**
independent processing/selection/detection constraints and prospective APC data,
with the simple sequence model as a mandatory comparator. **Open question:** how
to learn actual availability rather than only this observed MHC-elution response.
The broader availability goal remains active and unachieved.
"""
(ROOT / "README.md").write_text(text)
print(json.dumps({"cohort_counts": cohort_counts, "condition_results": condition_rows}, indent=2))
