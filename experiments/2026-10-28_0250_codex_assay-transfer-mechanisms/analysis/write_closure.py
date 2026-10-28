"""Render the completed, independently verified comparison without new evaluation."""

from pathlib import Path
import csv
import json
import statistics

F = Path(__file__).resolve().parents[1]
R = F / "results"


def read(name):
    return json.loads((R / name).read_text())


def main():
    verified = read("verification-summary.json")
    assert verified["status"] == "passed" and len(verified["conditions"]) == 12
    decision = read("validation-decision.json")
    assert decision["status"] == "sealed" and not decision["test_used_for_decision"]
    replay = read("gpu-training-replay.json")["results"]
    assert len(replay) == 12 and all(x["status"] == "passed" for x in replay)
    assert all(
        x["maximum_absolute_error"] == x["serving_maximum_absolute_error"] == 0
        for x in replay
    )
    training = [
        read(f"{name}-training-verification.json") for name in verified["conditions"]
    ]
    assert all(x["status"] == "passed" for x in training)
    comparison = read("comparison-summary.json")
    metrics = {
        (x["variant"], x["split"], x["mhc_class"]): x for x in comparison["primary"]
    }
    runtime = list(csv.DictReader((R / "coverage-runtime.csv").open()))
    variants = (
        ("small-ic50", "Small IC50-only"),
        ("wide-ic50", "Wide IC50-only"),
        ("wide-six-natural", "Six outputs, natural"),
        ("wide-six-class-balanced", "Six outputs, class-balanced"),
        ("wide-six-primary-matched", "Six outputs, IC50-matched"),
        ("wide-six-primary-shuffled", "Matched, shuffled auxiliary values"),
    )
    table = [
        "| Condition | I validation ρ / RMSE | II validation ρ / RMSE | I test ρ / RMSE | II test ρ / RMSE | Mean worker min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, label in variants:
        cells = []
        for split in ("validation", "test"):
            for cls in ("I", "II"):
                x = metrics[(variant, split, cls)]
                cells.append(f"{x['spearman']:.4f} / {x['rmse']:.4f}")
        minutes = statistics.mean(
            float(x["wall_seconds"]) / 60
            for x in runtime
            if x["condition"].startswith(variant + "-s")
        )
        table.append("| " + " | ".join([label, *cells, f"{minutes:.1f}"]) + " |")
    hours = sum(float(x["wall_seconds"]) for x in runtime) / 3600
    text = (
        """## Completed controlled comparison

All twelve fixed fits, independent heldout/full-training checks and fresh-H100
evaluation/public-serving replays pass. The validation-only decision retains
**wide-ic50 as this cohort's prespecified reference**: no real six-output arm
passes every class-protection gate. The prior full-cohort IC50 recipe remains
unchanged. These are two-seed means, not ensembles; RMSE is log10(nM).
Exact IC50 validation counts are I=1,433 / II=623; test counts I=1,476 / II=683.

"""
        + "\n".join(table)
        + f"""

Worker time includes setup, fitting and evaluation export, but excludes queue,
image build, parent archive packaging and transport. Total worker time is
{hours:.3f} H100-hours; this is not a billing estimate. All runs share the same
model/trainer. Six-output heads add only 965 parameters to the 1,598,977-parameter
wide IC50 model; the small control has 515,457 parameters.

![IC50 comparison](ic50-comparison.png)

## What the controls establish

- **Capacity is a real confound in the earlier comparison.** Wide versus small
  IC50 improves I Spearman by 0.0272 on validation and 0.0301 on test. II improves
  by 0.0298 on validation but falls by 0.0124 on test; its paired test interval
  includes zero. A general class-II capacity gain is therefore not established.
- **Primary-task dilution explains much of the II deficit.** Matching IC50
  exposure improves II over the natural six-output mix by 0.1151 validation and
  0.0485 test Spearman. Paired peptide-bootstrap 95% intervals are [0.0669, 0.1648]
  and [0.0143, 0.0797]. Mean II training Spearman is 0.7656 for wide IC50, 0.5862
  for natural six-output training, and 0.7289 after matching. The natural arm is
  fitting the primary target poorly, rather than showing only a larger train/test
  gap. Matching counts still changes batch size, compute and mean-loss
  normalization; it does not match the primary gradient contribution exactly.
- **Measured auxiliary values help I beyond the conditional shuffle.** Real
  versus shuffled auxiliary values improve I by 0.0251 validation / 0.0297 test
  Spearman, with intervals [0.0061, 0.0450] / [0.0120, 0.0493]. II differences are
  0.0020 / 0.0151, with intervals [-0.0283, 0.0293] / [-0.0139, 0.0443]. This
  supports peptide-specific quantitative auxiliary benefit for I; additional II
  benefit is uncertain. The shuffle preserves groove/qualifier groups and many
  constant bounds, so it retains informative structure.
- **Class balancing alone is a trade-off.** Relative to the natural mix, it
  recovers II test Spearman by 0.0492 but loses I by 0.0512. It still supplies
  fewer II IC50 labels than the primary-matched and IC50-only arms. Equal class
  totals do not equalize endpoint/locus coverage or primary-task exposure.
- **Selection is not the main explanation here.** A posthoc equal-class-RMSE
  history audit changes three of twelve checkpoint choices: both class-balanced
  runs and small-IC50 seed43. It leaves both natural, wide-IC50 and primary-matched
  choices unchanged. No alternative checkpoint was promoted or evaluated on test.

The primary-matched arm passes the macro-gain, two-seed macro consistency and
RMSE gates, but fails the no-class-rank-drop and each-class-improves gates. Its
II validation Spearman is 0.0175 below the wide control, even though its II test
point estimate is 0.0161 higher. Test inspection does not reverse the sealed
decision. All confidence intervals above are conditional paired peptide-cluster
bootstraps of the mean over two seeds, not uncertainty over new studies or seeds.

![Exposure curves](exposure-curves.png)

## Failure modes, support and scope

The II loss is concentrated in DQ. Validation DQ Spearman is 0.6450 for wide
IC50, 0.4584 for natural six-output and 0.6028 for primary-matched; corresponding
test values are 0.6490, 0.5757 and 0.6243. DP test values are 0.6076, 0.6224 and
0.6679. Only 11 validation / eight test exact IC50 rows are DR, which does not
qualify DR accuracy. The I gains occur in both A and B; no IC50 C outcomes exist.
Full per-groove, locus, length, target-range, censoring and novelty strata are
preserved. Novelty groups are defined relative to each arm's training set, so
some class-I group memberships differ between the IC50-only and six-output arms.

Every non-padding parameter coordinate has same-coordinate gradient and update
evidence. Only fixed padding remains untouched (128 small / 192 wide coordinates).
The 384/768 attention key-bias coordinates are structurally redundant despite
numerical updates; constant human-species input does not identify a species effect.
The II EC50 head has 17 training / zero validation / two test rows; half-life has
11 / one / zero. II direct KD has only 35 exact rows in each heldout split, and
KD-via-IC50 has 54 validation / 58 test. Allocated shared outputs remain unqualified
where supervision is sparse or absent.

II IC50 register weights remain diffuse: normalized test entropy approximately
0.972–0.990, with no maximum register weight above 0.5. These are statistical
weights without measured register truth or physical-occupancy qualification.
All test peptide strings are absent from training, but 1,475/1,476 I and 683/683 II
exact IC50 test rows share a study with training; 573/683 II test rows share at
least one nine-residue substring with training. This is not study- or homology-
independent biological qualification, MS ligand ranking, processing, nonhuman
prediction or source-class-only zero-shot transfer.

## Completed verification and retained evidence

Independent checks cover original targets/units, censor-aware losses, exact-only
training scales, all heldout regression and 500 nM classification metrics,
initialization equality, sampler exposures, checkpoint hashes and coordinate masks.
There are 208 endpoint/class/split metric rows, including loss, Pearson, Spearman,
RMSE, accuracy, balanced accuracy, precision, recall, F1, AUROC and AUPRC where
defined. Half-life retains log10(minutes) and has no concentration threshold.
"""
        + f"""
All {sum(x["unique_training_observations"] for x in training):,} saved training
predictions were independently checked. All {sum(x["n"] for x in replay)} saved
training probes replay exactly through evaluation and the guarded public predictor
on a fresh H100 (maximum error zero for both paths). No new heldout inference,
refitting or tolerance relaxation was used for replay. The training app released
both GPU containers before the separate one-H100 verification run.

Canonical files: `metrics-independent.csv`, `common-ic50-means.csv`,
`all-output-means.csv`, `training-metrics-independent.csv`,
`groove-metrics-min20.csv`, `failure-strata.csv`, `output-class-support.csv`,
`coverage-runtime.csv`, `paired-bootstrap.json`, `register-diagnostics.json`,
`checkpoint-selection-diagnostics.json`, `validation-decision.json`,
`verification-summary.json` and `gpu-training-replay.json`, all under `results/`.
Per-example validation/test dumps are in each condition directory. Full TRAIN
dumps are retained in checksummed raw archives and local ignored gzip copies.
The same common manifest trainer now supports explicit stratum sampling weights;
unweighted behavior and exact resume remain verified. No task-specific training
path or six-output default was introduced. PR131 stays draft pending #130/#132.

"""
    )
    (R / "closure-summary.md").write_text(text.rstrip() + "\n")
    (R / "comparison-table.md").write_text("\n".join(table) + "\n")
    print("Rendered verified comparison and closure narrative.")


if __name__ == "__main__":
    main()
