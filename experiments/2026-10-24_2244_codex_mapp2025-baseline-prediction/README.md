# Baseline MAPP: modest sequence signal; exact and RL beat uniform

Codex / GPT-6, October 24, 2026. Registered prospectively from Presto
`98d0ac85356692237b16903a2156ee2b6f51b013` with recorded task dirt and new experiment
code. [Frozen specification](reproduce/plan.md), [launch/env/source hashes](reproduce/launch.json),
[figure](results/heldout-comparison.png), and [per-example predictions](results/predictions.parquet).

**The primary sequence-versus-length gate fails:** validation RMSE improves 5.386%,
but test improves 4.334%, below the fixed 5% requirement. Paired test-component
bootstrap 95% interval is [2.288%, 6.643%]. No threshold, feature, split or model
choice was changed after seeing this outcome.

**Both matched distribution models beat uniform on validation and test.** Exact
test reverse KL is 0.422502 versus uniform 0.477545; REINFORCE mean is 0.422963
(seed SD 0.001912). Exact test RMSE is 0.415095 versus uniform 0.439892, a 5.637%
gain, bootstrap [3.220%, 8.118%]. This is useful, modest captured-signal prediction.
It is not evidence of APC concentrations, full intracellular availability or an RL
advantage. The simpler exact calculation remains the practical distribution reference.

## Data, source and split

Original Goldberg2025/PXD052314 six A549 baseline control acquisitions. Use the
complete original search export rather than the publisher-selected subset. Source
schema25/catalog31, manifest `cb1bbc22d03140aedff4b3d3a31df545712f36ce7709ec01e34abf7c76b7fc8f`.
Canonical source and all original baseline decisions were independently audited
in the preceding 2210 family. Runtime uses isolated Hitlist2.0.31 development on
base176a62e, with all changed runtime files frozen here; the shared installed
environment was not changed. Source PR827 merged5a8b1aa; source release and new
composition recipe publication remain separately tracked below.

Shared recipe `goldberg2025_observed_baseline_composition_v1` requires canonical,
positive finite, explicitly unflagged, source-human-only candidate rows; peptide
PEP in [0,.01), per-run `By MS/MS`, one distinct supplied parent accession, and
at least three distinct observed sequences per accession/acquisition group.
Duplicate eligible sequences within a group stop admission. Confidence is a
source-based policy, not a newly estimated cohort FDR. Peptide totals may aggregate
forms. Missing/zero/unobserved peptides are not negatives; no synthetic biological
labels, imputation, affinity filtering or chemistry-specific MS correction.

All30,364original peptide rows and every supplied parent label enter a dependency
graph before filtering, including excluded bridges. Salted compact sorted-node
SHA256/modulo10 split is fixed60/20/20. Source labels are not inferred aliases or
homology groups. Support requires100/20/20components and20held-out groups.

| Split | Components | Accession/run groups | Observations | Unique peptides |
|---|---:|---:|---:|---:|
| Train |274|1330|19593|5182|
| Validation |74|372|8649|2206|
| Test |110|649|12459|3302|

All394,732original peptide-cell decisions remain available.44,120cells pass the
pre-group policy;3,419cells in2,636small groups are excluded, leaving40,701cells.
Weights give equal mass to components, then groups within components, then peptides.
Repeated acquisitions do not establish independent specimens. The same study and
six runs occur across partitions; full reference, homology, specimen and exact
precursor pairing remain unresolved.

## Model and training

One peptide sequence ->61raw features (natural-log length,20amino-acid fractions,
20N-terminal/20C-terminal indicators) -> fitting-only weighted standardization ->
within-observed-group centering -> supported SVD projection -> one scalar log10
signal score. Supported dimensions are1for length and58for sequence. No output
intercept, pretrained encoder, parent input, identity embedding, or inferred context.
Targets are group-centered log10 intensity and positive observed-set intensity
fractions. Predicted fractions are softmax(ln10 times score), normalized only over
the supplied observed set; absent mass is not identified.

- Uniform: zero centered scores/equal fractions.
- Length/sequence ridge: weighted MSE+alpha*mean(beta²), alpha .001/.01/.1/1,
  closed-form solve. Validation-only choices are length .001 and sequence1.
- Exact: same58-coordinate linear sequence score, group-weighted reverseKL(q||p)
  +.01mean(beta²),500full-batch float64Adam updates,lr.02,betas(.9,.999),eps1e-8,
  zero initialization, fixed final checkpoint. All58coordinates receive gradients
  and change. Last gradient max9.96e-13; last50-step objective change is zero at
  stored precision. This is not a general global-optimum guarantee.
- Discrete REINFORCE64: identical model/objective/budget; fresh64candidate-identity
  draws per group/update, leave-one-out reward baseline, detached log(q/p) reward,
  analytic regularizer; seeds42/43/44. All58coordinates receive gradients/change.
  No clipping, group-standardization, reuse, reference-KL, restart or early stopping.
  This samples observed peptide identities, not complete processing trajectories.

No other Presto branches train: no explicit processing, transport, MHC affinity or
stability, TCR, species/class, KO or detector head. This experiment establishes a
sequence baseline before attempting to identify those latent contributions.

## Held-out results

| Fixed model | Validation RMSE | Test RMSE | Validation reverse KL | Test reverse KL | Test R² | Test Spearman |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0.419737 | 0.439892 | 0.431589 | 0.477545 | 0.00000 | undefined |
| length_ridge_0.001 | 0.410127 | 0.434413 | 0.410675 | 0.464617 | 0.02476 | 0.12796 |
| sequence_ridge_1 | 0.388039 | 0.415585 | 0.378544 | 0.424381 | 0.10746 | 0.29551 |
| exact | 0.388470 | 0.415095 | 0.376448 | 0.422502 | 0.10956 | 0.29363 |
| reinforce_42 | 0.387946 | 0.415376 | 0.375926 | 0.422539 | 0.10836 | 0.28992 |
| reinforce_43 | 0.389502 | 0.416381 | 0.378255 | 0.425052 | 0.10404 | 0.28877 |
| reinforce_44 | 0.388821 | 0.414296 | 0.376133 | 0.421299 | 0.11299 | 0.29057 |

Errors/R²/SD use the component/group weights; correlations are unweighted. Uniform
correlations are undefined. The exact model's test prediction SD is approximately
0.17 versus target0.439892: nonconstant but substantially less variable than the
measurements. Seed spread measures training stochasticity, not biological uncertainty.
All13variants,39split-model summaries and6,188acquisition/component strata are
archived; no test-selected replacement or ensemble is introduced.

## Verification, runtime and failures

Independent NetworkX/source-candidate replay reproduces all33,179dependency nodes,
all394,732decisions and40,701targets/weights, maximum arithmetic error3.55e-15.
Actual-cohort numerical gates at three frozen parameter points pass: autograd versus
analytic max3.11e-15, finite differences max2.08e-9, 256×64-sample gradient relative
mean errors .00326–.00370 and max2.79standard-error units. Qualification precedes fits.

All checkpoints and covariate predictions were hashed before the test-label file
was opened once. Independent NumPy/SciPy features, supported subspaces, augmented
least-squares coefficients and forward predictions reproduce529,113predictions
(max1.18e-14), all39overall metric records (max1.22e-15) and6,188strata including
rank metrics. Decision/bootstrap replay passes (max2.16e-16). Optimizer trajectories
and final optimizer/RNG states are retained but not independently replayed.

Local CPU, one Torch/BLAS thread; no Modal, GPU, installation or shared-environment
change. Source admission146.22s/2.185GiB peak polled RSS; data audit5.01s/1.030GiB;
qualification7.78s/.747GiB; allfits19.31s/.846GiB; evaluation19.08s/.679GiB;
prediction replay11.26s/1.028GiB. Each exact fit takes3.08s; REINFORCE4.30–4.34s,
about1.41times slower. Each trains with19,593,000candidate scores including the
500post-update objective evaluations; each RL fit also draws42,560,000identities.
Both score all candidates; no score-saving claim. Timings are single local runs,
not a replicated hardware benchmark. All declared experiment caps pass.

Initial fixture import/lint failures were corrected before source admission;
13focusedtests pass. The first full Hitlist suite had16disk-reserve failures and
3,477passes while source materialization occupied temporary space. Original logs
are retained; disk and memory guards remain unchanged. The unchanged serial rerun passed3,493tests/392warnings in314.19seconds after
materialization finished. The post-rebase full run later had3,481passes and15
setup errors at the unchanged disk-reserve guard (about0.84GiB free versus1GiB
reserve),309.60seconds. This remains unresolved; hosted CI40140838953is running. No scientific gate or outcome was changed.

The post-fit provenance audit finds3testpeptides in prior MAPP,2inHAP1 and2inlymph/DC
cohorts. No prior outcomes or pretrained weights enter this fit. Literal accession
strings across exports differ in format: direct UniProt field extraction finds134
supplied test accessions in the HAP1 reference inventory. Inventory presence does
not prove prior training membership or verified origin. Neither this audit nor the
primary split establishes independent-study/homology generalization.

## Reproduction and artifacts

From the pinned Presto tree and isolated Hitlist runtime, in a fresh output location:

```sh
for stage in admit audit qualify fit evaluate replay; do
  sh experiments/2026-10-24_2244_codex_mapp2025-baseline-prediction/reproduce/launch.sh "$stage"
done
```

The launcher refuses to overwrite receipts; do not delete completed attempts to
rerun them. See exact paths in launch.json, frozen code in reproduce/source and
reproduce/hitlist_source, source-dependent diff, package manifest and input/prediction
seals. Canonical admit/launch scripts received publication-only import ordering;
executed snapshots remain unchanged. Shared original/canonical inputs and recipe outputs are in repo-level
`data/mapp2025_source/`. Generated covariates and separated label JSONs are retained
locally under results and regenerated by admission; their hashes are published.
The original gzipCSV dump is retained locally; published Parquet preserves every
field and double exactly, verified by Arrow equality. Numerical summaries,
checkpoints, gradients, trajectories, environment, closed-out logs and figure are
published here. Affinity/nM/500nM binding metrics are inapplicable to this endpoint.

Status: biological comparison and independent outcome verification complete.
[Recipe PR829](https://github.com/pirl-unc/hitlist/pull/829) passed all hosted unit
(Python 3.10–3.14), integration, lint, docs and coverage checks and merged as
`58f67cd278d75d43432f3c20231de9c21c66ad46`. The original local post-rebase disk-guard
errors remain preserved. Source Hitlist2.0.30 is published; both PyPI distributions
are byte-identical to successful clean-main build40140479531. Recipe2.0.31 release
build40143097533 is pending. See the updated software-publication receipt.

The primary gate fails; the narrower distribution-versus-uniform gate passes.
All four frozen runtime files remained byte-identical after the earlier rebase.
Full APC availability remains unachieved. The next registered PBMC transfer family
2315 fails its minimum component support before scoring, so it adds no external
predictive validation. Hitlist825 continues to own source-reference, physical
precursor pairing and pooled-run recovery requirements.
