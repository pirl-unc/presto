# Affinity measurements, optimization, ranking and shared geometry

Date: 2026-10-29. Agent/model: Codex / GPT-6.
Family: `2026-10-29_1318_codex_affinity-objectives-geometry`.
Status: specification before implementation or experimental launch.

## Questions and protected baseline

The user requests ranking-loss and shared-geometry experiments, modern training
practices, an affinity-data census against established predictors, and agreement
between assay types on identical pMHCs. Previously quoted rho values are exact
IC50 only, separately by class. Sampling, loss normalization, shared-parameter
gradient conflict, checkpoint selection and a changing curriculum are distinct.
The temporal recipe has stationary sampling; gradient conflict is unmeasured.

Keep both active temporal fits and their source/test/runtime contracts unchanged.
Complete their existing collection/verification pipeline. At most two H100! GPU
workers concurrently across existing and follow-up fits. CPU audits can proceed.
Preserve the shared environment and original dirty checkout. PR131 stays draft.

## Phase A: source and measurement audit

Use corrected `kinetic-review-v2` inputs and their frozen manifest hashes. Reconcile
endpoint/class counts before/after admission, holdouts and peptide embargoes.
Compare published/released dataset counts using explicit denominators: quantitative,
qualitative, converted and synthetic labels; species; assay semantics; version.
Do not relabel processed predictor training data as literal IC50 without evidence.

Measurement agreement uses TRAIN observations only; no model/test predictions.
Pin and hash inputs. A pMHC is exact canonical peptide plus both verified groove
sequences plus class; require a single resolved MHC candidate. Preserve allele
pair provenance. Use exact positive finite concentration measurements in log10(nM)
under the verified unit contract. Exclude bounds from ordinary correlations and
count their exclusions separately. Keep all five concentration outputs distinct.

For every output pair, report I/II separately, preserving paired aggregate data
with hashed pMHC IDs for independent recomputation:
- source-row, unique pMHC, exact overlap and study counts;
- within-study comparisons using median log values per pMHC/endpoint/component;
- pooled comparisons: aggregate within study, then median across studies;
- disjoint-study-only comparisons where endpoint study sets do not intersect;
- Spearman, Pearson, RMSE difference, median signed difference, identical-value
  fraction and explicit unavailable/undefined cells;
- within-endpoint repeat disagreement, assay method/response/unit census.
Never create many-to-many Cartesian replicate pairs. A shared study does not
prove matched assay conditions, independent outcomes or assay interchangeability.
Agreement between labels does not establish or exclude neural gradient conflict.

Inventory source-attested matched panels for ranking. Common paper/allele/method
is a candidate group, not a verified matched-condition panel. Freeze qualified
membership/provenance before ranking; quantify unsupported candidates. Do not
invent cropped-peptide affinities or cross-lab pair constraints.

Phase A follow-up, specified after the first audit: the IC50/KD(~IC50) overlap
has unexpectedly high agreement and is concentrated in a few study pairs. Run a
CPU-only lineage census on the same frozen TRAIN source. Preserve study aliases,
reference titles/IDs, method/response counts and common curator comments, plus
unsequenced original assay IDs and values for the already defined overlap pMHCs.
Check study concentration, exact/near-identical values and source-attested reuse.
Different accession IDs or study components do not establish independent assays.
Do not merge observations or endpoints solely because their values agree.

## Phase B: common implementation and diagnostics

Add schedules to the common trainer with exact resume behavior: short warmup and
cosine decay for a fixed budget, with an explicit constant-rate control. Consider
warmup-stable-decay only for flexible-duration comparisons. Keep AdamW and matched
clipping across compared conditions; record learning rates, unclipped norms and
clipping frequency. Language-model hyperparameters are precedents, not a proven
pMHC optimum. Test interrupted/uninterrupted weights/optimizer/order/coverage parity.

Measure gradient norms/cosines for endpoint/class losses on fixed TRAIN probes,
using shared parameters, controlled dropout and no optimizer update. Repeat across
probes/checkpoints and retain distributions. This is local interference, not proof
of a generalization penalty. Isolate sampling changes; gradient projection is
conditional on measurements, not an automatic extra mechanism.

Add an optional generic within-panel ranking term alongside absolute/censor loss.
Require same endpoint, MHC pair and verified comparison group. Censored pairs
are eligible only when intervals imply ordering; exclude ties/ambiguous intervals.
Normalize pair/group contributions explicitly and record realized pair exposure.
Rank-on/off controls use identical paired batches. Preserve separate loss terms;
default-off reproduces the existing trainer exactly.

Geometry candidates use the same interaction for I/II: peptide N/C distances,
groove-half/within-half coordinates, plus a separate register-temperature ablation.
Relative sequence position is not verified structural alignment. No class expert,
independent register head or unsupported physical-occupancy claim. Record feature
and coordinate coverage and register entropy. Test geometry and pooling separately.

## Phase C: freeze exact matrix after source audit

Intended two-seed matrix: constant-rate baseline; scheduled baseline; scheduled
paired-batch rank-off; matched rank-on; scheduled geometry-only; scheduled
register-temperature-only. Combine ranking/geometry only after separate evidence.
Freeze actual cohort, budgets, lambda, schedule, geometry and temperature before
launch. Match common initialized weights, exposure budgets, validation cadence
and selection criterion where dimensions permit; report any remaining differences.

Choose study-disjoint partitions before supervised initialization. Never initialize
a fold from weights trained on its holdout labels. A reused temporal validation
pilot must be labeled development evaluation. Preserve closed tests; freeze a new
test before fitting and open only after all fixed fits and validation selection.
Document any staging choice before launch, not after test inspection.

Prespecify macro class IC50 improvement, per-class protection, calibration and
ligand-preservation gates with numerical tolerances and paired uncertainty before
launch. Report other assay outputs and sparse/absent output/class cells explicitly.

## Verification and closure

Register launched stages. Save exact commands/env, source/commit/dirty state,
data hashes, paired assay aggregates, per-example validation/test predictions,
all assay/class/study/threshold metrics, comparisons, runtime and H100 evidence.
Independently recompute correlations and predictive metrics; replay TRAIN serving
probes. Integrate qualified wins only into the common model/trainer. Close README,
canonical log, task records and recipe. Launching alone is not completion.

## Primary methodological precedents

- Warmup, NeurIPS2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/ca98452d4e9ecbc18c40da2aa0da8b98-Paper-Conference.pdf
- Cosine/cooldown, NeurIPS2024: https://arxiv.org/abs/2405.18392
- ModernBERT, ACL2025: https://aclanthology.org/2025.acl-long.127/
- Conflict/generalization tradeoff, JMLR2024: https://jmlr.org/beta/papers/v25/23-1287.html
- pMHC ranking: https://pmc.ncbi.nlm.nih.gov/articles/PMC8165219/
