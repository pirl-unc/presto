# Post-PR153 unified retraining, comparator, and reporting specification

Date: 2026-11-08  
Owner: Codex / GPT-5  
Status: active on successor branch; PR #153 merged at `5a44a0dd`

## Objective

Create one successor PR that adds a reusable, timestamped performance-report
pipeline and the data/model contracts needed for a fresh unified affinity,
presentation/processing, and assay-aware T-cell fit. Train only from a frozen
successor commit. Evaluate on newly disjoint populations with matched external
comparators: NetMHCpan, NetMHCIIpan, MHCflurry, and PRIME where their input and
license contracts apply.

Evidence quality is a first-class output. Current development cohorts must not
be relabeled as independent tests.

## Current evidence boundary

### Presentation and processing

- The 1915 development comparison scores exactly the same 48,530 rows, HLA
  sets, 15-aa flanks, 200 samples, and three held studies for Presto and
  MHCflurry.
- Presto TRAIN was embargoed against validation by whole study and by exact,
  core-8, and Hamming-1 peptide neighborhoods. Independent cohort checks passed.
- No 1915 confirmation or TEST row was used in fitting or selection.
- The comparison is still development, not an external test: Presto checkpoint
  selection used these three studies, and MHCflurry's full historical training
  lineage is unavailable.
- Preserve the matched result as a positive development signal: Stage B
  processing AUROC/AUPRC 0.90578/0.89572 versus MHCflurry 0.80567/0.79689;
  presentation 0.95205/0.95267 versus 0.94843/0.95428. Do not claim confirmed
  presentation superiority because intervals cross zero and only three studies
  are represented.

### Affinity and NetMHCIIpan

- Current Stage B exact-IC50 development Spearman is 0.66885 for MHC-I and
  0.49949 for MHC-II, below matched NetMHCpan/NetMHCIIpan references
  (approximately 0.75845/0.77786 on currently eligible rows).
- The failed register-temperature audit shows that simply sharpening class-II
  register pooling is not the missing mechanism.
- Class-II evidence is dominated by heterogeneous assay/study sources, has
  weaker exact quantitative support, and has already been inspected through
  repeated development experiments.
- Public NetMHC BA archives are admissible only under their native semantics:
  unspecified-assay IC50 for BA and a distinct ligand/generated-background EL
  output. Their overlap and redistribution license must be resolved before use.
  They cannot create an independent test.

### T-cell and PRIME

- Experiment 1130 prefers the binary-only development arm (functional
  equal-track/equal-study AUROC 0.630998 versus 0.491024 for binary+ordinal), but
  fixed-threshold balanced accuracy is 0.5 and the four tracks contain only
  1/5/8/3 two-class studies. Neither arm is promoted.
- `mhctools 4.9.0` contains a PRIME wrapper and the local PRIME checkout is at
  commit `7b18d4e11042141e7102f7c69be2b0e03d138dab` (PRIME 2.1). PRIME and
  MixMHCpred are academic/non-commercial licensed.
- The installed MixMHCpred is 2.0.2. A fail-closed smoke test proves it cannot
  satisfy PRIME 2.1, which requires MixMHCpred 3.0+. Obtain and pin a licensed
  3.x installation before PRIME scoring. Use explicit executable paths through
  `mhctools.PRIME`; never depend on PATH or a locally modified convenience
  default.
- PRIME applies to class-I/CD8 immunogenicity. It is neither a class-II
  comparator nor a substitute for assay-aware functional outputs.

## Phase 1 — PR #153 review and merge

1. Review the complete nine-commit diff for default compatibility, resume
   fingerprints, selection arithmetic, EMA/checkpoint state, sampler boundaries,
   dependency coverage, and T-cell ordinal inactivity.
2. Verify focused tests and exact-head hosted CI. Fix only evidenced blockers.
3. Record the evidence boundary in the PR and merge using the repository's
   established merge-commit convention.
4. Do not add the approximately 900-MB local experiment 1130 bundle wholesale.
   Preserve it outside Git and carry only lightweight manifests, hashes,
   reports, and scripts.

Review result: no blocking source defect found; 99 focused tests passed and two
independent hosted lint/test and docs runs passed at `85acdcf7`. A local full
rerun was invalidated by host-wide disk exhaustion and is not counted as model
evidence. PR #153 merged as `5a44a0dd50b249d277f13be13517ce8b439badb4`.

## Phase 2 — Successor branch and PR

1. Branch from the verified merged base in a clean worktree.
2. Port this spec and lightweight closures, but no raw licensed or oversized
   artifacts unless the storage policy below admits them.
3. Implement one reusable report API/CLI in the package, not another isolated
   notebook. Add fixture-based tests and deterministic rendering checks.
4. Open a draft successor PR before real model training. Freeze the commit used
   for curation, training, comparator scoring, and report generation.

## Phase 3 — Newly disjoint data contracts

### MHC-I and MHC-II affinity

1. Inventory newly available IEDB/CEDAR quantitative binding studies after the
   current chronology cutoff and any licensed NetMHC training archives. Preserve
   method, detector, preparation, endpoint, units, qualifiers, bounds, source
   study, and exact alpha/beta groove sequences.
2. Keep NetMHC archive BA as `unspecified_assay/IC50`; never merge it into a
   protocol-specific output. Preserve `y=0/1` as censored `>=50000/<=1 nM`.
   Keep archive EL separate from affinity and never treat generated EL
   backgrounds as affinity negatives.
3. Freeze whole-study chronology roles before training. Enforce transitive
   study/group, exact peptide, core-8/core-9, Hamming-1, source-protein, and exact
   MHC-groove embargoes as applicable. Report every overlap census.
4. Reserve a genuinely new TEST population unused for architecture, capacity,
   threshold, early stopping, or calibration. Prefer post-release studies that
   cannot be in NetMHCpan/NetMHCIIpan training. If none exists, disclose that and
   make no independent superiority claim.
5. Require adequate class-II exact quantitative support by allele, study, and
   assay cell. Report low-support cells instead of silently pooling them.

### Presentation and processing

1. Add new whole studies beyond the three exposed 1915 development studies.
2. Preserve exact HLA sets, N/C flanks, protein/sample identity, ligand/control
   pairing, and source prevalence. Give every comparator the same allowed HLA
   candidates.
3. Keep presentation and operational-processing outputs separate. Ligand labels
   alone do not identify cleavage, transport, or detection biology.
4. Score the frozen current Stage B checkpoint on the new cohort before using
   its labels for retraining, creating a valid temporal baseline.

### T-cell

1. Acquire newly disjoint IEDB studies after the frozen T-cell cutoff and retain
   whole publications/submissions as the minimum donor/cohort proxy.
2. Preserve the five fixed assay-aware tracks; add a track only by prospective
   support rule. Never pool multimer binding into cytokine/activation outputs.
3. Require observed assay negatives and mixed-outcome studies in TRAIN,
   development, and TEST. Do not fabricate random-peptide negatives for the
   headline immunogenicity test.
4. Balance batches by track, study, and outcome while retaining every unique
   observation. Freeze train-only prevalence/class weights before development.
5. Select on study-equal binary log loss with a discrimination guard; reject
   all-positive/all-negative threshold behavior. Calibrate on a dedicated
   partition or nested TRAIN folds, never TEST.
6. Keep conditional ordinal intensity as an auxiliary ablation until it improves
   discrimination/calibration on new development and confirms on TEST. The
   failed 1130 ordinal arm remains a negative control.

## Phase 4 — Model hypotheses and ablations

Use one common encoder/checkpoint with assay-aware outputs. Preregister a compact
matrix; do not run an open-ended capacity sweep on exposed data.

1. **Reference:** merged-code reproduction of Stage B plus binary-only T-cell
   head.
2. **Data-only:** reference architecture with newly qualified affinity,
   presentation, and T-cell TRAIN data.
3. **Class-II mechanism:** data-only plus a class-II-specific
   register-marginalized adapter over paired alpha/beta groove interactions.
   Preserve shared biological encoding and output-side assay semantics; no assay
   selector enters the encoder.
4. **Calibration/negative support:** class-II mechanism plus preregistered
   train-only class weighting/calibration for T-cell outputs.
5. **Ordinal ablation:** final binary recipe with versus without conditional
   ordinal loss, holding data and optimization fixed.

For class II, report register posterior entropy/core recovery and gradient/update
coverage in addition to predictive metrics. Require per-cell gains on new
development before accepting greater capacity.

## Phase 5 — Comparator execution

1. Pin exact executable/package versions, hashes, licenses, commands, allele
   mappings, failures, and supported-row census for NetMHCpan, NetMHCIIpan,
   MHCflurry, and PRIME.
2. Use identical eligible rows and biological inputs for each paired comparison.
   Report every excluded row and reason.
3. Run PRIME only after model/threshold/calibration freeze and only on the new
   class-I T-cell TEST. Invoke it through `mhctools.PRIME` with explicit PRIME
   and MixMHCpred 3.x paths. Preserve score and percentile rank separately.
4. Use paired study/peptide-family bootstrap intervals. Report pooled and
   equal-study metrics; the latter is primary.
5. Qualify comparisons across different hardware, package weights, data
   populations, or prevalence contracts.

## Phase 6 — Timestamped performance reports and figures

Implement a deterministic command such as:

```text
presto-report --manifest REPORT_MANIFEST.json --output-root reports/
```

Each `reports/YYYY-MM-DD_HHMM_<git>/` directory must include:

- `README.md`: state of Presto, qualified/unqualified outputs, conclusions,
  limitations, and next decisions;
- `report.json`: schema version, supplied UTC timestamp, Git commit/dirty status,
  model/dependency hashes, training recipe, runtime/hardware, sources, and every
  metric/interval;
- `data_manifest.json`: every TRAIN/calibration/development/TEST source, license,
  row count, curation/split policy, content hash, and artifact location;
- `predictions/`: per-example dumps or immutable external references with hashes;
- `figures/` and `tables/`: deterministic PNG/SVG figures and CSV/JSON summaries;
- `reproduce/`: exact frozen command, environment, and source snapshot.

Start from the matched-row and uncertainty logic in the existing MHCflurry
evaluation, but generalize it. Required figure families are:

1. task-level comparator overview with confidence intervals;
2. MHC-I/MHC-II exact-affinity scatter, residual, and calibration panels;
3. `<=500 nM` ROC/PR and threshold metrics by class and assay cell;
4. presentation/processing paired-study forest and ROC/PR panels;
5. T-cell pooled/equal-study ROC/PR, reliability, distribution, and per-track
   forest panels;
6. split/data composition and overlap-embargo audit panels;
7. runtime, memory, and model-size table.

Figure code must consume only versioned manifests/prediction dumps and never
query mutable external state. Test row alignment, metrics, schema, caller-supplied
timestamps, filenames, and rerun hashes with deterministic fixtures.

## Data storage policy

"Save all training/test data" means every exact row remains recoverable, not that
all bytes belong in ordinary Git.

- Commit redistributable small manifests, decisions, summaries, and scripts.
- Store large frozen rows, checkpoints, and predictions in a canonical Modal
  volume or approved immutable object store. Record byte count, SHA-256, and
  immutable location in each report.
- Do not commit academic/non-commercial binaries or raw archives whose
  redistribution rights are unresolved.
- A report is invalid if a referenced artifact cannot be retrieved and verified.

## Promotion gates

No final unified model is declared unless one checkpoint, without post-test
tuning:

1. exceeds matched NetMHCpan on new MHC-I exact affinity and threshold metrics;
2. exceeds matched NetMHCIIpan on new MHC-II exact affinity and threshold metrics;
3. exceeds MHCflurry on both presentation and operational processing with
   positive paired intervals on the primary equal-study metrics;
4. exceeds PRIME on the new class-I T-cell TEST primary metric while passing
   calibration and non-degenerate threshold gates;
5. preserves assay-aware output identity and passes full coverage, resume,
   artifact, leakage, and reproducibility verification.
