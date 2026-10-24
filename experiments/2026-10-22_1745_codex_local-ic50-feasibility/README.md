# Local CPU / Apple GPU training feasibility

**Completed and independently verified, October 22, 2026. Codex / GPT-6.**
Local training works on this M1 Pro. One complete epoch of the unchanged modular
IC50 sequence model takes 169.97 seconds including evaluation, with 2.59 GiB peak
process RSS. It improves held-out RMSE and ranking over the fixed composition ridge
baseline, has nonconstant predictions, and updates all 32 allocated parameter
tensors. This is one seed and one epoch, not a converged or biologically qualified
availability model.

## Hardware comparison

Apple M1 Pro: 10 CPU cores, 16 GPU cores, 32 GiB unified memory. Python3.12.6,
PyTorch2.14.1, NumPy2.5.4, macOS26.5 arm64. Four CPU/intra-op threads, one inter-op
thread. MPS is available outside the sandbox; a sandboxed availability check
returned false. No Modal, CUDA, mixed precision, compilation, package installation,
or CPU fallback from MPS. Shared environment was not modified.

| Backend | Median update | Updates/s | Observations/s | Measured update range |
|---|---:|---:|---:|---:|
| CPU, four threads | 0.4274 s | 2.340 | 149.74 | 0.340–0.487 s |
| Apple GPU / MPS | 0.4866 s | 2.055 | 131.53 | 0.416–0.598 s |

Same seed42 initialization and same shuffled 12 batches of64 admitted training
observations; first4updates warmup, next8measured. Backend RNG implementations
produce different dropout masks. MPS synchronized before and after each update.
Both use the complete canonical train_step, including gradient/update audits;
input loading and initial setup are outside per-update timing. CPU then MPS were
run sequentially, not concurrently. Raw step records preserve batch identities,
losses, memory, and timing. Initial/final checkpoints are retained for both.

CPU benchmark peak process RSS2.27GiB. MPS peak process RSS0.59GiB and maximum
post-step driver allocation about2.05GiB are distinct measurements of unified
memory; neither is a measured transient GPU-allocation peak and they must not be
compared as equivalent footprints. Full CPU fit peaks at2.59GiB.

**Interpretation:** this short comparison favors CPU by about14% in elapsed update
time. It is preliminary, not a replicated hardware ranking. The small455,041-
parameter model uses Python loops per observation/register, and per-tensor scalar
reads in coverage checks force GPU synchronization. These are plausible overhead
sources identified in code, not a completed profiler attribution. Larger batches,
batched fusion, and fewer synchronization points remain untested. Choose CPU for
this diagnostic using timing alone; no validation/test score influenced the choice.

## Frozen data and architecture

Admitted original manifest:
`data/modular_cohorts/2026-10-21-full-ic50/manifest-exact.json`, SHA256
`914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be`.
The [admission audit](../2026-10-21_0647_codex_full-ic50-admission/README.md) documents
source qualification. The executed [local manifest](reproduce/local-manifest.json)
retains the individually pinned normalized files and changes only the execution
schedule/path representation.

- Training: all21,636 exact purified human HLA IC50 measurements,14,375classI and
  7,261classII. No cap, averaging, label clipping or generated controls.
- Validation:2,925 observations,2,726 exact and199 bounds.
- Test:2,803 observations,2,600 exact and203 bounds.
- Original peptide-disjoint partitions and historical exposure retained; studies
  overlap and donor/specimen/acquisition independence is not established. These
  are historically inspected held-outs, not a newly blind test.
- Inputs: peptide plus two MHC groove sequences; supplied human molecular species
  and MHC class. Shared d128/l2/h4 sequence encoder, classI full peptide, explicit
  classII nine-residue registers with flanks and learned pooling, direct IC50 head.
- One fresh seed42 fit; AdamW lr3e-4,weight decay1e-4,batch64,dropout0.1,constantLR,
  one complete epoch339updates. All455,041parameters trainable. No pretraining,
  auxiliary labels/replay, APC, MS or TCR branches, scheduler or variance penalty.
  Disabled modules are absent; this does not train the entire holistic architecture.
- IC50 maps to a single scalar in log10(nM), standardized with train-only exact
  mean/SD. Weight1 squared error; evaluated bounds use one-sided squared error.
  Other assay families excluded. No synthetic data. The single epoch is selected;
  canonical selected-checkpoint evaluation runs on all three splits.
- Mean baseline: exact training log10(nM) mean. Ridge: length,20residue fractions,
  two class indicators; train-only centering/scaling, alpha1, unpenalized intercept.
  Neither baseline uses MHC sequence or allele identity, so this is a basic control.

## Predictive outcomes

Exact-value RMSE is in log10(nM). Undefined constant-baseline correlations are
shown as a dash. Full metrics, class strata and threshold measures are in
[metrics.csv](results/metrics.csv) and [independent verification](results/independent-verification.json).

| Condition | Train RMSE | Validation RMSE | Validation Spearman | Test RMSE | Test Spearman | Test AUROC at500nM |
|---|---:|---:|---:|---:|---:|---:|
| mean | 1.3435 | 1.3606 | — | 1.3420 | — | 0.5000 |
| ridge | 1.2023 | 1.2235 | 0.4581 | 1.2291 | 0.4306 | 0.7067 |
| network | 1.1129 | 1.1431 | 0.5521 | 1.1613 | 0.5117 | 0.7371 |

Neural selected-model standardized loss: train0.68624, validation0.75106,
test0.76238 (bounds included where present). Online mean minibatch training
loss0.76685 is a different statistic, averaging predictions at changing weights.
Test prediction SD0.69486 versus targetSD1.34194; SD ratio0.5178. Test predictions
range1.00468–4.52838log10(nM), clearly nonconstant.

| Neural test stratum | Exact n | RMSE | Spearman | Binder recall | Binder F1 |
|---|---:|---:|---:|---:|---:|
| ClassI |1758|1.2791|0.4141|0.6801|0.6812|
| ClassII |842|0.8650|0.4223|0.0882|0.1390|
| Overall |2600|1.1613|0.5117|0.5791|0.6184|

The neural result improves test RMSE by0.06782log10(nM), about5.5%, over ridge.
It does not win every endpoint: overall binder F1 is0.6184 versus ridge0.6199,
and recall0.5791 versus0.6142. ClassII binder recall remains poor. Only2/199
validation bounds and0/203 test bounds are satisfied; mean and ridge have the same
bound-satisfaction counts. These exact-only, single-epoch fits leave censored
observations poorly modeled. No recipe was tuned from these test outcomes.

**Decision:** local execution is viable and this architecture can learn varying
outputs on the admitted cohort. Continue a separately registered convergence/
replication comparison using local compute. Do not claim the older underfitting
cause is solved: cohort, architecture, and evaluation contract differ from the
older9,053-row run. This does not measure antigen-processing availability or
intracellular concentration, and does not admit the pending MAPP sources.

## Verification, reproducibility and artifacts

- All82,092 prediction records (three methods ×27,364observations) were independently
  joined to source IDs, raw targets, qualifiers, peptide, group and MHC class.
  NumPy/SciPy/scikit-learn recomputation reproduces losses, correlations, RMSE,
  class metrics and500nM metrics. Ridge coefficients independently refit through
  augmented least squares agree. Fresh-process selected-checkpoint replay on the
  first64training observations has zero prediction difference; test not rescored.
- All32allocated parameter tensors changed and received real gradient-driven
  updates; feature coverage passed with no exposed feature lacking updates.
- The initial verifier exposed a metric bug: a constant floating-point vector
  can have tiny positive computed SD. The old correlation guard emitted seven
  meaningless near-zero Pearson coefficients for the mean baseline. Original
  outputs and failed verifier are preserved under `attempts/`; the independent
  report explicitly records seven corrections to null. The canonical guard now
  tests zero range. Both regression cases failed before the fix;35training/resume
  tests pass afterward. No training weights, prediction, checkpoint selection,
  reported nonconstant correlation, RMSE or runtime changed.
- Runtime:12.55sCPU benchmark,16.11sMPS benchmark including setup;169.97sfit including
  evaluation;183.76sfit wrapper including5.62sbaselines and other setup/copying.
  Independent verification4.84s. No GPU requested/observed for the full fit.
- Launch commit `ef4b4c1a51e74add1557b8aa5fcc8d95a84309c5`, dirty documentation and
  registered experiment files; runtime implementation hashes and full snapshots
  preserve the exact code. The later correlation fix is recorded separately.
- Reproduce with [launch.sh](reproduce/launch.sh); [launch.json](reproduce/launch.json)
  indexes exact per-action command/env/git receipts. Four-thread overrides,
  inter-op1, `PYTORCH_ENABLE_MPS_FALLBACK=0`, import shim and Python path are frozen.
  Use a fresh output root; launcher refuses to overwrite completed experiments.
- Raw checkpoints, resume state, evaluation receipts, baseline outputs:
  `artifacts/2026-10-22_1745_codex_local-ic50-feasibility/`. Full fit subdirectory
  `one-epoch-cpu/` contains `initial.pt`, `selected.pt`, `last.pt`, `resume.pt`.
  Local checkpoints are intentionally not committed; hashes/sizes and paths are
  in [execution.json](results/execution.json). Shared source data remain under
  `data/modular_cohorts/`; portable Git artifacts include compressed prediction
  dumps for all methods/splits, metrics, code, hashes, and logs.

Status: complete local feasibility diagnostic; broader availability goal active.
Next step: local convergence/replication and profiling; finish separate biological
source qualification before exact/RL availability fitting.
Open questions: GPU synchronization cost, larger-batch throughput, multi-epoch
learning, classII sensitivity, censored-label training and independent-study transfer.
