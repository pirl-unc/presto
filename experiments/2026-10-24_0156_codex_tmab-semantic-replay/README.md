# Corrected TMAB source semantics: full real-source replay

Codex / GPT-6, registered October 24, 2026 at 01:56 America/New_York.
Presto source b298ea991f7b1ed603ab0c5d700f6b8c3e827107 with recorded task dirt.
Hitlist f4d3a6d0f1933162f9ac6eac42ac1b134c8a27dc plus the exact frozen
correction diff and file snapshots in reproduce/. Execution version 2.0.16,
revision tmab_native_peptidomics_v2, package schema 16 / catalog 21. This is a
local correction build; merged release delivery is tracked separately.

## Result and scope

The shared public writer and EvidenceStore preserve all original source rows,
measurement IDs, literal row arrays, ratio values, qualifiers, missing/overlap/error
states and roles. Independent whole-string grammar checks every projected
sequence, supported modification position, identification status and source
eligibility. Candidate reuse memberships match the independently decoded original
workbooks. No Presto source repair, shared environment change or model fitting.

| Check | Result |
|---|---:|
| Source rows preserved | 10,564 |
| Measurement slots preserved | 61,908 |
| Cross-paper candidate row groups | 830 |
| Rows in candidate groups | 1,660 |
| Groups with normalized matching experiment labels | 552 |
| Groups with different experiment labels | 278 |
| Exact treated/control source candidates | 7,933 |
| Censored treated/control source candidates | 4,662 |
| Unexplained disagreements after comparator correction | 0 |

These are source candidates, not independent training examples. Controls and
source-derived averages are deferred. Unknown/tentative/unresolved identities
remain queryable; no identity is inferred from numeric exactness. Acetylation and
methionine oxidation survive normalization. Unsupported pyroE and dehydration
annotations remain unresolved. All 173 Gelman heatmap projections use the original
sequence column; heatmaps remain deferred republications. Cross-paper matches
are possible reuse, not specimen adjudication; all specimen independence and
physical control denominators remain unknown.

Publishing-study instruments and original acquisition scope are distinct. New
2014 panels use the source G2 protocol; reused acquisitions remain unresolved.
Incubation and harvest annotations remain separate from source total exposure.
The 2014 inhibitor-present wash is not relabeled washout. Every protocol context
and new measurement/lineage field is exercised through the public query surface.

## Preserved failed execution and corrected comparison

The original producer completed 2,506,129 assertions, then
failed with 1,242 context-label comparisons: projected context labels trim trailing
whitespace, while raw source arrays retain it. This behavior already existed in
v1. The first launcher, failure log, disagreements and replay.json remain unchanged.
The separately frozen code/close.py verifies all 1,242 differences against both
versions and independently decoded raw rows; every other check already passed.
No production code, source value or parser rule changed for this correction.
See reproduce/comparison-correction.md and results/closure.json.

## Dataset, training and evaluation contract

Same four hash-pinned public Dasgupta 2014 / Gelman 2013 primary and heatmap
workbooks as family 0117, native human HEK293T/SH-SY5Y whole-cell peptide extracts.
Exact input paths and source manifests are in reproduce/launch.json and the
copied old/new manifests. Parent references, charge-state contributors, sample/run/
channel identities, shared controls and original 2012 acquisition lineage still
require recipe-level qualification. These isotope-channel ratios are neither
absolute concentrations nor MHC binding, elution or APC-specific measurements.

No pretraining, training parameters, optimizer, synthetic biological labels, loss
terms, assay heads, predictive validation/test split or prediction dump applies:
this is source qualification before training admission. Full semantic changes,
query outputs and candidate memberships are retained instead. No new predictive
performance or improvement over the stopped MAPP transfer experiment is claimed.

## Reproduction and resources

Run reproduce/launch.sh using the exact pinned Hitlist base with the frozen patch
and listed source hashes. It refuses to overwrite the immutable package or a
completed replay. Original execution intentionally reproduces the recorded
whitespace-comparator failure; then run code/close.py under the same one-thread
environment to reproduce corrected closure. Do not replace original artifacts.

- Local CPU, Python 3.12.6; OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1.
- No Modal/GPU, installations or shared-environment maintenance.
- Build/query/replay: 38.99 s, 1152.34 MiB peak RSS;
  prespecified cap 180 s / 2 GiB.
- Separate closure: 2.49 s, 554.45 MiB;
  cap 30 s / 1 GiB.
- 43 focused Hitlist tests pass. The regression file gives 14 failures / 1 pass
  against pinned old code. Full ./test.sh passes: 3,403 tests in 723.92 s, maximum process RSS 954,269,696 bytes.
  Hosted/release gates remain separate; source replay does not substitute for them.
- New package manifest SHA-256: `f84bd8e738a2c14d446956c719cfeaba91bc867a95aebd62dac1c91d3e3a1666`.
- Raw package: `/private/tmp/presto-modular-model/data/native_peptidome_source_audit/2026-10-24_0117_codex_native-peptidome-admission/2026-10-24_0156_codex_tmab-semantic-replay-package`.
- Prior source evidence: ../2026-10-24_0117_codex_native-peptidome-admission.
- results/ contains coverage/manifests, all changed source rows/measurements,
  public query outputs, context assertions, candidate memberships and closure.

## Decision

Source semantic correction passes after the disclosed verifier correction.
Proceed to a canonical shared native-response recipe that groups reuse/charge/
shared-control dependence and retains exact versus censored labels. Do not call
these source candidates independent samples or concentrations. Hitlist #791 and
implementation owner #792 remain the upstream work items. The full APC availability
objective remains active; biological prediction is the next acceptance target.
