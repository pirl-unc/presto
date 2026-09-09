# Real-source endpoint coverage and update audit (#48/#50)

Agent/model: Codex / GPT-6. Status: planned; not launched. Prerequisites are
merged: row/panel exports in PR #57 and prospective coverage in PR #58. The
evidence branch starts from merged main `d9a666a07085289b168105089d29dd4e675add61`.
The implementation specification is in `tasks/todo.md`.

## Question and decisions

Which declared endpoint/column has effective measured, proxy, auxiliary or
generated-label support in each split, and which supported columns receive
gradients and actual parameter updates on real training batches? Use the result
to select a prospective supported subset for #53, not to claim predictive
quality. Count #50's affected T-cell pathway-MIL source observations and every
selected axis/column explicitly.

## Source contracts

Audit the canonical default merged path, exclusive Hitlist path and explicit
bulk-MS supplement separately. Preserve their source-specific loader choices
even when competing files are present. No implicit union or fallback is allowed.
The current candidate merged file is `data/merged_deduped.tsv` (1,028,889,889
bytes at planning time); freeze its SHA-256 and the associated funnel artifact
before execution. Hitlist currently uses `/Users/iskander/.hitlist`; inventory
the actual selected datasets and underlying artifacts, with hashes after load.
Do not substitute the earlier capped PR #45/49 runs for this corpus census.

Record both the measured curated base and the effective training population
with canonical generated-label families. Pin the current default ratios
explicitly in the launch bundle; do not inherit them from a future parser.
Use no modality record caps for the census. Freeze data seed separately from
split/model seeds, mask unresolved source context, require class-correct complete
MHC resolution, and retain every qualifier/family policy and exclusion reason.
Use peptide-disjoint train/validation/test splits with fixed explicit fractions;
record cross-source duplicates and any generated-parent relationships separately
from effective observation multiplicity. Missing parent provenance is a finding,
not evidence of absence of leakage. #53's fitting subset must come from train.

At planning time the shared environment reports hitlist 1.55.8, mhcseqs 2.5.12,
mhcgnomes 3.41.0, torch 2.7.0 and runplz 3.24.31. These are observations, not a
frozen environment. Use an isolated environment/image and freeze exact installed
versions plus editable source commits/dirty state and relevant parser source
hashes at launch. User-published runplz 4.4.4 may be used when pinned; use a
supported existing Modal launcher for detached jobs if its capabilities require
that route. Do not upgrade the shared environment used by other projects.

## Census and gradient conditions

- Record raw loader/curation counts and effective post-split observations.
  Count source rows, unique source observations, peptides, original/resolved
  alleles, class/species/context, exact/censored labels and response balance by
  canonical endpoint/column and source/evidence family. Keep aliases and repeated
  objective terms visible without inflating independent endpoint counts.
- Track generated wrong-enzyme bulk-MS labels separately from observed peptides;
  retain depth-derived detectability's proxy status. Keep generic KD supervision
  from non-KD evidence separate from exact KD measurements. Organism-derived
  foreignness and input-derived identity labels are not measured immune outcomes.
- Retain zero support for all declared configurations. Compare expanded and
  collapsed topology and supported residual/grouping output configurations on
  the same frozen target population; report absent/no-loss leaves explicitly.
- Select a deterministic, modality-stratified subset of real training examples
  for bounded gradient/update diagnostics. Freeze IDs before optimization. Use
  the canonical d128/l2/h4 model, fresh weights and explicit optimizer/loss
  weights, including uncertainty and alias multiplicity. Record initialization,
  trainability/stage, observed label support, output gradients, parameter-row
  gradients and optimizer updates separately. Include zero-initialized/frozen
  and unused-column controls without representing them as predictive results.
- Record full counts even when diagnostic updates cover only a bounded subset.
  A column omitted from the gradient subset is untested, not verified inactive.
  Preserve counts of optimizer-state/decay updates without direct gradients;
  shared downstream gradients cannot satisfy direct-label requirements.

The exact gradient-step budget, frozen subset IDs, configurations and optimizer
settings must be written into the timestamped experiment bundle before launch.
The census has no model fitting. Gradient updates are mechanism diagnostics,
not an adequate training budget or a replacement for #53's fitting/baseline work.

## Hardware, artifacts and closure

The user explicitly requests upstream issue filing for data/curation problems.
File confirmed Hitlist defects in `pirl-unc/hitlist` with exact source versions,
row/study identifiers, impact counts, expected/observed behavior and reproducible
commands. Check existing issues first and attach additional evidence there when
the defect is already tracked. Link every finding in the experiment README and
PR. Presto adapter defects belong in Presto; do not silently alter shared input
caches to conceal either category.

Existing Hitlist #444 reports that `exclude_from_ms` has no reader and non-MS
studies enter the observations corpus. Check the actual frozen cache/curation
version for this condition before interpreting its elution labels. The issue's
reported counts are not results of this experiment. Upstream #361 already tracks
unobserved bulk candidate generation; do not file a duplicate for that limitation.

Use CPU for source preparation/census. If gradient diagnostics use Modal,
request `PRESTO_MODAL_GPU=H100!` explicitly and save observed GPU/memory evidence.
Use a Volume for large raw artifacts and a durable job handle for detached work.

Create one timestamped `experiments/YYYY-MM-DD_HHMM_codex_output-coverage/`
directory with README, `reproduce/launch.sh`, `reproduce/launch.json`, frozen
launcher source, code/dependency/source/curation hashes, raw artifact references,
condition manifest and launch logs. Preserve complete JSON/CSV coverage matrices,
prospective support assessments, loss multiplicity/weights, per-column update
records, measured/generated source-role tables and the selected real subset IDs.
Keep exact distinct-count storage disk-backed for the uncapped corpus.

No validation/test predictive metrics are claimed by the census or the bounded
gradient audit. State that explicitly in the experiment README and canonical log;
their validation/test counts establish eligibility only. If any evaluation is
performed, preserve the full per-example predictions and required split metrics.

Close every completed or materially informative condition: extract all metrics
and counts, reconcile summary tables with raw artifacts, document failures and
unsupported columns, update the README and `experiments/experiment_log.md`, and
link the evidence to #48/#50. Freeze #53's supported endpoints and success/failure
criteria before its fitting or model-selection runs. Do not close #48/#50 on
fixture coverage alone.
