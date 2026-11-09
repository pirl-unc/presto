# Shared measurement curation and source recovery

Date 2026-10-29; Codex/GPT-6. Family 2026-10-29_1653_codex_measurement-curation-recovery.

## Objective
Implement the user's requested source corrections and upper/lower limit handling,
recover supplementary MS metadata/data, and acquire a traceable inventory of
available pMHC binding, stability, cellular-binding and ligand datasets. Preserve
the single observation/training path. This is implementation plus descriptive
curation validation, not a new architecture or predictive benchmark.

## Plan before implementation
1. Trace raw IEDB method tags through Hitlist to Presto; create focused upstream
   issues for incorrect/ambiguous source method semantics, linking #721. Fix any
   consumer claim that literal KD establishes direct equilibrium measurement.
2. Extend common curation with original/effective qualifier and evidence state.
   Parse explicit inequalities from structured and numeric-text fields; retain
   conflicts. Screen repeated floor and ceiling values within study, MHC, method,
   quantity and units. No magic universal 1/20k/25k/50k rule. Distinguish explicit,
   source-confirmed, conservatively inferred and unresolved censoring. Preserve
   exact source identity and frozen rule version. Validate both orientations,
   unknown qualifiers, conflicts, unit conversions and unchanged raw records.
   Integrate the process into common source admission/preparation, not only a
   one-off analysis. Strong heuristic decisions must be identified as inferred;
   unresolved candidates cannot silently become confirmed exact/bound labels.
3. Fix supplementary PTM transport at its owning library in an isolated checkout;
   preserve explicit False/True/unknown and modifications/row locators through
   exports. Review original supplementary sources for actual modification/filter
   evidence; use evidence-backed file/row-scoped declarations. Do not globally
   replace unknown with False or equate alphabetic strings with known chemistry.
   Recover qualified source rows and quantify remaining exclusions separately.
4. Acquire/index public source archives: current IEDB/CEDAR already downloaded;
   Hitlist supplementary registry/original sources; published predictor training
   archives and additional primary binding/stability/cellular datasets found by
   a literature/catalog search. Track URL/date/hash/license/access, measured versus
   predicted labels, method/quantity/censor/PTM fields and overlap with existing
   IEDB accession/study/pMHC identities. Downloaded duplicates are not new labels.
   Preserve unavailable/restricted sources explicitly; no author outreach.
5. Freeze a new curation receipt/cohort delta and verify with focused regression
   tests plus relevant suite. Keep the current temporal fits/artifacts immutable;
   preserve existing study holdouts and report any new admission by source/type.
   No inference thresholds or curation overrides tuned on model/test performance.
   Any later predictive retraining needs its own frozen contract and registry.

## Execution and verification
Use CPU/Modal for large source scans; no GPU required for curation. No pretraining,
fit, loss weights, synthetic labels or held-out predictive metrics in this family.
Audit code/invocation/git state, original/effective labels, per-row source decisions,
source inventory and exact qualified counts must be preserved. Reproduce unit
fixtures before full source application. Review source-library AGENTS before edits.
Check active temporal collector and retain its independent validation/test gate.
Success means working common code, real source-backed recovery, actionable upstream
issues/patches and reconciled data—not merely a list of potential extra rows.

## User clarification: local thresholds before shared evidence
Keep decision scope exactly reference/study × reported MHC × assay method × literal
quantity (units normalized only for comparison). Repeated panels within one linked
study component are not independent replications. Add a descriptive shared-limit
summary requiring at least three independent study components, explicit same-limit
bounds in at least two, at least 100 affected observations and >=95% directional
agreement. A shared-rule candidate additionally requires a verified protocol-family
mapping; unreviewed detector tags are insufficient. This stronger evidence screen
never creates row decisions automatically. Any transfer still requires a cited,
source-fingerprinted review. Publish all 71 local panels, including the 618 withheld
rows, with original reference/PMID, method, quantity, MHC, original qualifier counts
and directional evidence; distinguish observations from unique peptide–MHCs.

## Continued source reconciliation while the frozen retrain runs

Review the downloaded NetMHCstabpan stability archive against its primary paper
and the curated TRAIN physical observations. Count literal zero values, positive
half-lives, allele coverage, exact peptide–MHC overlap and unit-normalized repeated
half-life values using canonical MHC nomenclature. Keep predictor-generated
augmentation separate from reported measurements; zero hours must not become an
arbitrary log-space floor. Preserve hashes and source locators, publish an admission
decision, and do not mutate the current retrain or claim all non-overlaps independent.

Stability inventory encountered a reported mutant MHC identifier that the canonical
parser does not resolve. Preserve the failed inventory. Count unresolved constructs
separately by their exact source identifier; do not strip the mutation or match them
to wild-type data. This source-only census must tolerate and report rejection causes.

More than two hours have elapsed since the serial public PSM retrieval received
persistent HTTP 429s (recorded Retry-After was respected). Make one new serial
retrieval pass using the same authorized public endpoint/client, five seconds
between files, skipping all previously verified successes. Preserve a new receipt
and stop on the first new 429 rather than repeatedly retrying. Any recovered files
belong to a future source-qualified cohort; current training inputs remain frozen.

The bounded retry obtained nine files, then stopped at HTTP 429: 20/132 selected
files now available, 112 remain. Freeze the union of successful source receipts
as batch `late`; recover forms on Modal CPU using the same published-ligand and
explicit-chemistry contract. Write to a new output directory and preserve the
original recovery. Verify all original recovered pMHC identities remain present,
then report the source-qualified delta separately from model admission and training.

After publication of the verified Hitlist 2.1.3 artifacts, admit the late batch
through the same Hitlist export and common Presto MS adapter, with a new immutable
output/call/receipt namespace. Preserve the initial batch and development-wheel
pin. Expand the candidate embargo to all original heldout files, including
generated comparisons, and retain explicit study/year provenance. Report actual
length/restriction/overlap dispositions and installed dependency versions. These
remain candidates for a subsequent frozen background/cohort audit; active training
and its original inputs are unchanged.

The late admission's first attempt wrote its candidate outputs but failed while
recording a version for `mhcseqs`, which is not installed in this reader image:
the reader consumes the already pinned mhcseqs catalogue artifact. Preserve that
attempt and partial volume directory; rerun as `late-v2`, recording actual runtime
dependencies and catalogue-provider provenance separately. No partial candidate
output is promoted and no active model input changes.

After successful late-v2 admission, independently hash the collected archive and
every member; check positive-only observations against the reviewed source forms,
MHC class/length, provenance and study fields. Audit all reviewed positive sequences
(including the short excluded peptides) against the frozen current generated
TRAIN/validation/test comparisons and the original blacklist. Verify no candidate
overlaps any fixed holdout peptide and calculate the exact retained/new identities
relative to the recovered positives already in the active fits. This read-only CPU
audit records conflicts for the next cohort; it cannot alter the active manifests,
labels or prespecified comparison/sensitivity denominators.
