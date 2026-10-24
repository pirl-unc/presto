# MAPP deposited-series and processed-evidence admission audit

October 22, 2026; Codex / GPT-6. Extend the verified recovery of PXD009092
SEARCH.7z at Presto30cc536. The archive is intact but source membership is not
qualified. Preflight summary labels suggest protocol-validation controls; do not
assume these are cytokine-response samples or combine them with another deposit.

## Scientific question and immutable source scope

Which study experiment does each of the seven public PRIDE datasets represent,
and can the recovered processed peptide measurements support a biologically
interpretable relative-availability endpoint? Audit source assertions before
training or selecting favorable examples. Retrieve project/file metadata for
PXD009092, PXD009360, PXD007263, PXD009181, PXD009438, PXD009112 and PXD009508.
Keep the original pinned PXD009092 metadata/SEARCH.7z; freeze new retrieval bytes
and input hashes separately. Read the primary paper and supplementary descriptions
for explicit experiment-to-deposit evidence. Dataset titles/run names are clues,
not evidence of exact specimen identity. Patient/mixed-PBMC data are discovery
metadata only; prefer the cell-line cohort and do not download patient raw data.

## Read-only processed-table audit

After verifying size/hash/CRC and safe archive paths, stream selected original
TSV members to the shared source cache. Preserve originals with byte hashes.
Record every header, row count and run/experiment mapping. Inspect parameters,
summary, peptide, evidence and protein-group tables; do not search new RAW data.
Report native/tryptic search scope, reference file/version versus unavailable
reference hash, exact quality fields and levels (PSM, peptide, protein), decoy and
contaminant flags, modifications, direct MS/MS versus matching-between-runs,
run-specific intensity presence/zero/blank, peptide-to-parent multiplicity, and
whether an actual independent global-proteome assay is present.

Preserve all source facts and conflicting or differently scoped assertions for
shared Hitlist curation. No prototype schema is a production alternative. Source
rows/runs are not independent biological specimens without evidence. A label such
as NO-INH or a numbered suffix is not silently assigned a biological intervention
or donor. No intensity zero is automatically a measured absence. LFQ/summed
fragment protein-group intensities are not independent precursor abundance.

## Prospective diagnostics and decision

Count source support without filtering away failures. Provide source-linked
per-run coverage and a peptide-row eligibility ledger, distinguishing sequence,
reverse/contaminant, direct-ID support, parent ambiguity, quantity coverage and
unresolved sample/measurement meaning. Counts may support a later frozen recipe,
but this audit itself admits no training cohort. Inspect peptide/protein-group
reported ID links and reconcile them independently. If the intended cytokine
experiment lies in another deposit, explicitly replan around that source before
fitting. Preserve contrary source evidence rather than fitting the wrong cohort.

## Reproducibility and closure

One registered family with read-only script and invocation snapshots, immutable
source manifests, sample/header inventories, counts and source-linked metadata
gaps, independent parsing/reconciliation and actual runtimes. No model,
pretraining, training, loss, synthetic data, GPU/Modal or predictive split/metrics
apply; this is biological endpoint admission, not predictive evaluation. Do not
claim intracellular concentration, free-pool distribution or APC generalization.
Update the canonical log and upstream shared-curation issues with decisive gaps;
then specify the actual same-objective exact-versus-sampled biological comparison.

## Header-review refinement before computation

The complete run map contains18runs, including UT and TNF+IFN in addition to
protocol controls; the first-three-run preflight was insufficient to classify the
whole archive. Preserve all18, with no grouping from the first few entries.
`summary.txt` also contains experiment aggregate rows and a Total row, which must
not count as acquisitions. Peptide intensity columns include `MAPPbeadFilt` in
15labels and omit bead-control intensities, while evidence and identification
columns retain18experiments. Audit that postprocessing explicitly: compare each
reported peptide/run quantity to its sum of linked source evidence intensities,
using relative tolerance1e-4 and absolute tolerance1 for a diagnostic of exported
rounding/filtering, not a claim the two must be equal. Preserve zeroed-vs-positive,
both-positive agreement/disagreement and both-zero counts separately. No adjustment
is selected based on the outcome.

The source-row ledger will report structural coverage for the potential UT versus
TNF+IFN contrast: canonical peptide, no reverse/contaminant flag, one protein-group
ID, >=2positive intensities of3 in each arm and >=1direct MS/MS evidence-bearing
run in each arm. This is a prospective support diagnostic only; confidence/source
provenance/normalized metadata admission remain unresolved. No PEP threshold is
substituted for a global FDR threshold, and no intensity sum is called precursor
protein abundance. The paper reports peptide FDR0.01, project metadata says0.1,
and parameters report PSM0.01/protein1; retain these distinct claims and levels.

## Source-shape stop and replan after attempt 1

The first audit stopped before evidence processing when protein-group IDs were
not unique. Original TSV records all have the declared175fields, but36records
have empty IDs, including entirely empty records and numeric-list-only records.
Preserve the failed recipe/log and its partial inventories. On the revised audit,
retain an explicit source-row quarantine with raw-row hashes and occupied fields;
index only rows with nonempty numeric source IDs and require those IDs to be
unique. Do not reconstruct missing group content from adjacent physical rows.
Keep total source record count distinct from indexable group count and preserve
all unresolved/reciprocal links in the report. This is source qualification, so
no affected parent abundance or identity may enter training. The other four
members have consistent row widths and valid IDs where applicable.

## Biological comparison handoff after source audit

The source audit now identifies2,272structurally supported peptides in572groups,
including1,971peptides in271multiple-peptide groups. All remain deferred until
source curation/admission. These counts bound a first real-data comparison; they
must not be replaced with paper-wide counts or treated as an entire product pool.

Next execute the independent PXD009360 whole-proteome table audit and explicit
material-pairing decision. Preserve whether matching is exact specimen, biological
replicate, or only a condition-level surrogate. Resolve the actual filtering rule
for9,276positive-evidence/zero-export cells; original source values and exclusions
must survive even if a recipe uses only common positive support. Preserve the
peptide-FDR conflict and choose any later PEP requirement as an explicit separate
eligibility policy, not as a retroactive change to the deposited search threshold.

The first biological hypothesis should be whether peptide sequence and verified
cell context predict *observed relative fragment signal* on held-out protein groups
beyond a length/composition baseline. Compare exact summed distribution loss and
REINFORCE with the identical endpoint/candidate support/score network, retaining
one tested hybrid if useful. Start with the fixed exact and fresh REINFORCE
references; do not rerun a large GRPO factorial before basic predictive signal is
demonstrated. Group standardization/clipping/reuse alter the estimator and remain
separate ablations, not hidden defaults.

Before launch, a new fit-specific spec must freeze the exact normalized Hitlist
source generation, row-level eligibility, duplicate/reanalysis joins, ambiguous
parent grouping, source-protein connected-component holdouts, target transform,
loss, sequence/context features, optimizer budget and selection rule. Preserve
train/validation/test predictions, nonconstant-output diagnostics, and real
runtime/score counts. Use held-out same-peptide condition changes to assess signal
beyond fixed peptide response, with precursor adjustment only at the supported
pairing precision. Do not call an intensity-normalized observed set a molecule
probability or infer free-pool products from unobserved zeros. Independent APC and
concentration evidence remain separate requirements for the full active goal.
