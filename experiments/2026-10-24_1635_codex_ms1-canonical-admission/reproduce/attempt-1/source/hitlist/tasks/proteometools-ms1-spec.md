# PXD021013 measured relative MS response (#762)

## Purpose and scientific boundary

Provide original non-MHC peptide MS measurements through EvidenceStore and a
registered recipe. This is an observation constraint for subsequent Presto
processing comparisons. It is not a model of absolute detector efficiency:
crude synthesis, unknown individual purity and estimated 200/500 fmol injection
confound synthesis, handling, chromatography and MS response. HLA library origin
is not measured MHC binding. Search enzyme is not laboratory digestion. Missing
planned products are not detector-negative labels.

## Source package

Use a standalone `proteometools_ms1` family, package schema 23, catalog 28,
revision `pxd021013_ms1_v1`. Preserve the original README index, positional SDRF
columns, primary article XML and intended-product workbook. Pin these reviewed
core assets by their audited hashes. Accept any explicitly supplied subset of
deposited SEARCH ZIPs, with local filename, original indexed source name, hash
and byte count. Archive hashes are caller-pinned acquisition claims, not a claim
of remote signature verification. Validate every supplied ZIP against its receipt
and exact README SEARCH->RAW link. Do not require a new release for each pool.
No network acquisition or shared-environment installation in the writer.

Typed relations: assets; positional source records; all intended pool/sequence
rows with sheet/cell locators; runs; peptide quantities; molecular-form quantities;
protocol contexts. Keep all raw quantity strings, finite numeric values or explicit
missing/zero/invalid statuses, PEP and exclusion flags. Preserve the source bytes
of ZIP members outside the typed projection. Resolve pool names against the
reviewed intended mapping, never infer biological protein origins from concatenated
search-reference coordinates. Reject ambiguous links, duplicate IDs, sequence/form
inconsistency, multiple raw runs, or aggregate/channel intensity disagreement.
The first projection supports single-run MaxQuant SEARCH archives; unsupported
layouts fail explicitly instead of silently inventing aggregation.

Raw SDRF assertions remain positional source records; typed protocol fields retain
the primary synthetic-pool context and explicit SDRF conflict statuses. Preserve
fixed and variable modifications, original run/acquisition labels, software/search
settings, library origin, source URLs, estimated-load alternatives, unknown purity
and unknown per-run load choice. No specimens are inferred from SDRF `cell/run1`.

Build atomically, refuse existing destinations, hash all artifacts, replay the
typed projection from original sources on verification. Register catalog relations,
links and EvidenceStore family dispatch without changing old package contracts.

## Relative-response recipe

Registered `pxd021013_intended_unmodified_ms1_v1`: require planned full-length
sequence in the same pool, canonical residues, positive finite sequence and form
intensity, PEP in [0,.01), neither reverse nor contaminant at either grain, and
one uniquely identified `Unmodified` variable-modification form per sequence/run.
Retain fixed-modification context: `Unmodified` does not mean no fixed CAM.
Do not sum ambiguous forms or make absent products into zeros. Preserve every
form disposition and planned-but-unobserved product separately. Output raw log10
signal plus within-run centered log10 signal; centering defines the observed
conditional composition target, not a deployable offset inferred from held-out
outcomes. No held-out CV filtering. Group pools and repeated sequences before
splitting, using intended sequences and observed identities, never intensities.
Expose components without promising enough independent groups for a predictive fit.

## Verification and prospective biological use

Meaningful fixture tests cover positional duplicate SDRF columns, raw missing/zero
states, off-planned products, multiple forms, contamination, source-row/quantity
conservation, inconsistent joins, tampering even with rehashed manifests, subset
extension and connected pool grouping. Fixtures are software tests, not biological
labels. Run format.sh, lint.sh and test.sh with capped workers and the original
memory guard. Run real-source qualification in a registered Presto experiment,
compare original source counts and literal values independently, preserve exact
source/code/env receipts. No model training is claimed from package verification.

After canonical admission, preregister additional pool selection from index/mapping
metadata and a grouped sequence-versus-constant/length response comparison. Freeze
the successful observer before future matched exact/sampled processing experiments;
previous biological held-outs stay closed to tuning. Release via PR with version
bump, full required gates and clean-main PyPI deploy; unshipped work is unfinished.

## Environment

Use the feature worktree directly on PYTHONPATH, never install it into the shared
environment. Creation triggered an existing repository hook; neutral-cwd inspection
confirmed shared Hitlist 1.82.2 still points to /Users/iskander/code/hitlist and pip
check passes. Command-scoped hooks disabling prevents further incidental refreshes.
