# PXD024917 transition lineage

**In progress: source cells and aggregate sums replayed; published-signal lineage only partly reproduced.**
This source-only family has no model fitting, calibration training or biological
held-out access. The original resource and discovery contracts remain active.

## Retrieved evidence

The official PXD024917 project and 67-entry first file-list response identified
the two exact processed exports. Fetch attempt4 retrieved both without redirects:

| Export | Bytes | SHA256 |
|---|---:|---|
| Iso18_DirectSpike_TransitionResults.xlsx |151,291|d359bf7b8afd0ade9df9539d941c0f7931b8c39657a444d3bf8025e54ff1d995|
| Iso18_hipMHCspikein_TransitionResults.xlsx |69,651|8426f43364ea1d84244403811956f501664abf8001e7043ec8fc1ae546175308|

Both match their deposited byte counts and fetched SHA256 receipts. Their computed
SHA1 values also match the inventory's opaque checksum strings; the API does not
declare a checksum algorithm. S7's original hipMHCspiikein spelling (double i)
and PRIDE's hipMHCspikein filename remain separate source claims.

Inspect attempt1 independently compared all42,941 stored cells using OOXML and
openpyxl. DirectSpike has29,825 stored cells in A1:N2593; hipMHC has13,116 in
A1:L1093. Stored cells are not a count of independent observations.
Both exports have the same twelve primary columns: peptide, modified sequence,
replicate, precursor m/z/charge, product m/z/charge, fragment ion, retention time,
area, peak rank and Concat. DirectSpike's separate N-column legend specifies
nominal amount units; it is not row-aligned sample metadata. The hipMHC preview
contains precursor rows, which must remain distinct from fragment-ion signals.

Original assets remain once under
`data/nontryptic_transition_sources/2026-10-25_1857_codex_nontryptic-transition-lineage/`.
See `results/transition-inventory.json`, immutable receipts, and the exact
hash-pinned requests under `reproduce/requests/`. Four of eight allowed network
operations have been consumed; no further retrieval is needed for these exports.

## Semantic qualification and remaining mismatch

Verify attempt 2 passed the original guards and the pinned Hitlist reader replay.
The exports contain 2,592 DirectSpike rows in 432 peptide/form/replicate groups
and 1,092 hipMHC rows in 252 groups. There are no duplicate transition identity
keys. DirectSpike has 1,296 absent Concat values; none are nonempty contradictory
Concat values. The original detailed diagnostic called these missing values
“concat_disagreement”; that diagnostic is preserved and this distinction matters.

The two fixed aggregation candidates are product ions only and all transitions
including precursor. DirectSpike's candidate sums agree within relative 1e-6 with
144/213 S3A cells under an **unproven** ordinal-to-suffix correspondence; exact
equality is 0/213 and the conditional integer-rounding bound holds for 125/213.
Some differences are far larger than rounding: SLYSYFQKV[+6] S3A D66 is
165,253,246.4165 versus an export sum of 238,591,035. This does not identify a
normalization or transition-selection rule. No matching subset or scale is fitted.

The initial S3B lookup has a confirmed audit bug: it requires its bare peptide
labels to equal modified export forms, producing no candidates for all 135 cells.
That does **not** establish missing biological data. The source-declared 2H form
must first be resolved against S6B's explicit labeled positions and nominal shift.
The hipMHC export also lacks a `100fmol_2` label; blank cells remain unfilled.

## Corrected source mapping

Mapping-audit attempt 2 passed under the original guards in 1.189 seconds, with
103,596,032 bytes peak child RSS. All 54 exported forms have a unique match to
one of the 72 S6B assertions by bare sequence, explicitly labeled positions and
reported total nominal shift. This resolves reported-form correspondence; it
**does not resolve canonical isotope atoms or repair the conflicting S6A claim**.
All 684 groups and both fixed aggregation modes independently reproduce their
stored sums from original workbook cells. All DirectSpike groups have six product
ions. hipMHC has 140 groups with six products plus precursor and 112 groups with
precursor only; those are different measurement types.

| Source and fixed sum | Numeric ordinal comparisons | Relative 1e-6 matches | Exact matches | Maximum relative difference |
|---|---:|---:|---:|---:|
| S3A product-only (all-transition identical) |213|144|0|46.63%|
| S3B product-only |117|105|0|0.5494%|
| S3B products plus precursor |117|1|0|11,230%|

S3B's 135 slots now all have form/dose candidates. Only 117 have both an observed
signal and an ordinal-equals-export-suffix candidate. At 100 fmol the export has
labels 1 and 3, while S3B has values in C/D and a blank E for every peptide.
Thus equating C/D/E with 1/2/3 is demonstrably unsafe. No replacement run mapping
is asserted from intensity resemblance. The nine blanks are not fitted zeros.
Product-only sums are consistent with the reported signal type, but incomplete
agreement and unknown acquisition mapping prohibit declaring full reconstruction.

The first mapping attempt's startup refusal remains preserved. The companion
launcher adds only this stage to the immutable original's dispatch choices;
all guards are unchanged. Code, scope, sources and invocation are pinned in
`reproduce/requests/mapping-audit-1.json`, `reproduce/mapping-audit-plan.md` and
`reproduce/mapping_audit-attempt2-launch.json`. The source-only 1 GiB proposal
remains unapproved and inactive.

## Shared ingestion decision

The measured source schemas and literal values are eligible for source-faithful
curation. Store reported S3 quantities separately from transition sums, and retain
candidate correspondence with its precision instead of making it an exact join.
Keep protocol, spike stage, standard amount assignment, matrix, precursor/product
signal type and isotope conflicts queryable. Unknown acquisition/aliquot identities
remain null. This family does not yet provide the shared typed package or a recipe.

A future empirical response test may use the 18-peptide free-standard panel with
peptide-grouped folds, nominal dose, sequence and source-reported signal. It must
state that aggregation and isotope chemistry are unresolved, preserve all source
values, and avoid calibrating endogenous APC concentrations from this panel alone.
Embedded standards involve capture/recovery and remain a separate observation
process. No model fit has run; the full biological objective is still open.

## Runtime and software evidence

- Fetch attempt1: metadata passed,3.408s,46,219,264bytes peak child RSS.
- Fetch attempts2/3: refused before child/network at2,804,875,264 and254,410,752bytes.
- Fetch attempt4: both exports passed,1.939s,46,383,104bytes peak child RSS.
- Inspect attempt1: passed,0.975s,107,790,336bytes peak child RSS.
- Verify attempt1: resource refusal before child/network; no numerical result.
- Verify attempt2: passed, 1.222s, 105,496,576 bytes peak child RSS.
- Mapping-audit attempt1: startup refusal; no child, no numerical result.
- Mapping-audit attempt2: passed, 1.189s, 103,596,032 bytes peak child RSS.
- All scientific stages use local CPU, one numerical thread,20min/stage,3GiB startup
  and RSS limits,1GiB disk reserve+32MiB startup margin,50ms RSS polling and OS peak checks.
  No Modal/GPU/shared-environment installation. All launched stages are terminal.

The internal reader is in [Hitlist draft PR850](https://github.com/pirl-unc/hitlist/pull/850),
head0263dc639fdebe0a1532260f09cf575f99c4bf32. All hosted software CI passes:
five Python unit versions,3,621passed/two optional-dependency skips on3.11,96
integration tests, lint and combined coverage. Reader line coverage is99%.
Those synthetic software tests do not qualify source semantics. The reader now
passes original-source replay; source package, recipe and trained model are not
yet implemented by this family. Conflicting isotope assertions and unproven
acquisition joins remain explicit exclusions from stronger claims.

Starting Presto717ed59; Codex/GPT-6; unrelated work preserved. The frozen original
plan is `reproduce/plan.md`. No training/pretraining, synthetic biological labels,
loss/weights, predictive splits or validation/test metrics apply to this source-only
task. No earlier biological held-outs were reopened. Full precursor-relative APC
availability and biological exact-versus-RL validation remain unachieved.

Status: source correspondence audit complete; shared ingestion still pending.
Next step: implement the typed source package and separately register a response recipe.
Open questions: acquisition joins, aggregation exclusions, isotope atoms and transfer to native peptides.
