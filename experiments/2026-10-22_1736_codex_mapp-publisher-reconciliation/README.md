# MAPP publisher / deposited-quantity reconciliation

**Completed October 22, 2026; Codex / GPT-6.** The deposited background filter and
publisher's nonimputed peptide summaries now reproduce exactly. The imputed
columns have a major condition-label inconsistency and remain excluded from
training. No model was fitted and no biological rows were admitted by this audit.
The full availability/concentration objective remains open.

## Decisive results

| Question | Result |
|---|---|
| Deposited bead-filtered quantities |160,095/160,095 cells reproduced by the fixed run-versus-median-beads rule|
| Publisher untreated log10 medians |4,983/4,983match same-sequence deposited medians|
| Publisher TNF+IFN log10 medians |5,311/5,311match same-sequence deposited medians|
| Publisher sequence membership |Both sets reproduced exactly;6,311 unique sequences in their union|
| Reverse/contaminant records in publisher |0/0|
| Publisher imputed E versus named UT |Only2/6,311match;4,868match CHX instead|
| Publisher imputed F versus named TNF+IFN |Only2/6,311match;4,913match UT instead|
| Data3 |131 distinct gene labels,524 positive intensity cells; no acquisition/specimen map|

This eliminates one uncertainty before biological modeling: the9,276 reported
zeros with positive underlying signal are accounted for by background filtering.
They must retain that selection history; they are not measured absence of the
intracellular peptide product.

## Sources, curation and provenance

Primary study: Wolf-Levy et al., DOI[10.1038/nbt.4279](https://doi.org/10.1038/nbt.4279).
Original public publisher supplements were linked from its Supplementary Information
section; exact URLs, sizes and SHA256 hashes are in
[publisher_assets.json](reproduce/publisher_assets.json). No authentication was
used. Earlier PMC downloads that returned HTML challenge pages were excluded.

- Data2:467,804bytes, SHA256`22ca91a68f772042a8a96848bc2a9446b120046fcfe2baa31968e5d05d839ce3`.
- Data3:16,664bytes, SHA256`cfd7f0e008f5d7cbae261092fba68cf5b1495b5e34cadcae65a2105d9e0e39fb`.
- Supplementary Note:414,962bytes, SHA256`9d626daebfbb32040f32d080bf8dc8f79741ec6684c1dac49ae994ffe1fd163c`.
- Five original PXD009092 archive members reuse the independently hashed receipts
  from the [source audit](../2026-10-22_1630_codex_mapp-source-admission/README.md).
  Its10,673 peptides,149,959 evidence rows and18 runs remain the original source.

All three publisher assets are now immutable in the shared datacache. Exact local
paths are in [input_paths.json](results/input_paths.json). Original workbook cells,
including headers and blank values, are preserved in a compressed JSON inventory;
workbooks were read only, never edited/recalculated. Comparisons use exact peptide
sequence/ID and explicit cell coordinates, not row-order joins.

This is proteasome-associated endogenous peptide capture from human HEK293 cells,
not MHC elution or a professional APC population. The main paper's establishing
experiment and methods describe the capture and search scopes; supplementary
methods distinguish native captured fragments, tryptic whole-cell lysates and
tryptic protein eluates. Those are different observation processes. The methods
pages were visually inspected. No raw448–456 treatment or specimen mapping was
obtained from the gene/intensity-only Data3 table or the supplementary methods.
The earlier PXD009360 archive/run audit therefore remains the source of the
unresolved proteome mapping, not an assumed order of triples.

New source-specific imports belong to the shared [Hitlist external quantitative
MS work](https://github.com/pirl-unc/hitlist/issues/746), following its native
modality/schema work. These audit outputs are acceptance evidence, not a second
production metadata registry.

## Frozen arithmetic and selection comparisons

The prospective [plan](../agents/codex/plans/mapp-publisher-reconciliation.md) fixed
nine candidate twofold rules: individual-run, condition-mean or condition-median
signal versus mean, median or maximum of the three bead controls. Controls and
condition summaries include zero entries for reconstructing the deposited
algorithm. The primary boundary is >=2; tolerance is1e-4relative/1absolute to
allow source-export rounding. No threshold tuning or fitted model was used.

The sole complete match among these nine candidates is:

```
B[peptide] = median(sum evidence intensity in each of three BEADS runs)
filtered[peptide,run] = evidence_sum[peptide,run] if evidence_sum >= 2*B else 0
```

This reconstructs all 15non-bead quantity columns and160,095 cells; there are no
positive exact-boundary cases for this rule. The paper directly reports a twofold
background criterion. The operational run/median implementation is an empirical
reconstruction, not an author-supplied algorithm or proof of uniqueness among all
possible rules. Keep those evidence types separate.

| Numerator | Bead summary | Mismatching cells |
|---|---|---:|
|run|mean|5,252|
|condition_mean|mean|6,985|
|condition_median|mean|7,352|
|run|median|0|
|condition_mean|median|2,607|
|condition_median|median|2,357|
|run|max|21,516|
|condition_mean|max|22,802|
|condition_median|max|22,851|

Data2 Sheet1 C5:C6315 contains 4,983 numeric untreated log10 medians and1,328 blanks;
D5:D6315 contains 5,311 numeric TNF+IFN log10 medians and1,000 blanks. Every numeric
value agrees after exponentiation with the median of its three deposited
background-filtered runs, including their reported zeros. Taking the median of
positive runs alone reproduces only 3,512 UT and3,261 TNF+IFN values, so it is not an
interchangeable reconstruction. Missing publisher values stay missing; no fitting
label is synthesized from them.

Membership is exactly all non-reverse, noncontaminant peptide rows with at least
two positive filtered runs in the corresponding condition. An additional strict
single-group-ID requirement retains4,424 UT and4,707 TNF+IFN rows, excluding559/604
publisher members. This strict policy is a possible downstream ambiguity restriction,
not a reproduction of the complete publisher set. The source provides leading
razor assignments alongside alternative protein/group mappings; neither should
be discarded or confused with unique biological origin.

There are1,022 UT and1,031 TNF+IFN distinct nonempty leading-razor labels;3,298 UT and
3,520 TNF+IFN rows list multiple protein IDs. These counts do not reconstruct the
paper's1,004 protein result because label/grain/filter definitions differ. That
protein-summary correspondence remains unresolved. The row-level
[membership ledger](results/publisher_membership.csv.gz) retains all mapping fields.

## Imputed-column conflict and eligibility

Data2 E/F are labeled imputed UT/TNF+IFN but each matches only two medians under
its own condition header. Their much stronger matches to CHX/UT are recorded in
all 40 column×condition×aggregation diagnostics and the229,160cell comparisons.
They support a condition-label conflict, not silent relabeling. The remaining
unmatched values and the imputation procedure have not been reconstructed;
E/F must not supply measured quantities, controls or missing-value replacements.
C/D and their source replicate quantities have separate, verified evidence.

Data3 is an outcome-selected131-gene panel. Its524 positive numeric cells are not
an independent validation set, a general proteome denominator or a specimen map.
The paper's search/confidence metadata and PRIDE peptide-FDR statement retain
an unresolved level/value disagreement from the prior audit. Exact original
2016reference FASTA bytes and parent-sequence mappings also remain unverified.
Source quantities can be reconciled while these provenance limits remain explicit.

**Next admission requirements:** materialize the reconstructed filtering,
reported/derived quantities, blanks, source imputation conflict, direct-versus-
transferred identification and all parent alternatives in Hitlist's common
source generation. Use endpoint-specific eligibility, source confidence and
reference mapping, preserving exact versus unresolved proteome relationships.
A first local exact/RL comparison may address relative observed fragment
composition; it must not be advertised as free-peptide concentration, absolute
processing flux, whole-pool coverage or APC generalization.

## Verification and reproduction

- Primary audit57.791s, independent scalar/CSV replay5.958s. All8 core input hashes,
  8 primary output hashes,160,095 quantity cells,1,440,855 candidate-rule decisions and
  229,160 publisher/source cell comparisons verified.
- Independent membership verification passes all 10,294 condition/peptide rows and
  six mapping fields each, plus524Data3 quantity cells. Membership computation0.416s,
  independent check0.276s. Both publisher memberships match exactly.
- First verifier was deliberately interrupted after observing repeated compressed
  NPZ reads inside its inner loop. Source/log/terminal130receipt are retained in
  `attempts/01-repeated-npz-decompression/`. Loading each immutable array once fixes
  the I/O bottleneck; no arithmetic or selection policy changed.
- No training/pretraining, loss terms, label-to-head mapping, synthetic examples,
  GPU/Modal, predictive split, held-out metrics or prediction dumps apply. Complete
  source comparison dumps replace predictive artifacts for this audit.
- Launch commit`e80fd42ef4fbbc44c009b1c8896002dcfe6a8e36`, dirty registered plans and
  source-audit files. Python3.12.6, NumPy2.5.4, pandas3.0.6, openpyxl3.1.5 on local
  macOS arm64. No non-default environment overrides. Exact commands are in
  [launch.sh](reproduce/launch.sh), execution metadata in [launch.json](reproduce/launch.json),
  and frozen scripts under`reproduce/source/`. Original datacache source/member
  receipts from the prior audit are prerequisites; preserve existing results and
  use a fresh output location for a rerun. Current scripts pass explicit Ruff
  E4/E7/E9/F checks; strict documentation build is recorded at publication.

Status: completed source reconciliation; all biological training admission remains
explicitly deferred to the shared source-generation/task contract.
Next step: qualified parent/reference mappings and shared Hitlist import, then a
registered local exact/RL comparison on observed relative fragment distributions.
Open questions: imputed-column provenance, original peptide-FDR setting,1,004 protein
summary membership, precursor run/condition/specimen map and latent concentration
identifiability.
