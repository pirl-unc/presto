# Shared MAPP source import: measured fragments with preserved selection

October 22, 2026; Codex / GPT-6. **The new Hitlist adapter materializes the original
PXD009092 measurements through the shared external-evidence mechanism.** The
first real generation and independent source replay pass. Implementation: [Hitlist draft PR #758](https://github.com/pirl-unc/hitlist/pull/758), commit `d012527`. Reference occurrence
integration, shared query/recipe admission, deployment and biological fitting
remain unfinished; this does not complete the availability goal or Hitlist #752.

## Source and curation contract

MAPP measures captured proteasome-associated fragments from human HEK293 cells.
It does not directly measure the free intracellular peptide pool, production flux
or concentration. No MHC-binding filter removes nonbinders. The original search
covers lengths 6–40; products outside it are unobserved, not negative labels.

Eleven byte-pinned original inputs: five MaxQuant tables from PXD009092 SEARCH.7z,
publisher Supplementary Data2/Data3 workbooks, and project/file snapshots for
PXD009092 and parallel tryptic PXD009360. Receipt and source URLs are frozen in
[reproduce/source/mapp_assets.json](reproduce/source/mapp_assets.json). Original
assets are copied into the local shared package; they are not committed here.
PXD009092 declares EBI terms of use, not CC0. Publisher workbook redistribution
terms are unverified; they are not bundled in the Python distribution.

The source is added alongside PXD057588 to the verified 1.82.21 supporting
predecessor. This is an **external-source generation**, not a claim that later
T-cell/numeric/query generations have already been rebuilt around it. Parent and
input hashes, exact invocation and source snapshots are in `reproduce/`.

All source records remain, including reverse/contaminant peptides and 36 group
rows with missing/nonnumeric IDs, which have distinct quarantined row identities.
Raw TSV fields and original publisher cells stay inspectable. Literal source PEP
`NaN` is preserved with `unreported_source_nan`, separate from missing intensity
and the conflicting project/paper peptide-FDR claims.

The actual UT, TNF+IFN, CHX, NO-DSP, NO-INH and BEADS run labels are preserved. The
common T+I filename prefix does not assign treatment. NO-DSP/NO-INH are procedural
preservation controls, not gene knockouts. Acquisition runs do not establish exact
specimens. Source-listed parents, leading coordinates and flanks remain source
claims; separately pinned 2023/2026 references have not yet been integrated into
this production adapter. The unavailable original 2016 FASTA stays unresolved.

## Reconciled output

| Relation / check | Count | Meaning |
|---|---:|---|
| Original TSV records | 162,767 | Every field independently compared |
| Peptide rows | 10,673 | Stable source measurement identities |
| Evidence rows | 149,959 | Direct/matched observations remain distinct |
| Acquisition channels | 18 | Six conditions, three run labels each |
| Peptide/run results | 192,114 | Includes explicit absent-evidence states |
| Quantity representations | 362,882 | Run sums, background medians, filtered values |
| Publisher quantity representations | 25,244 | Measured, missing and imputed remain distinct |
| Derivation inputs | 540,034 | Every edge independently reconstructed |
| Original publisher cells | 36,207 | Independently read from workbook XML |
| Missing evidence PEP (`NaN`) | 74,573 | Preserved as missing confidence |
| Filtered zeros with positive evidence | 9,276 | Selection outcomes, not biological absence |

All 160,095 published filtered cells reproduce under the registered rule: sum
source evidence within peptide/run, take the median of the three BEADS sums,
retain the run sum when it is at least twice that median. Comparisons preserve
the original rounding tolerance, relative 1e-4 or absolute 1 intensity unit;
this is not bitwise equality. Maximum relative difference is 8.16e-5. The largest
absolute difference is 393,010 on an intensity of approximately 10.24 billion;
[rounding details](results/filter_rounding.json) preserve the values.

All 4,983 numeric UT and 5,311 TNF+IFN publisher values reproduce as log10 of the
three-run median, including filtered zeros. The imputed E/F header conflict is
retained with no assigned condition; those cells are not measured training
labels. The selected 131-gene Data3 panel remains source cells without inferred
specimen joins or precursor denominators. Parallel tryptic condition/specimen
pairing is still unresolved.

## Verification, runtime and failures

The producer took **43.29 seconds** locally. An independent implementation took
**9.46 seconds** to verify the entire 111-file shared inventory, original TSV
records, 362,882 quantity values/statuses, evidence confidence, workbook XML cells,
all publisher measured medians and every derivation edge. See
[independent verification](results/independent-verification.json).

Small fixtures cover filtered versus absent zeros, missing confidence, imputed
condition conflict, quarantined groups, original parent alternatives, procedural
context and rejection of malformed quantities, changed filters and bad source
relationships. Sixteen focused MAPP/catalog/external tests and normal format/lint
pass. The normal full Hitlist test suite passes: **3,224 tests** in 356.83 seconds (10 warnings). The final catalog change also passes the focused suite.

Two failed attempts are retained in `attempts/`: attempt1 stopped before ingestion
on a launcher Path/string comparison; attempt2 rejected literal PEP NaN. The
correction preserved source missingness rather than changing data eligibility.
The first completed generation is immutable. The version was development 1.82.24
at execution; the pending importer PR reserves 1.82.25 because numeric PR756 also
uses 1.82.24. This artifact is not presented as a deployed release.

No training, pretraining, optimizer, synthetic biological labels, model, loss,
assay-head mapping or predictive split exists in this source-integration family.
Held-out regression/binding metrics and prediction dumps do not apply. Source
conservation is not biological predictive accuracy. No Modal or GPU was used.

## Reproduction and remaining work

[launch.sh](reproduce/launch.sh) records the exact local command; the launcher and
executed importer snapshots, input locations and manifest hashes are retained.
Restore `reproduce/input_locations.json` to `/private/tmp/mapp-seed-paths.json`
when reproducing the frozen command. Use a fresh output path/family copy, since
completed packages are never overwritten. [verify.sh](reproduce/verify.sh) runs
the independent verifier. Raw package:
`/private/tmp/hitlist-v2-mapp-external-first`; its manifest is copied into
[results/shared_manifest.json](results/shared_manifest.json).

The missing parallel-proteome metadata/precursor recipe is tracked in [Hitlist #757](https://github.com/pirl-unc/hitlist/issues/757).

Next: integrate reference-scoped alternatives, complete the shared endpoint recipe,
and register the biological comparison with the same observable objective and
holdouts for exact and sampled training. Preserve precursor uncertainty, capture/MS
response and clearance separately; observed fragment intensity does not identify
intracellular concentration. The broader goal remains active.

Source attribution: Wolf-Levy et al., [primary study](https://www.nature.com/articles/nbt.4279),
*Nature Biotechnology* 36, 1110–1116 (2018); PRIDE PXD009092/PXD009360.
