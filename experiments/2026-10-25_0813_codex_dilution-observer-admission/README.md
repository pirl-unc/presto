# PXD024809: source audit passes; calibration training remains unadmitted

The public processed table is usable for shared source materialization. Independent
readers reconcile **64,070 rows / 896,980 original string cells**, a complete
**6,407 precursor × 10 run** grid. Nine of ten nominal standards occur in the
table, with90standard/run slots. This provides a concrete candidate for a small
dose-response diagnostic, not an admitted calibration model or APC concentrations.

The export is a **DIAlignR-aligned quantity table**. Original scoring/integration
semantics, parameter provenance and the absent standard's query eligibility need
reconciliation before defining a training likelihood. A positive number alone
does not establish an independently identified signal. No predictor was fitted.

The prospective contract is `reproduce/plan.md`, copied from
`experiments/agents/codex/plans/native-baseline-observation-admission.md`.
Bounded public metadata and article XML were inspected before selecting the single
13MBprocessed quantity table. No raw MS downloads, private access, author outreach
or package installation occurred.
Original assets live once in `data/dilution_observer_sources/2026-10-25_0813_codex_dilution-observer-admission`.

Source inventory and target joins must distinguish standards from HLA-selected
background, nominal dose from certified amount, technical acquisitions from
biological replication and missing observations from zero. The indexed article's
twofold versus1:10 dilution conflict remains unresolved pending original evidence.

Agent/model: Codex / GPT-6, October25,2026. Base commit
`869b2acfffb6a15ac914b9a335dfef66f8ad507e`; dirty with unrelated task edits and
historical untracked evidence preserved. Local CPU only, no Modal/GPU. No
pretraining, learned weights, synthetic biological labels, model losses or assay
heads. No validation/test predictions or500nM metrics apply to source admission;
original native outer partitions remain closed. This is not a new biological
exact-versus-RL comparison.

## Data, units and observation grain

[Primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12406853/) and
[PRIDE project](https://www.ebi.ac.uk/pride/archive/projects/PXD024809).
The68-file inventory matches the API's total-records header, with unique names.
It includes24RAW,15mzTab,15idXML,12TSV,oneconvertedmzMLandoneCSV. No deposit
checksums are supplied; our SHA256 values are retrieval fingerprints. The selected
`Dilute_peptide_quantities.csv` is actually TSV,13,073,500bytes,SHA256
`646f8054ca9f60d9ebad46722a994c507c0b30c0c897a75df356586f0bfcccb8`.

All14columns survive: peptide_id, precursor, run, RT, intensity, leftWidth,
rightWidth, peak_group_rank, m_score, alignment_rank, feature_id, sequence, charge,
group_label. There are6,406peptideIDs/sequence forms,6,407precursors/group labels,
two charge states and no duplicate precursor/run slot. No quality filter, zero
imputation, censor substitution, sequence collapse or protein assignment was applied.

| Retained scope | Positive intensity | Literal zero | Literal NA | Missing m_score |
|---|---:|---:|---:|---:|
| All64,070query slots |58,063|5,743|264|10,939|
| Nine observed standards × ten runs |88|2|0|3|

The tenth nominal standard has **no row**, a different state from either zero or
NA. Its library/query eligibility has not been inspected. All observed standards
are charge2with the specified heavy leucine position preserved in both the
sequence UniMod695 and group_label representations. The annotation agrees with
[UniMod695](https://www.unimod.org/modifications_view.php?editid1=695); modified
standards remain distinct from endogenous unmodified peptides.

Table1 reports nominal standard concentrations0.2/0.1/0.05/0.025/0.0125fmol/µL.
The project protocol agrees with that twofold grid; the article Results says1:10.
Both source assertions remain, with an explicit conflict. All ten run labels match
depositedRAWnames after recorded`#`→`_`normalization, with two acquisition indices
perlevel. These are filename-derived links, not proof of biological replication,
specimen pairing or equal recovery. No concentration-to-on-column-amount conversion
was made; injection volume, lot/purity certification and aliquot provenance are
not yet qualified. Source MHC alleles annotate the matrix, not standard restriction.

## Processing semantics and admission decision

Pinned [nf-core/diaproteomics1.2.2main](https://github.com/nf-core/diaproteomics/blob/1.2.2/main.nf)
runs`bin/DIAlignR.R`, then renames`DIAlignR.tsv`to`*_peptide_quantities.csv`.
The wrapper calls`alignTargetedRuns`with several supplied FDR/alignment parameters.
Both files are retained once underdata/; GitHubblobIDs independently match their
downloaded bytes. This establishes export type, not actual study parameter values.
Pipeline defaults must not be substituted for unverified run settings.

**Proceed to shared Hitlist source ingestion and targeted scoring/library
reconciliation; do not train yet.** Existing per-run scoredTSVs and the master
library are available in the inventory. The next work should use those assets,
preserve the standard panel and all query slots, and separate directly scored
from aligned/integrated observations. Full field/lineage/conflict/recipe criteria
are in [the shared admission contract](reproduce/hitlist-admission-contract.md).

Even a qualified nine-standard diagnostic would not establish broad sequence
generalization, native non-MHC MS1transfer, intracellular molarity, MHC recovery
or precursor-relative APC availability. Matrix, isotope form, chemistry and query
selection remain relevant. The broader model goal stays active and unachieved.

## Verification, runtime and preserved failures

The production reader usescsv/ElementTreeand the independent reader usespandas/
DOMplus a separate literal-protocol parser. They agree on all896,980string cells,
numeric missingness states, source rows/identities, standard forms/rows, all68file
identities, five dose cells and run-name joins. Independent verification imports
no production auditor. No fitted slope, recovery curve or biological target
summary was used to select the source or nine observed standards.

| Stage | Outcome | Seconds | OS peak RSS |
|---|---|---:|---:|
| Initial metadata retrieval | Project/XML pass; generic file route empty |2.123|32.8MiB|
| Correct project-file route | Pass; all 68 files |1.282|32.5 MiB|
| Processed retrieval | Bytes/size pass; comma-header guard fails on TSV |20.361|73.3MiB|
| Original audit | Stops before table scan on Unicode thin-space unit separator |0.126|25.1MiB|
| Unicode-aware audit | Pass |0.882|47.6MiB|
| Independent verification | Pass |1.191|159.5MiB|

The failedroute/format/Unicodeattempts, original code/configs, sourcebytes and
nonzero exits remain. No completed stage receipt or source asset was overwritten.
The Unicode amendment accepts the original separator; it changes no numericvalue.
Numerical stages obey20min/3GiB/1GiBdisk+32MiBstartup,50ms process monitoring and
OSchild peak; all resource gates pass. Retrievals have240second/60secondsocket
budgets and4MiBmetadata/32MiBselected-table caps. Source code was frozen before
each attempted stage; stage-start environment records are included.

## Reproduction and artifacts

`reproduce/launch.json`and`launch.sh`freeze initial metadata retrieval.
`files-retrieval.json`/`files.sh`record the corrected route;
`processed-retrieval.json`/`processed.sh`the selected table. Their original
failure outcomes are intentional historical evidence, not commands to rerun over
existing receipts. The admitted source scan uses:

```sh
sh experiments/2026-10-25_0813_codex_dilution-observer-admission/reproduce/audit-unicode.sh audit_unicode
sh experiments/2026-10-25_0813_codex_dilution-observer-admission/reproduce/audit-unicode.sh verify_unicode
```

Completed stage names refuse overwrites. Replays require a fresh output location
and the same pinned inputs; do not delete existing receipts to force execution.
Large inputs stay once underdata/, pinned by`results/input-manifest.json`.
`source-audit.json`, original-standard-row records,`standard-summary.json`,
`quantity-state-summary.csv`,`verification.json`and`admission-decision.json`
preserve findings. `reproduce/source/`retains every executed audit/retrieval source.
Logs, environment and resource receipts include unsuccessful attempts. No predictive
dump applies because no model or biological split was fitted/evaluated here.

Shared source/metadata work is filed as [Hitlist#845](https://github.com/pirl-unc/hitlist/issues/845). The posted body was read back exactly.
