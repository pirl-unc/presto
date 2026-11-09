## Recovered controls and time-course outputs — October 23, 2026

[Completed source audit and immutable reproduction](https://github.com/pirl-unc/presto/blob/dfe3af2/experiments/2026-10-23_0434_codex_mapp-controlled-admission/README.md). Both public processed archives are now fully downloaded, PRIDE-SHA1/size verified, and independently replayed. Do not repeat source discovery or treat these panels as unavailable.

| Source | Actual recovered coverage | Quantitative verification |
|---|---|---|
| PXD009508 | 6,060 peptide rows, 19,581 evidence rows, 18 runs; `alpha6`, `iso`, `gap` capture labels crossed with `ut`/`epoxo`, three numbered replicates per arm | All 109,080 peptide/run comparisons reproduce linked evidence sums within fixed rounding tolerance or are both zero |
| PXD007263 | 3,146 peptide rows, 27,196 evidence rows, 12 runs; `TNF-TP-0-6` labels with 0/1/2/6 and three replicate numbers | All 37,752 comparisons reproduce under the same policy |

All 30 deposited raw-file names map to actual summary/evidence rows. Ten original source tables and all 146,832 comparisons were independently checked using BSD tar extraction and a separate pandas aggregation by evidence `Peptide ID`. Complete inventories, row-level values/contributors, exact literal labels, failed attempts and hashes are published. Controls SHA256: `87d3711ad4a0e59d2618d079c2d6e95e14b8c544739701349d2b36bb4d5a14b8`; time course: `05e46c2be2bc6046a4dee1f264d2c4ffd5fd3cba821ae778ece8c34906e8dde4`.

### Concrete remaining shared-curation work

- Encode the controls' source-label mismatch explicitly: peptide column `1511_7181-300a_alpha6-ut_1` versus evidence/summary `1511_7181-300a_alpha6_ut1`. Unique punctuation aliases are supported by the full arithmetic audit, but are derived relations, not supplied specimen identities. Preserve both names and every contributor. The first strict-join failure is retained.
- Curate capture type separately from live-cell epoxomicin treatment and lysate-preservation inhibitors. Tie `alpha6`/`iso`/`gap` interpretations and paper-level dose/duration to cited source assertions; retain unknown run-specific specimen/aliquot links. Numbered repeats and the paper's replicate statement do not supply an exact material crosswalk.
- Preserve the time-course literal `TNF` labels alongside the paper's TNF-alpha plus IFN-gamma description. Resolve or explicitly scope that assertion before exposing a treatment default.
- Preserve source confidence and observation restrictions: PSM FDR 0.01 versus protein FDR 1; match-between-runs enabled; 7,136/12,131 `MULTI-MATCH` evidence rows; 400/500 evidence rows with unreported intensity. Reported zeros are not biological negatives, even though no filtered-zero/positive-evidence discrepancy occurs in these panels. Reverse, contaminant and ambiguous-parent rows remain discoverable and explicitly excluded by any training recipe.
- Recover/qualify the source sequence reference: the archives name `Human_UP_2016_02.fasta` but do not contain its bytes. Preserve original IDs/positions and reference-version uncertainty; do not silently replace it with a newer proteome.
- Materialize these sources and a declared recipe in the existing canonical query system, independently replay the export, then assess repeated-observation signal. Keep total detected signal and normalized composition separate: a unit-sum distribution cannot express a global inhibitor-associated decrease. Neither endpoint independently establishes molar intracellular abundance.

No Velcade- or ZsGreen-labeled run, or match to the four displayed reporter peptide sequences, was found in these **two selected processed panels**. That does not prove absence from all raw acquisitions; a human-only search reference is not a reporter-negative assay. Exact reporter construct/PRM quantities and Velcade run mappings remain open requests here. No author outreach occurred.

Complementary purified-protein quantitative time courses now have a separate source-specific owner, #770 (MSV000092631), including actual public file inventory and background/selection requirements. The recovered cellular inhibitor panel remains the immediate priority for this issue.
