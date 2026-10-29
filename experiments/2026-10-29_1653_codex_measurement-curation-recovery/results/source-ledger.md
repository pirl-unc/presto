# Additional source acquisition ledger

All listed archives were downloaded and hashed. Acquisition is distinct from independent, admissible training coverage. The current retrain uses only the source-qualified recovery and censor corrections, not these unreviewed bulk archives.

| Source | Bytes | Admission / overlap status |
|---|---:|---|
| [mhcflurry-data_references](https://github.com/openvax/mhcflurry/releases/download/pre-1.4.0/data_references.20190927.tar.bz2) | 139,213,669 | Reference proteins/context, not additional assay labels. |
| [mhcflurry-data_mass_spec_annotated](https://github.com/openvax/mhcflurry/releases/download/1.6.0/data_mass_spec_annotated.20191226.tar.bz2) | 101,446,143 | Historical MS annotations; source, PTM, predicted restriction and overlap review required. |
| [mhcflurry-data_systemhcatlas](https://github.com/openvax/mhcflurry/releases/download/pan-dev1/data_systemhcatlas.20190506.tar.bz2) | 15,703,955 | Historical 2019 archive; not the current SysteMHC release. |
| [mhcflurry-data_iedb](https://github.com/openvax/mhcflurry/releases/download/pre-2.1/data_iedb.20231019.tar.bz2) | 184,104,290 | Historical IEDB mirror; current complete export already supplies the baseline. |
| [mhcflurry-data_curated](https://github.com/openvax/mhcflurry/releases/download/pre-2.1/data_curated.20231023.tar.bz2) | 77,540,318 | 249,802 affinity-category rows (177,438 quantitative;72,364 qualitative). Literal assay identity is collapsed; mixed preparations/species and IEDB overlap require review. |
| [abelin-2019-original-s1b](https://ars.els-cdn.com/content/image/1-s2.0-S1074761319303632-mmc2.xlsx) | 998,734 | Original ligand supplement has sequence lists but no row-level modification annotations; do not fill unknown chemistry with False. |
| [mhcflurry-data_published](https://github.com/openvax/mhcflurry/releases/download/pre-1.7.0/data_published.20200501.tar.bz2) | 257,046,105 | Published datasets and metadata; original-source and overlap review required. |
| [netmhciipan-4.3-training](https://services.healthtech.dtu.dk/services/NetMHCIIpan-4.3/training_data/NetMHCIIpan_train.tar.gz) | 139,440,263 | 129,110 BA and7,594,916 EL rows. EL includes generated background; transformed/clipped BA does not preserve literal assay identity. |
| [netmhcstabpan-stability](https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/Stability.txt) | 799,880 | 28,166 stability rows for75alleles, hours;5,679 zero values require measurement-floor/rounding review before logarithmic admission. |
| [netmhcpan-4.2-training](https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/training_data/NetMHCpan_train.tar.gz) | 269,075,958 | 214,804 BA and17,349,161 EL rows. EL includes generated background; IEDB/CEDAR/source overlap unresolved. |
| [hla-ligand-atlas-2020.12](https://hla-ligand-atlas.org/rel/hla_2020.12.zip) | 12,684,944 | Primary donor/sample/peptide files acquired; donor-linked admission remains tracked by Presto#127. |
| [hla-peptideatlas-2026.09](https://peptideatlas.org/builds/human/hla/202609/atlas_build_617.tsv.zip) | 1,577,040,076 | Current large atlas acquired (30TSV members). Study dates, PTMs, restriction provenance and2026holdout overlaps need review; archive size is not training-label count. |

Original Stražar human monoallelic PSM recovery:11of132selected files downloaded. Persistent HTTP429 and an anonymous FTP timeout prevented the remaining121; no alternate credentials or access bypass was attempted. The11files yielded14,187source-reviewed unmodified pMHC observations before length/partition exclusion.

Source hashes, URLs and acquisition receipts are in `acquisition*.json`; archive member/row inventories are in `archive-inventory.json` and `peptideatlas-files.json`. Public-source coverage remains incomplete; downloaded mirrors must not be counted as independent additions.

## NetMHCstabpan follow-up source review

The archived table contains 28,166 unique pMHC rows over 75 reported constructs:
22,487 positive half-lives and 5,679 literal zeros. Of the positive rows, 5,012
already share a TRAIN pMHC half-life label and 4,991 have exactly the same value
after hours-to-minutes conversion. These cannot be counted as independent additions.
There are 1,135 rows across three explicit mutant-MHC identifiers unresolved by
the current canonical parser; mutations are preserved and never stripped.

The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/) describes
scintillation-proximity dissociation measurements and geometric means of repeated
half-lives, and separately describes 1,000 predictor-selected zero-hour synthetic
negatives per allele. The downloaded archive's 28,166-row count matches the paper's
measurement subset, not its augmented training size; that does not by itself prove
the meaning of each exported zero. Keep zero values out of exact log-half-life
regression until their censor/non-detection semantics are documented. Per-row
modification, study lineage, construct identity and heldout overlap need resolution.
No rows from this review have entered the current retrain.

See `stability-review.json` for source hashes, row locators and overlap IDs. The
source census took 5.38 seconds locally on CPU, no GPU, training or heldout scoring.
The initial parser failure and corrected inventory are both preserved.

Upstream source integration is tracked in [Hitlist #875](https://github.com/pirl-unc/hitlist/issues/875); filed body was verified by read-back.
