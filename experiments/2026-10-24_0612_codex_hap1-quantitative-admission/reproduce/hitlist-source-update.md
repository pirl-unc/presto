Original quantities are now retrieved and independently replayed against PXD056426
processed results. This advances source qualification; canonical typed ingestion
and a biological fit are still pending. Current Hitlist main inspected:
`fda1eff79d7a77f195e6a5f3a4c5abbee5cfd7c4` (2.0.18).

**Recovered assets**

- EuropePMC original supplements for PMC12090245: SHA-256
  `8f9327e501cdf4831fa2a9e25a67d986f631457ca331c15966f6180225b98153`.
- [PXD056426](https://www.ebi.ac.uk/pride/archive/projects/PXD056426) `Search.zip`:
  130,237,651 bytes, SHA-256
  `487af2106fa6f73d40c79dfc7d17c3c736a4e8c97a3aa97747bccf98797978b9`.
- Original sample workbook, search FASTA, DIA precursor report, protein report,
  and community SDRF are available. The noncanonical processed archive is
  inventoried but has not yet been downloaded/replayed.

**Semantic findings for the adapter contract**

1. `mmc6.csv` contains 20,696 canonical peptides × 36 biological samples. Despite
   the `sampling_density` field suffix, all 296,483 numeric cells reproduce sums of
   `Precursor.Normalised` over peptide precursors, averaged over all declared
   technical runs (maximum absolute difference 7.45e-9). Raw precursor quantities
   do not reproduce them; averaging only observed runs disagrees for 84,011 cells.
   Preserve the original field name and explicit derived quantity semantics.
2. All 166,028 numeric `mmc8.csv` protein intensity cells reproduce deposited
   **MaxLFQ Intensity** exactly. They are not raw protein intensity.
3. Missingness needs source-level representation: 36 peptide supplement NA cells
   have a deposited reconstruction of zero. Of 65,704 protein supplement NA cells,
   42,608 have zero deposited MaxLFQ and **23,096 have positive deposited MaxLFQ**.
   Do not fill these missing cells without establishing their source-filter lineage.
4. The DIA report has 69 runs; only 67 map to the original sample workbooks.
   Two additional runs contribute 40,529 rows and must retain unresolved context.
   All 36 proteomics run names have unique punctuation-normalized deposit matches,
   but filename aliasing alone does not establish paired specimens across assays.
5. Preserve all confidence columns. Mapped rows meet per-run/library Q≤.01, while
   2,059 have global Q>.01. These are different fields, not interchangeable filters.
6. Keep background-removal and analysis-only imputation policies separate from
   observations. Keep predicted allele restriction distinct from supplied genotype.
   HAP1 is not a professional APC, and MHC-eluted abundance is not free intracellular
   peptide concentration.

Presto audit bundle:
`experiments/2026-10-24_0612_codex_hap1-quantitative-admission/` in the PR #131
working tree. It retains receipts, complete headers/sample maps, frozen replay
source and per-cell comparison dumps. No source quantities were used for fitting,
split selection or predictive evaluation in this audit. This comment does not
close #726; shared typed source/query/recipe support and missingness/lineage
qualification remain required.
