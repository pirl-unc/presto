Parent #723. Complements #759 (cellular MAPP perturbations), #757 (independent precursor context), and #762 (MS response calibration). This requests a shared Hitlist source and recipe, not a Presto-only workbook adapter.

## Concrete source and modeling need

The current cellular MAPP composition comparison has not beaten uniform on held-out data. A controlled substrate-to-products time course would let us test whether a model can learn peptide production and subsequent removal before introducing APC capture and MHC selection.

[Soh et al., Nature Communications 2024](https://doi.org/10.1038/s41467-024-45339-3) report quantitative digestion of 15 full proteins by purified human 20S standard proteasomes. [MassIVE MSV000092631 / PXD044463](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000092631) is public and lists:
- `other/aSPIRE_outputs_zip.zip`: 963,763,954 bytes.
- `metadata/Identifier.xlsx`: 28,339 bytes.
- `methods/PSP_proteins_PRIDE_v2_JL.docx`: 20,209 bytes.
- 15 non-resource-fork substrate FASTAs under `sequence/AA_Sequences/`.

The public file inventory has 505 entries; it has been retrieved and parsed, but the output ZIP/table schema and exact sample crosswalk have **not yet been admitted or independently replayed**. Its recorded license is CC0. Prefer extracting the original processed tables and metadata before downloading raw spectra or the 7.8 GB spliced-search database.

The paper describes unnormalized MS1 peak-area measurements followed by background filtering/subtraction and averaging technical replicates within biological replicates. Most substrates have 0, 1, 2, 4 and 24 h measurements; alpha-synuclein has a different schedule. Products without a positive 4 h value (3 h for alpha-synuclein) were removed from the final quantified panel. These are selected relative MS responses, not molar peptide concentrations.

## Shared curation requirements

1. Preserve original bytes, hashes, source/member/row coordinates, file-to-run-to-digest relationships, biological preparation versus technical acquisition, substrate sequence/construct/reference, enzyme preparation/isotype/species, substrate species, input concentrations and enzyme:substrate ratio, buffer, temperature, time and termination procedure. Purified enzyme origin (e.g. a cell line) is not an APC specimen or a gene-knockout condition. Unknown links stay unknown.
2. Preserve all original quantity layers where available: precursor charge/modification/isotope contributions, measured peak area, background, corrected signal, technical-replicate aggregation and final selected values. Represent zero/unreported/below-background/removed/not-assayed separately. Recover excluded rows from upstream outputs when possible; never turn the final selected panel into a complete negative set.
3. Replay the publication's background rule using original contributors: per-peptide maximum at time zero, global median cutoff, threshold/removal and subtraction, technical averaging, then terminal-time selection. Confirm the implementation from the deposited outputs; retain conflicts instead of silently forcing agreement.
4. Keep contiguous hydrolysis products separate from cis/trans-spliced products, synthesis errors and contaminants, while retaining their measurements and ambiguous mappings. Preserve all possible sequence/position assignments and original I/L handling. A simple span model may exclude spliced/ambiguous rows under a named policy, but must not relabel them as negatives or duplicate quantities across mappings.
5. Expose a registered time-course recipe through the common evidence query/recipe interface. Keep fragments shorter than typical MHC ligands and nonbinders; no affinity filter. Declare the source's 5–40 residue quantification/search scope and all additional selection. A normalized composition target and total captured/detected signal are distinct endpoints; neither establishes total molar production.
6. Freeze source/recipe manifests, all admission and exclusion reasons, original versus derived values, and suitable substrate/preparation/run split constraints. Assess time-course repeatability before a new exact-enumeration versus REINFORCE experiment. Keep related fragments, modified forms and repeated acquisitions grouped, and report the small number of independent parent substrates.

## Acceptance

- Source inventory and documented access/coverage, including missing raw versus processed layers.
- Independent replay of quantitative derivations and all retained source joins.
- Homogeneous context/quantity/selection fields and original-source round trips; no separate experiment-only curation schema.
- A declared endpoint that supports a bounded local kinetic test, with prospective validation/test groups and missingness policy.
- Explicit limits: purified standard-proteasome digestion does not measure APC trafficking, TAP, cellular clearance, MHC binding, or intracellular concentrations. MS response and source selection still limit interpretation.

InvitroSPI / ProteasomeDB (doi:10.1038/s41597-022-01890-6, Figshare 17143865.v1) is a related candidate for identification support, but its documented 26-column table supplies PSM scores/identities rather than a quantitative abundance column. Do not use PSM counts as concentration targets by default.

No author outreach is requested.
