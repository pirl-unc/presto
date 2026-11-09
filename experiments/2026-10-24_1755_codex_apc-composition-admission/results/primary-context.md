# Primary measurement context

[Cui et al., primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12616400/),
MS data search and MS data processing sections, inspected through the indexed
primary text. Direct PMC rendering returned a browser challenge; the official
Europe PMC XML endpoint returned500, retained in the retrieval receipt.

The search used Proteome Discoverer3.0, SEQUEST/INFERYS and Swiss-Prot2024_06.
TMTpro labels on lysine and N termini and fixed carbamidomethylation were specified;
methionine oxidation was variable. Reported search filters include rank1,
length8–12 and q<=0.05, with a separate bovine-match exclusion.

For immunopeptidomics, intensities from matching spectra were added per peptide,
including oxidized and nonoxidized forms. Rows missing any channel were removed.
The stated channel-total normalization applies to global proteomics; no
cross-peptide row rescaling is described for immunopeptidomics.

Thus the tables can represent conditional summed reporter signal, with unresolved
form mixtures and acquisition/specimen links. They do not provide form-specific
MS1 response, pure detection efficiency or molarity. The1702observer's unmodified
MS1-to-TMT transfer is unvalidated. Source ascertainment and the2024_06search versus
2026_03mapping-reference difference must remain explicit.
