# Prospective IC50-to-MS same-protein ranking follow-up

Status: design only; no new data admission, fit or evaluation. Requested by the
user's question about whether the current binding checkpoints identify native
MS ligands among other peptides from the same protein. Current IC50 comparisons
and earlier synthetic-background MS AUROCs do not answer this.

Use the existing observation manifest, sequence encoder and endpoint machinery.
Start with a frozen IC50-score evaluation, then compare explicitly supervised MS
training only under a separately frozen contract. Do not create a competing model
or trainer. Preserve affinity and observed-MS endpoints as distinct labels/losses.

Before scores, require measured ligand identity, original sample/acquisition,
source-protein accession/version/sequence/mapping and source-supported MHC pairs.
Use original unmodified human source-protein cohorts where qualified; ambiguity,
PTMs and multi-allelic samples need explicit handling rather than guessed labels.
Construct matched source-protein/sample candidate sets with length-matched controls.
Unobserved controls are not confirmed nonpresented peptides. State the sampling
ratio, candidate universe and excluded known positives, including nested ligands.

Split by source-protein homology family and sample/study where feasible; audit
overlap with IC50 training and earlier inspected MS sources. A source-frozen external
study is preferred. Freeze all settings and exclusions before viewing predictions.
Measure within-protein ranking, peptide-cluster AUROC/AUPRC, top-k ligand recovery
and per-protein macro metrics. Report sample/class/length strata and uncertainty.
Any processing claim also needs comparison among affinity-matched candidates;
MS detection alone cannot identify a physical processing rate or abundance.

Separately distinguish peptide-internal MHC-II register flanks from residues outside
the ligand in its source protein. The current IC50 input supplies the former,
not the latter. Source-protein processing context must flow through the existing
typed context/channel machinery and requires measured supervision.

Evidence: the earlier DP/DQ8 MS family uses length-matched composition backgrounds,
not same-protein controls. MHCflurry 2.0 demonstrates the general value of combining
binding and processing on held-out MS experiments (PMID 32711842); it does not
validate these Presto checkpoints. Check existing Hitlist source issues before
filing a duplicate request for missing protein/sample provenance.
