Follow-up: distinguish MHC selection covariates from measured HAP1 processing responses.

The canonical quantitative source and response recipe now support measured KO/WT modeling. The frozen held-out sequence regression improved test RMSE by 9.17%, but the physical trimming extension did not add predictive value. A new, explicitly training-only diagnostic also fails its predeclared parent-context gate: CV RMSE 0.573501 versus 0.574832 for the original sequence model (0.232% gain; three of five folds improve). Position-only controls underperform the composition-containing model. This does not establish whether the sequence signal reflects processing or allele-specific selection.

[Full diagnostic, source audit and reproducibility](https://github.com/pirl-unc/presto/blob/4b135addf5f871726cc8b6f7ecb0701e3f5a57a6/experiments/2026-10-24_2018_codex_hap1-context-ablation/README.md).

The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12090245/) names **MixMHCpred 2.2**, evaluates the three HAP1 alleles, and assigns peptides using the best predicted affinity. These are predictions, not experimentally measured peptide-specific restriction. A bounded audit of nine original CSV/TSV/TXT first-line headers in supplements.zip and Search.zip finds no rank/HLA/affinity/binding field. This does not prove absence from embedded values, other assets or author-held output.

Please locate and represent the original selection predictions with a homogeneous provenance contract:

- Original per-peptide scores/ranks for each supplied candidate allele, including nonbest alleles; retain molecular-form/sequence keys, allele names and source row/file hashes.
- Predictor/version/model configuration, score type and direction, rank-reference population if documented, exact candidate genotype, assignment/tie rule, threshold and missing/unsupported statuses.
- Keep genotype, inferred best-allele assignment and measured restriction as distinct assertion types. Do not turn predictions into affinity or processing assay labels.
- If original outputs are unrecovered, record that scope explicitly. Newly computed scores must be a separate pinned derivation with input/software/model hashes, licensing/access constraints and training-overlap caveats, not retroactively attributed to the paper.
- Verify joins to all quantitative response rows without adding binding-based eligibility filters, imputing scores, changing existing splits or discarding alternate parent origins.

The immediate modeling use is an explicit selection-covariate control before another exact-versus-sampled processing comparison. It will not by itself identify intracellular peptide concentrations or validate causal machinery effects. Existing inspected outer sets remain closed.
