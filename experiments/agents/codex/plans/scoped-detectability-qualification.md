# Prospective bounded within-search qualification

Agent: Codex (GPT-6); CPU only, one Torch/BLAS thread, no GPU. Use the canonical
Presto trainer after the Trypsin/P vocabulary and internal-site scoring fixes.
Source: public PXD013455 HeLa Trypsin/P 46-fraction bundle, manifest SHA-256
441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b.
Original/reanalysis citations, cell conditions and protocol are retained by the
source registry. MHC alleles do not apply. Fractions are not biological replicates.

Select 10,000 globally collapsed peptides, data seed 42 (canonical scoped sampling
uses data_seed + 23), freeze 80/10/10 train/validation/test with split seed 42.
Claim only peptide-disjoint within-search generalization. Protein overlap is
reported; experimental/donor/specimen independence is not established.
No generated negatives, excision labels, depth targets, live MHC/UniProt inputs or
mixed-source augmentation. MS/MS, matching and absence remain proxy categories.

Prospective supported output: ms_detectability_logit only, proxy evidence from
scoped_msms/scoped_matching/scoped_unobserved in the one exact source. Require at
least 500 distinct peptides/unique proxy observations and at least 50 positives
and 50 negatives in each split. These thresholds are fixed before preflight; do
not lower them after inspecting counts. Every other output stays undeclared.

First run canonical admission/coverage preflight without constructing a model.
Then freeze its exact source contract into the prospective supported-output
manifest and require the same split-support fingerprint for fitting. Fit a small
expanded model (d_model 32, one layer, four heads), two epochs capped at 100
training batches/epoch, batch size 64, seed 42, learning rate 0.001. No pretraining;
all consistency losses zero. Evaluate full held-out validation/test membership
using the selected checkpoint. Compare with train-prevalence and peptide-length/
amino-acid-composition logistic baselines fitted on train only. Preserve each
held-out prediction and compute log loss, Brier, AUROC, AUPRC, balanced accuracy,
precision, recall and F1. Binding regression/500 nM metrics are not applicable.

Per-stage wall-clock limit 30 minutes; record runtime, runner peak RSS (child
processes excluded), versions, code commit/dirty state, exact arguments and all
hashes. All reusable source/partition/run artifacts use datacache; only small
summaries belong in Git. Publish downloadable archives and per-file inventories.
This bounded exercise cannot establish general proteomic detectability or full
multi-output training readiness. A failed support/fit/evaluation gate remains
visible and gets an issue; do not mask it by silently changing the cohort.
