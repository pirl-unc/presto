# PXD057588 quantitative transform reconciliation

October 22, 2026; Codex / GPT-6. Follow the initial source inventory at Presto
51b895e. The authoritative Hitlist context migration is underway in a separate
worktree; do not introduce a competing normalized metadata system here.

## Question and prespecified comparison

Can the published primary log2-fold-change panels be reproduced from the exact
reported intensity cells, using the distinct immunopeptidomics and global-proteome
policies described in the primary paper? A matching source label does not prove a
matching target transformation. Reconcile before choosing a biological loss.

Use the existing seventeen-asset source bundle without changing any bytes.
For each Table 7–12, preserve the primary sequence block and channel/sample map.
Read data rows 6 onward. For Tables 7, 8, 9, 10 and 12, compute log2 of each treated
channel divided by the arithmetic mean of its same-peptide untreated controls.
Table 12 has distinct GL261 and CT2A controls. For Table 11, first divide each
channel by its sum over all retained digest-peptide rows, then use the same
control-mean ratio (the common multiplicative mean-total factor cancels).

Compare against all 200,724 published primary fold-change cells with absolute
tolerance 1e-8. Preserve per-cell source/calculated values and errors; do not tune
normalization after seeing discrepancies. As a diagnostic negative control, apply
total-signal normalization to each elution table and omit it from the proteomics
table. Report error distributions and mismatches for both policies. Wrong-policy
results are not candidate training targets.

Audit source target variation and parent-label support: distinct literal parent
labels, singleton versus multiple-peptide groups, and overlap between Tables 9
and 11. Count support before proposing within-parent distribution learning; a
singleton observed candidate set provides no within-parent distribution signal.
Parent strings remain unvalidated source labels, never exact reference mappings.
No missing peptides are fabricated, and absence is not a measured zero.

## Outputs and verification

Register a new dated source-qualification family with source hashes, frozen script,
launch command/environment, aggregate/per-table JSON/CSV, compressed per-cell
comparison dump, parent-support counts and a narrative decision. Independently
recompute declared transforms from original workbook cells using scalar math.
If the paper policy disagrees with source values, preserve the discrepancy and
defer that target; do not silently fix the paper or overwrite the source.

This is measurement reconciliation, not predictive fitting: no model, pretraining,
loss, synthetic labels, GPU, train/validation/test split or predictive metrics.
Numeric reconciliation errors describe transformations, not model generalization.
The observable is relative presented/digest-peptide signal; concentration and
precursor-normalized native availability remain unqualified. The next biological
fit still requires the common metadata/export path and endpoint admission.
