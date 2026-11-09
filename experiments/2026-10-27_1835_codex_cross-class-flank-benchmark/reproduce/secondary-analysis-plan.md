# Sequence-only audit and secondary strata

Recorded before any profile update or predictive fit. Main numerical conditions,
splits and primary metrics remain fixed. No predictive scores have been inspected.

The sequence-only audit found whole-peptide separation but extensive9mer substring
sharing:814/958 class-II test observations share at least one9mer with class-II
training;502/1845 class-I test observations do with class-I training. This is not an
experimentally assigned binding-core annotation and does not itself prove leakage.
All same-class held-out MHC sequence pairs occur in training. Cross-class sharing
is1/958 (Itrain→IItest) and8/1845 (IItrain→Itest); source training sets have no exact
peptides shared between classes and only22 shared9mer strings.

Preserve main metrics on all original holdouts. Additionally report, without any
model selection/tuning, exact regression metrics on observations with/without a
nine-residue substring seen in the model's training classes. Report per-MHC-pair
metrics and macro-average Spearman where defined, alongside pooled metrics.
These secondary analyses diagnose sequence familiarity and allele stratification;
they are not substitute new test splits. Any future core/sequence-family-disjoint
benchmark must be a separate prospective family with new training.

Script: analysis/audit_sequence_overlap.py. Results: results/sequence-overlap-audit.json.
