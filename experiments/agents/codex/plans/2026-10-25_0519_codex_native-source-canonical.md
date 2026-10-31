# Canonical native-source replay before biological fitting

Question: does the shared Hitlist EvidenceStore preserve the qualified PXD018722
Hi20S source without creating artificial concentration labels, parent certainty or
biological replicate identities? This is a source-admission audit, not training.

1. Freeze the source adapter at Hitlist base 8008fd713bfdb8f340bd635cc830803662c9a186
   with its complete changed-file overlay and the 15 original asset hashes.
2. Build a native package through the public writer. Input staging uses symlinks;
   package snapshots use independent copy-on-write files where supported. Preserve
   failures and never overwrite a package. No source downloading or shared-env edit.
3. Independently compare all public quantities, entity/native-ID links, declared
   parents, complete reference records and every origin against original sources
   and qualified family0424. Compare all original text/XML/workbook records/cells.
   Keep raw and LFQ, sequence/form/feature grains, zero/missing and MS matching
   distinguishable. Check date-corrupted annotations without inferring gene names.
4. Reconstruct source-wide dependence using a graph traversal independent of the
   compiler; include declared alternatives from all views. Compare membership to
   the earlier peptide-only audit, explicitly reporting any additional links.
5. Check every public relation's primary key and relationship target closure.
   Report study-scoped replicate claims separately from unresolved per-acquisition
   specimen/pairing. No engineering/reporter sequence is a predictive target.
6. Close the registered audit with metrics, exact source/runtime/environment paths,
   reproducibility artifacts and a canonical log entry. Run software gates separately.

Local limits: 20 minutes per stage; 3 GiB process-tree RSS; 1 GiB free disk plus
32 MiB startup margin. Watchdog checks every second and terminates violations,
retaining logs/receipts. Full software suite requires at least 2 GiB disk headroom.
Do not lower guards. Retain all failed attempts and any revised verification contract.

Training/pretraining: none. Assays: native intracellular MS source signal only;
no MHC elution, affinity labels, engineered-reporter targets or molar calibration.
No synthetic biological data, losses, tuning, train/validation/test split or
prediction dumps: no predictor is fitted or evaluated. Software fixtures are
explicitly separate. This audit cannot prove APC transfer or availability accuracy.

Next: tested canonical release, explicit model-facing curation contract, then the
registered same-parent contrast fit. Matched differentiable/discrete experiments
remain contingent on a useful biological representation; this audit does not
establish that condition or complete the availability goal.
