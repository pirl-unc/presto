# Prospective effective-target IC50 fit

Status: recipe preparation; no model has run. Agent/model: Codex / GPT-6.

This is a new family after the stopped raw-target attempt. It explicitly accepts
Presto's historical float32 `log10(clip(nM, 0.001, 50000))` objective and retains
separate raw-measurement/bound metrics. The previous family, plans, declarations
and all source/partition artifacts remain immutable. The new declaration also
records the expected membership-policy receipt change from producer 0.1.13 to
0.1.16. Its partition recipe bytes, actual membership and all source/assignment/
parent hashes stay fixed and must be recomputed by the canonical gate.

The [prospective specification](../agents/codex/plans/effective-target-fit.md)
fixes the exact cohort, optimizer, baselines, target policies, support minima,
selection criteria, resource limits and limitations before launch. Complete
source context covers citable studies, cell types/conditions, original/resolved
MHC alleles and assay endpoints/units/operators. Missing biological identity is
explicit; shared studies and unknown donors/specimens/acquisitions prevent an
independence claim. This is exploratory peptide-disjoint qualification.

Inputs: the public corrected-human Hitlist export (producer 1.78.0, manifest
`b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372`) and the exact
9,053-row Presto cohort (7,241/906/906), catalog and whole-component membership.
Only the direct IC50 output is prospectively declared. Other active objectives
remain optimized and visible but undeclared. No generated examples, live
auxiliary sources or new splits. All inputs and outputs use shared datacache.

One CPU-only d32/l1/h4 expanded model, two epochs × at most 100 shuffled batches
of 64; learning rate .001, weight decay .01, uncertainty weighting, all specified
priors/augmentations zero. Complete validation aggregate loss selects checkpoint;
test is descriptive. Train-only exact-value mean and fixed alpha=1 ridge use the
effective target space; raw errors and censor satisfaction remain separate.
The plan preserves exact regression and <=500 nM classification metrics, peptide
bootstrap intervals, every held-out prediction, training diagnostics and output
updates. It permits a failed quality screen without selecting a different rule.

Runtime contract: eight CPU, 64 GiB RAM, 512 GiB scratch, no GPU, one Torch/BLAS
thread, no workers; 90-minute work alarm and 95-minute Modal function deadline.
Launch requires the released 0.1.16 wheel and a clean, frozen recipe. The canonical
full-data support gate and target audit must pass before the first model is
constructed. Outcomes, including failures, will be retained and published before
this family is considered complete. No validation/test predictions exist yet.
