# Prospective effective-target IC50 fit

Status: frozen before launch; no model result exists. Agent/model: Codex / GPT-6.

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

## Frozen declaration and producer

The [prospective contract](https://github.com/pirl-unc/presto/releases/tag/effective-target-fit-contract-v1)
publishes 11 independently verified assets, including the paired old/new
manifests and membership-policy receipts. New manifest SHA-256:
`a90c6d620554635079a5e8ec5927ec3050822d8fa7327279d54327924c7bd661`.
Its full gate must still pass on the actual cohort before model construction.

Actual package producer is released Presto 0.1.16 (`e35d759`), wheel SHA-256
`0eecf96f12308580e5f59047c77364a2b5386cc5b1b133527dcb34e3ec8253a5`,
with Hitlist 1.78.0 and the unchanged pinned consumer environment. The repository
PR version 0.1.17 records this experiment and is not the fitted package producer.
`reproduce/producer.json`, `frozen-inputs.json`, `launch.json` and `source/` freeze
the exact inputs and executed source. The clean launch commit is recorded by the
launcher; no mutable wheel, manifest or source fallback is permitted.

Validation before launch: nine experiment checks pass against released 0.1.16,
including effective/raw metrics and pre-model gate refusal. Full repository
regression passed 2,281 tests (four skips); formatting, lint and strict docs pass.
These checks are prerequisites, not predictive evidence.
