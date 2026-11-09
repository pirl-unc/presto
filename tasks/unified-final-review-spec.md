# PR131 integration review and follow-up — 2026-11-07

Base d9ed01a5 (PR131), integrating b01142f6 (PR151) in an isolated checkout.
The overall user scope is in the 2026-11-07 unified-final-model plan.

## Concrete review findings to verify before changing behavior

1. Investigate staged-elution integration with other cellular outcomes. Initial
   suspected missing `cellular_score` for tcell_response was RULED OUT: that
   endpoint has its own head, and a joint IC50/elution/T-cell forward is finite.
   Do not add an unnecessary component to fix a nonexistent failure. Continue
   checking actual quantity-domain and freeze dependencies separately.
2. Experimental selection pools different assay methods, MHC classes and alleles
   within a study. Ranking already separates methods but pools classes/alleles.
   Add an explicit prospective assay/allele
   grouping mode for new fits, preserving historical within-study reproduction.
   Test a class/assay-offset counterexample; selector hierarchy must weight classes
   equally, then allele/study/assay cells, with coverage and missing cells explicit.
3. Verify external benchmark units, allele aliases, missing study identities,
   baseline-version identity and overlap exclusions before expanding it.

## Verification

- Add meaningful regression fixtures and demonstrate failure before fixes.
- Run staged route/dependency, affinity-control, transfer and exact-resume tests.
- Use isolated imports in parent and child test processes. Existing golden numeric
  traces depend on their recorded runtime; compare baseline/candidate at identical
  runtime instead of silently weakening tolerances for a different PyTorch build.
- Record original failures, targeted fixes and validation evidence. No biological
  improvement follows merely from passing software tests.

## Verified results, 2026-11-07

- Added opt-in `within_assay_allele_spearman` selection and `assay_allele` ranking/
  batching, retaining historical modes. Counterexample prevents assay offsets from
  winning selection; all configured class/endpoint cells must be evaluable.
- Interrupted/resumed strict-mode fits reproduce reference checkpoints exactly.
- Full suite: 3005 passed, 4 expected skips, 327.63s in isolated torch2.14.1 env.
  Ruff and strict MkDocs build pass; isolated dependency check passes (71packages).
- Historical golden failure affected only attention QKV bias sums: the key-bias
  slice cancels in softmax, while floating-point gradient residue drifts under
  Adam. Unchanged base d9ed01a5 and candidate tensors match bitwise in the same
  one-thread runtime. Test retains original numerical tolerances and checks Q/V
  sums against that unchanged-base reference; historical losses and sampling stay
  fixed. No production model math was changed to satisfy this test.
- PR150 MHCflurry overlap inputs were2.2.0, predictions used2.3.0 weights; annotated
  historical results and started actual2.3.0 training/selection inventory. Installed
  NetMHC binaries are4.2c/4.3e, current releases4.2e/4.3k; new academic license
  download approval is pending, development can use existing binaries explicitly.
- Recovered all15 Nov1 fits and independently verified85,920 held-out prediction
  rows. Corrected stale v5 prose: hardware/source records establish actual v7.
  Registry closure does not promote a final model.

## Next evaluation implementation

Preserve the historical benchmark API for exact reproduction. Add a prospective
adapter with explicit executable/version/checksum identity, generic tool names,
correct current input flags, strict study/assay/allele cells, physical target
validation and common eligible rows. Include missing metadata and tool failures in
the census. Compare paired differences with study/peptide clusters; never certify
independence from absence of exact pairs alone. An unchanged-label assay-offset
control must fail to achieve within-cell correlation. Validate on small fixtures
and only then run real development rows. Locked confirmation remains inaccessible
to recipe selection.
