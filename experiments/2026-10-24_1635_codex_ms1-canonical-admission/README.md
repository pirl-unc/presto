# Canonical measured nontryptic MS1 source admission

Codex / GPT-6, October 24, 2026. **Source and recipe qualification passed.**
This supplies a shared non-MHC observation endpoint; no model was trained and no
new predictive improvement is claimed. [Prospective plan](../agents/codex/plans/ms1-canonical-admission.md).

## Data and target

Original PXD021013 chemically synthesized nontryptic peptides, primary study
[DOI10.1038/s41467-021-23713-9](https://doi.org/10.1038/s41467-021-23713-9), two DDA
pools selected before quantities in the prior 1546 audit. Preserve original archive,
article, intended-product workbook, deposited README and community SDRF bytes.
The full intended table has 305,331 mapping rows across 421 pools. Source hashes and
original paths are in reproduce/archive-receipts.json and results/source-manifest.json.

Crude synthesis and estimated 200/500 fmol injection do not establish individual
molar amounts. These are synthesis/handling/LC/MS measurements, not APC processing,
MHC pull-down, pure ionization efficiency or intracellular concentration. HLA library
origin is audit context, not a restriction label. Search digestion is not lab digestion.
The source preserves primary synthetic/Lumos protocol context separately from
conflicting SDRF cell/Fusion/HCD/run1 claims, repeated positional modification columns,
fixed/variable modifications and unresolved load/purity/specimen information.

Canonical implementation: Hitlist feature worktree
`/private/tmp/hitlist-proteometools-ms1`, based on bafa4fa87e0a3d5ece9093a285a9c02f06d57148,
with the exact dirty source frozen under reproduce/source/hitlist. Package family
proteometools_ms1/schema 23/catalog 28, candidate version 2.0.26. The new registered
recipe is pxd021013_intended_unmodified_ms1_v1, accessed through EvidenceStore.

Require intended full-length pool membership, canonical sequence, positive finite
sequence/form signals, PEP in [0,.01), no reverse/contaminant flag and exactly one
Unmodified variable-modification form; fixed CAM remains explicit. Preserve all
form dispositions and all planned product/run statuses. Output raw log10 intensity
and its within-run centered value. The target center is audit-only; predictions
must be centered over the same conditional candidate set. No missing zeros,
intensity/CV filtering, imputation or true molar labels are invented.

| Pool | Original peptide rows | Original form rows | Admitted products |
|---|---:|---:|---:|
| TUM_HLA_1 |1,961|1,963|677|
| TUM_HLA2_1 |1,688|1,775|470|
| Total |3,649|3,738|1,147|

Of 1,500 intended product/run pairs, 1,147 are admitted, 204 observed but excluded and
149 unobserved with unknown cause. All 3,738 original forms retain dispositions;
2,591 are excluded. Multiple exclusion reasons may apply to one row.

## Verification and dependence correction

Independent pandas decoding agrees with all 320,222 original peptide/form literal
cells and all typed identity, value and flag projections. All 1,147 example identities
and log10/centered targets independently reproduce; counts match the earlier
independent membership audit's 677/470 form candidates. Original source IDs are
neither dropped nor multiplied. Source/recipe verification replays derived outputs.

The initial conservative dependence graph connected all observed identities,
including 106 shared background sequences. Independent inspection shows none of
those 106 is intended in either pool; intended and admitted intersections are empty.
They cannot enter this endpoint's features, targets or normalization. Before any
model fitting, the contract was corrected to connect **all intended products**, even
unobserved/failed-quality ones, across selected pools. Two intended components
remain. Every example/target is unchanged except component identity. The initial
recipe generation and summary remain in results/initial-grouping and the raw recipe
directory; [amendment](reproduce/dependence-amendment.md), original overlap and final
independent comparison are retained. This does not establish study/specimen independence.

Two original-layout failures were also preserved: MaxQuant summary includes
experiment/overall totals, and parameters.txt repeats several settings. The parser
now selects the raw run by its exact deposited mapping, preserves all subtotal and
parameter rows, and rejects ambiguous extra runs/conflicting required settings.
No quality threshold or source value changed. Both failed source snapshots and logs
are under reproduce/attempt-1,attempt-2 and launch_logs. Synthetic regression fixtures
exercise these source semantics but never become biological training data.

## Reproduction, resources and artifacts

Presto launch commit a39d527bedf16bade33bd615bd5c227b44e5b821 with recorded task dirt.
Run reproduce/launch.sh for the original canonical admission and then the exact
analysis/refine_groups.py invocation in results/dependence-verification.json with
its separately frozen final recipe source. Restore the relevant Hitlist source
snapshot/diff; neither command overwrites existing immutable output directories.
PYTHONPATH explicitly selects the feature worktree; shared Python 3.12, local CPU,
requested OMP/BLAS threads 1 for the main launch. No Modal/GPU or package installation.
The repository hook had refreshed the existing shared editable checkout earlier;
its target was preserved and pip check passed. This experiment does not alter it.

- Source: `data/measured_ms1_source_admission/2026-10-24_1635_v1/source`.
- Original grouping: same root `/recipe`; final: `/recipe-intended-groups`.
- Original source bytes 117,129,479; source/recipe files under 1 GiB artifact cap.
- Build+independent original-cell checks 27.78 s, peak 1.011 GiB process RSS.
- Final dependence replay 9.89 s, peak 1.116 GiB, below 3 GiB cap.
- Source manifest SHA256 fa32ddc4ec30675dea28ba873dbf2667954833183dcfabf60f61d6364f7454ff.
- Final recipe manifest SHA256 181ea4fb2e25a8c2035495dc7ee361da58d3a6c0754f9ed291ce77672aa221f5.

Results include source/recipe manifests, every example, every form/planned-product
disposition, group assignments, original-cell and dependence verification, frozen
source/hash receipts and all launch logs. Initial failed attempts have no fitted
checkpoint. No pretraining, optimizer, loss weights, model outputs or held-out
predictions exist. No train/validation/test split was used because this is source
admission; affinity/binder metrics do not apply.

**Decision:** canonical admission works. The next step is a prospectively selected
larger pool cohort and grouped sequence-versus-constant/length prediction of this
conditional signal. These two pilot pools are development evidence, not sufficient
independent evaluation. Freeze an independently evaluated observer before later
matched exact/sampled processing comparisons. APC peptide availability and
precursor-relative concentration remain unvalidated; the broader goal is active.


## Implementation checks and publication

The resulting Hitlist implementation is [PR814](https://github.com/pirl-unc/hitlist/pull/814),
commit b9b10470058e32e679652c6fd405cc834b649bf3 (version 2.0.26), with release
still pending. `format.sh`, `lint.sh` and `mkdocs build --strict` pass.
The final `./test.sh` run passes all 3,458 non-integration tests in 325.27 seconds;
process peak RSS is 926,728,192 bytes. It uses one worker and the original memory
preflight outside the sandbox. The initial sandbox preflight refusal and earlier
six catalog-version assertion failures are preserved in compressed logs. The
corrections change the live catalog expectation to 28 while retaining historical
package versions. Full integration/release gates are separate and pending.

Original git diffs are archived as byte-identical `.patch.gz` files. Their context
lines contain meaningful leading spaces; compression preserves those bytes without
misclassifying the patch content as whitespace errors in the enclosing commit.
