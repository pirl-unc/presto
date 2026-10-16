# Explicit affinity target policies and preflight change audit — #110

## Problem and scope

The public corrected-human audit in #109 verified 211 IC50 values/bounds above
50,000 nM, including 54 exact measurements. The stopped raw-target experiment
cannot be qualified by the historical transform. The canonical census already
fingerprints raw and effective values, and OutputConfiguration already rejects a
non-default model maximum. The remaining gap is an explicit, executable policy
in the qualification contract and legible per-objective changed-target evidence.
Preserve historical numerical behavior, loss weights, source IDs and qualifiers.
No training or new target-policy choice is authorized by passing this software PR.

## Implementation

1. Introduce one frozen, callable affinity policy description, reused by the five
   existing affinity loss transforms and their panel aliases. It calls the same
   existing normalization with explicit nM input, 0.001/50,000 nM clipping, float32,
   log10(nM) output and unchanged qualifiers. Metadata comes from the same object
   as execution; do not create an unrelated descriptive range that can drift.
2. Include that policy in each applicable output-objective binding and in the
   canonical preflight source contract. Version the output-contract description.
   Preserve the existing manifest schema but reject older source contracts through
   exact comparison. Existing archived producers remain the reproduction path for
   old manifests; no implicit migration or manifest rebind.
3. During the existing canonical census, count every active affinity row/panel
   objective by split and column, including zero-support objectives. Distinguish
   total observations, exact/left/right labels, below/above clipping and clipped
   exact/left/right labels. These are objective exposures; aliases are not new
   independent measurements. Report distinct clipped source-observation identities
   separately. Count only active masks and use explicit n_* unit names.
4. Reuse the census SQLite store for changed rows only; keep unaffected totals in
   bounded per-cell counters. Persist every clipped row's objective, canonical
   endpoint, output column, sample/evidence/source IDs, peptide, citation links,
   original source endpoint/value/unit/operator, original/resolved MHC alleles,
   collated input value, effective target, qualifier and clipping direction.
   Original source fields remain text; distinguish collated float32 from original
   measurement precision. Do not retain a Python set of all source identities.
5. Write affinity_target_audit.json and affinity_target_changes.tsv before the
   manifest gate, including failure paths, alongside existing coverage/lineage.
   Bind the explicit policy even if all current observations fall inside its
   range. Preserve the ordinary exploratory path while making its policy visible.

## Verification

Documentation correction #111 also removes stale fraction-depth supervision
claims and records the shipped scoped-search trainer path. It changes no labels.

- Compare new executable transforms with historical behavior for normal values,
  both clipping boundaries, outside-range values and all affinity/panel aliases.
- Audit real collated fixtures with exact/left/right censoring, inactive masks,
  repeated source IDs and panel selectors. Verify exposure counts versus distinct
  clipped identities and exact exported lineage/raw/effective values.
- Require a missing, changed or stale policy to fail prospective qualification,
  including data where no value clips; preserve reports on failure.
- Reuse the public frozen IC50 rows for a bounded canonical policy audit; its 211
  changed observations and 54 exact values must match the independent #109 table.
  This is data qualification only, with no model construction or predictions.
- Format, lint, focused tests, full ./test.sh, strict docs, final CI, review, merge
  and verified clean-main distribution. Version 0.1.16. Rebase the stacked branch
  onto merged #109 before final review.

## Follow-up fit

A later, separately registered prospective family can use a freshly qualified
manifest with the explicit policy. Baselines must use its declared effective
space, while separate raw-measurement errors and censor satisfaction remain
visible. The original stopped attempt, plan, threshold rules and data partitions
stay immutable. #48/#53 and Hitlist #678 remain open until evidence supports the
claimed outputs and held-out performance; unknown biological identity is unchanged.
