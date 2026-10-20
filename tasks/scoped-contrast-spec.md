# Scope-aware contrast groups — Presto #81

## Problem and boundary

The sampler validates variation before truncating a group to eight rows. Eight
positives followed by one negative therefore yields a nominal contrast group
with no opposing outcome. The same defect can remove machinery variation.
Detectability grouping also ignores captured bundle/search/protocol/reference
identity and uses generic label buckets, which can describe excision instead.

## Change

Choose at least two genuinely distinct comparison categories before filling the
remaining capped group slots. Detectability candidates must have finite binary
detectability targets and a known parent; the grouping key includes parent,
machinery and the complete scoped identity. Scoped records require complete
bundle/search/protocol/reference fields. Legacy records stay in a separate
explicitly legacy source stratum; excision-only rows never create detectability
contrast groups. Parent-masked ambiguous peptides stay eligible for ordinary
sampling but do not acquire arbitrary parent comparisons.

Protease groups must have explicit excision labels and retain machinery variation
after capping. Scoped detectability-only records do not enter protease groups.
This preserves existing comparison intent without claiming that co-batching
identifies intrinsic detectability, causality or biological independence. No new
labels, training objectives, synthetic examples or source acquisitions.

## Validation and release

Reproduce the eight-positive/one-negative failure, assert actual emitted batches
contain both outcomes, and cover different scopes, invalid/missing labels,
excision-only records, ambiguous parents, deterministic behavior and caps. Cover
machinery variation retained after truncation and legacy grouping. Compare the
full existing sampler/loader suite, then formatting, lint and complete tests.
Version 0.1.5; review, PR, merge and verified public GitHub distribution release.
Presto #70 remains open for canonical frozen membership and actual fitting.

## Actual-batch review correction — #86

The new emitted-batch test also exposed over-allocation: contrast rows were added
on top of full task quotas. Keep contrast groups within the same assay/task and
debit each selected group from its task allocation before ordinary draws. A task
with fewer than two available slots cannot receive a contrast pair. Preserve the
exact batch size, per-task coverage when feasible and both contrast categories;
never truncate a completed batch to repair an over-allocation. Test small and
normal batches over repeated epochs with mixed tasks.
