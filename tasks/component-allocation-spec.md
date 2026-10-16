# Versioned component allocation — Presto #105

The actual human preflight has 510 direct-KD observations across 119 intact
components, but 506/0/4 train/validation/test; 19 kinetics observations across 12
components all enter train. The absolute-deficit allocator sorts largest first
and initially favors the 80% partition. Overall exact counts hide this component-
size/assay-support imbalance. Fix allocation without breaking grouping or touching
any historical membership, predictions or source labels.

Add a named relative_deficit_v2 policy. Preserve component construction, descending
size order, seeded digest tie ordering, required nonempty partitions and partition
name tie ordering. For each whole component choose the partition with largest
(target_count - assigned_count) / target_count, rather than the absolute count
deficit. The existing last-components/nonempty guard remains. This is label-blind:
no assay label, model output or validation/test performance enters allocation.
It distributes early large components across requested fractions rather than
filling most of train before considering holdouts. It does not promise exact
counts when indivisible components make those infeasible.

Compatibility: the low-level allocator and Python preparation API retain
absolute_deficit_v1 defaults. Canonical frozen Hitlist training explicitly selects
relative_deficit_v2 by default through a Hitlist-specific configuration/CLI option;
users can request v1 to reproduce the old policy. The scoped-search path explicitly
retains v1. Record the actual policy, versioned algorithm identity and source code
hash in frozen receipts. External hash-pinned assignments remain authoritative and
must still preserve whole components and required nonempty partitions. Never claim
that versioned policy/producer receipts remain byte-identical across versions.

Validation: compare frozen legacy outputs exactly; show the absolute-policy size
bias on heterogeneous components and the new proportional allocation; test giant
components, unequal fractions, zero test fraction, determinism, unknown policies,
whole-component integrity and external assignments. Verify canonical Hitlist flag
propagation and scoped historical membership preservation. Run format/lint/full
suite and final-head CI, version 0.1.13, separate PR/merge/clean-main distribution.

Actual qualification is a preregistered allocation audit on the exact 9,000 public
roots from human-corpus-preflight-v1. Recover existing groups and verify v1 exactly
reproduces original assignments before comparing v2. Retain every component and
observation; do not re-cap/re-filter/construct a model. Publish both assignments,
per-assay endpoint/class/censor support, component-size distribution and overlap/
source-lineage limitations through datacache, with hashes and readable summaries.
No predictive outcomes enter this comparison. Missing biological identities remain
unknown; grouping remains limited by the frozen reviewed registry. A better support
balance is not biological independence or training readiness. The corrected 1.78
corpus is a separate generation and requires its own subsequent frozen qualification.
