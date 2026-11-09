# Post-PR153 unified retraining and performance reporting

Canonical implementation and scientific specification:
[`tasks/post-pr153-unified-retraining-spec.md`](../../../tasks/post-pr153-unified-retraining-spec.md).

Execution order:

1. review, verify, and merge PR #153;
2. branch from the verified merged base in a clean worktree and open the
   successor PR;
3. implement and test deterministic timestamped report generation;
4. obtain and pin licensed MixMHCpred 3.x, then validate PRIME 2.1 through
   mhctools;
5. freeze newly disjoint affinity, presentation, and T-cell cohorts;
6. freeze a compact data/mechanism/calibration ablation matrix;
7. train sequentially on explicit `H100!`, preserving per-example predictions;
8. run matched NetMHCpan/NetMHCIIpan/MHCflurry/PRIME comparisons only after
   checkpoint and calibration freeze;
9. publish the timestamped state report and promote only if all gates pass.

No current development or consumed locked cohort becomes a new TEST. Large or
license-restricted data remain in immutable artifact storage with hashes rather
than ordinary Git.
