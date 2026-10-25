# Execution-order amendment before biological fitting

The original registration required both PyPI releases before preparation. That
was a self-imposed engineering sequence, not a scientific selection rule or a
user requirement to delay local experiments. Hitlist AGENTS.md requires work to
be merged and deployed before it is called done; publication remains required
and outstanding. This amendment does not declare that work complete.

Before any biological fit or held-out effect summary, permit the frozen local
screen to run against the exact merged, fully verified code while the serialized
release builds finish. Required evidence replaces the release-before-fit gate:

1. PR841 is merged as d02ca1aefc0e2753843bff842cfb773273226f03. Its complete Git
   tree equals fully tested head90a7a62. Independently compare both tree IDs.
2. All eight final-head test jobs pass: lint, integration, coverage and five
   Python versions. The docs build passes. Docs deployment is deliberately
   skipped for PR events by its checked-in push-to-main condition.
3. All six preregistered native-recipe code hashes match the merged commit and
   the source-admission execution. The input manifest and every recipe artifact
   retain their pinned hashes; all three source-support gates already passed.
4. Persist these exact code/CI/source bindings before fitting and verify their
   hashes before each subsequent stage. Keep original launch/config/prepare
   files and the original release-first policy for a complete audit trail.

Source release40168710669 remains running; recipe release40169827556 is pending
behind it in the same main-branch concurrency group. Neither is restarted or
cancelled here. Both tested publications remain outstanding completion work.

No data, features, models, hyperparameter grid, selection rule, split, weighting,
bootstrap seed, objective, success threshold or numerical replay tolerance changes.
The data-free qualification already passed on the unchanged production fitting
code. This is an execution-order amendment, not a post-outcome model revision.

The qualification completed in less than one polling interval; its 2.6MiB
observed RSS is not a useful peak-memory estimate. For subsequent stages, the
separate local launcher polls every50ms and also records OS child-process peak
RSS after wait. An observed or terminal peak above the same3GiB budget fails the
stage; no resource limit is relaxed. The original qualification receipt stays
unchanged and is not used for memory-performance comparisons.

Use `reproduce/launch-local.sh prepare`, then `fit`, `evaluate`, and `replay`.
The original launcher remains release-gated. Numerical modules and their frozen
qualification snapshots must remain byte-identical.
