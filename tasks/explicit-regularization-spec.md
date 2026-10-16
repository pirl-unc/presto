# Explicit effective regularization — Presto #95

The first real scoped fit exposed a hidden binding-orthogonality weight of 0.01.
Keep that existing numerical default for compatibility, but declare it in the
canonical defaults and expose it through direct and unified/synthetic CLI routes.
Explicit zero must remove the term. JSON/YAML config and explicit CLI precedence
must remain consistent; reject nonfinite/negative regularization weights before
loading data or creating a model. Audit every inherited regularization key against
the canonical saved defaults. Also declare balanced_batches=True: direct Namespace
runs currently use that fallback without saving it, as observed in #94's run.

Before any data/model work, resolve and validate the effective regularization
mapping. Save a regularization.json receipt alongside config.json even for
preflight: base parameters, epoch schedule policy and intended epoch-specific
mappings. Training/evaluation must use that same base mapping. The receipt is
configuration, not proof that every configured loss has labels or fired; actual
metrics/output updates remain the evidence of execution. Do not change scheduling
semantics, supervision support, sampler behavior or existing experiment files.

Regression tests must exercise direct/unified parser parity, config/CLI override,
explicit zero versus nonzero real loss computation, invalid-weight early failure,
all inherited keys represented, and an actual canonical preflight writing the
receipt without a model. Version 0.1.9; format/lint/full suite/strict docs, reviewed
PR, merge and verified distribution. Any rerun needs a separate prospective
registration, with the original failed-quality/deviating run retained unchanged.

CLI precedence verification exposed #96: explicit values equal to defaults are
silently overwritten by configuration. Track explicit option destinations from
the selected command parser (including nested unified/synthetic commands), use
that metadata in config merging, and keep Namespace callers without metadata
compatible. Test omitted versus explicitly default-valued and zero weights.
