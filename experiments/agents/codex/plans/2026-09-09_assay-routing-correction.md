# Full-source assay-routing correction audit

Agent/model: Codex / GPT-6. Implementation contract:
[`tasks/assay_routing_spec.md`](../../../../tasks/assay_routing_spec.md).
Experiment: `experiments/2026-09-09_1133_codex_assay-routing/`.

Compare production merged-input classification and constructed record payloads
before and after #59, following the blocking finding in the output-coverage
inventory. Use the identical full `data/merged_deduped.tsv` (SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`), no study
exclusions, no generated rows, no changes to the file or upstream cache.

Use the existing isolated pinned environment under
`artifacts/2026-09-09_1106_codex_output-coverage/.venv`, recording its versions
and source hashes. Local CPU, OMP/MKL threads one, no GPU or Modal contract.
Instrument canonical classification and pre-cap append calls to count every
row and fingerprint every constructed typed record in source order per assay
descriptor group. Retain only the first record per modality to bound memory;
all reported evidence counts are pre-cap. All original functions are restored
in `finally`. Reconcile classification counts to loader statistics and pre-cap
record counts to observed append calls. The audit must not replace production
classification, labels, sampling or record construction.

Record full-source descriptor/response/route groups, representative source
descriptors, per-modality counts and payload hashes, all drop reasons and source
stability hashes before and after. Compare unchanged group fingerprints and
explain every routing transition. Preserve both failed and successful receipts.
Never overwrite an output directory or frozen launcher. Each phase is launched
only after committing its exact implementation, and before editing production
code while the previous phase is live.

No pretraining, optimization, synthetic data, splitting, MHC filtering or model
evaluation occurs. No loss terms/weights are changed by this experiment, and no
validation/test metrics or per-example predictions are claimed. Effective
training eligibility and per-output supervision coverage remain in #48/#50;
known-example fitting and held-out generalization remain in #53. Close the
available before/after evidence in the experiment README and canonical log.
