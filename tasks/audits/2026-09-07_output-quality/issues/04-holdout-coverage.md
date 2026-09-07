The canonical selected-checkpoint loss pass and prediction dump do not evaluate the same observation contract. Shared row model inputs fixed by #47 do not establish parity for bag aggregation or non-scalar outputs.

Audited at `1535b5a208b2ab59b70b83d850b1f0a7e1959537`; high priority before interpreting all-output validation/test metrics. Follow-up to #46/#47.

### Reproduced discrepancies

1. **Elution MIL uses a different prediction.** Two instance probabilities of 0.5 train a Noisy-OR bag probability of **0.75**, with BCE **0.287682** for a positive bag. The held-out CSV records **0.50** from the single row forward. This also affects presentation; `ms` is its existing elution alias.
2. **T-cell pathway-MIL responses disappear.** Their row masks are off, and the collector does not visit `tcell_mil`; an actively trained positive bag produces zero dumped T-cell observations.
3. **Categorical auxiliaries are skipped.** A supported MHC-class target with a two-logit prediction yields zero rows. Flattened prediction length differs from scalar target length, so collection silently skips it. Origin/fine-type/core CE has the same shape issue when supported.
4. **Vector BCE loses component identity and lineage.** One three-component receptor-evidence example yields sample IDs `["receptor-evidence", "", ""]`, all without a component identifier. In larger batches, the second component of sample A can receive sample B's identity.
5. **Panel coverage is incomplete.** Binding and excision panels are losses outside LOSS_TASK_SPECS and absent from collection. Selected T-cell columns are extracted but their selector/column identity is missing from the CSV.
6. **Final loss remains instance-capped.** The final pass sets `max_val_batches=0` but still forwards `max_mil_instances` (default 128). Large bags can be subsampled, while the separate row dump cannot reconstruct those loss values.

### Code evidence

- [Training MIL aggregation](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/scripts/train_synthetic.py#L1570)
- [Canonical final loss and separate row collection](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/scripts/train_iedb.py#L6013)
- [Collector flatten/skip behavior](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/training/holdout_eval.py#L548)
- [Accumulator indexing row identities by flattened position](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/training/holdout_eval.py#L280)
- [Output CSV schema](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/training/holdout_eval.py#L629)

The selected-checkpoint reconstruction and full batch traversal already work in the tested paths; preserve them. The required repair is the observation/output extraction contract.

### Acceptance criteria

- [ ] Share effective target, mask, qualifier, selector and row/bag aggregation resolution across training, support audit and held-out evaluation.
- [ ] Export one identifiable record per supervised observation/output, with bag ID/membership or sample ID, canonical output/column name, raw target, transformed target, units, qualifier, prediction and source lineage.
- [ ] Handle CE with class identity/probabilities and proper categorical metrics; handle vector BCE per component with correctly repeated source identity.
- [ ] Include supported binding, excision and T-cell panels; explicitly record absent/unsupported columns.
- [ ] Evaluate complete bags deterministically, using chunked instance forwarding if required; a training memory cap must not silently alter final metrics.
- [ ] Assert prediction support reconciles with expected effective loss support for every split and output. Aliases must not inflate independent endpoint counts.
- [ ] Recompute representative per-task losses from dumps and match the canonical selected-checkpoint loss pass under the same declared weights/regularization.
- [ ] Fail experiment closure for missing/corrupt expected output artifacts while retaining the trained checkpoint and original error.

The tests above need mixed batches and bags with unequal sizes, not only a single scalar output per row.
