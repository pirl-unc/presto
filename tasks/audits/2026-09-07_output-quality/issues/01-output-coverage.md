The existing support and gradient gates cannot establish that every declared output is trained. They can pass while entire output columns have no examples or only one response class.

This is a scoped follow-up to #46 after #47, audited at `1535b5a208b2ab59b70b83d850b1f0a7e1959537`. Priority: high for any all-output training/quality claim.

### Verified gaps

- [Split support](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/training/data_support.py#L65) enumerates `batch.target_masks`, while the [loss registry and resolvers](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/scripts/train_synthetic.py#L173) also derive MHC targets and train bag objectives and extra panels. Those effective objectives are absent from this census.
- The six T-cell axis tasks now use BCE, but [DEFAULT_BINARY_TARGETS](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/training/data_support.py#L18) omits them. Counts are per axis, never per selected column.
- `require_all_active` requires only targets observed somewhere; an output with zero support everywhere is not required. Gates default off; the minimum requested support defaults to one.
- Counts do not distinguish measured labels from synthetic negatives, cascades or biological proxies.
- [Gradient tests](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/tests/test_gradient_coverage.py#L364) sum over complete named parameter tensors after three constructed-fixture updates. One live embedding row can hide all the unused rows. `core_start` has an explicit no-source exemption.
- The default model emits 26 binding descriptor columns and 56 T-cell descriptor columns, plus aliases and diagnostic tensors. Vocabulary size is not evidence of label coverage.

### Reproduced gate blind spot

For **each** train/validation/test split, provide an ELISPOT-positive sample and an ICS-negative sample. Calling:

```python
validate_split_support(
    audit_split_support(splits),
    require_all_active=True,
    require_all_active_binary_balance=True,
)
```

passes. Each observed method column is one-class, most method columns have zero examples, and the axis report has no positive/negative counts. A T-cell pathway-MIL fixture also trains a response but reports no response support. A labeled MHC-class fixture is absent from the support report.

### Concrete existing source evidence

The [PR #45 merged preflight](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/experiments/2026-09-05_0957_codex_pr45-integrity-e2e/README.md#L17) scanned 3,266,972 source rows. Its capped, filtered 173/57/57 split had EC50 support **1/0/0**, Tm **1/0/1**, receptor evidence **13/3/6**, all positive, and no receptor-method support. These are historical counts for that exact contract, not current full-corpus coverage.

The [current Hitlist adapter](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/data/hitlist_source.py#L1324) returns no measured processing, T-cell or receptor-evidence records. Synthetic cascades cannot fill those measured-label gaps.

### Acceptance criteria

- [ ] One executable output/supervision registry identifies canonical endpoint, aliases, shape, units/transform, selectors, masks/qualifiers, allowed source families, direct/proxy/synthetic/auxiliary/diagnostic status and supported configurations.
- [ ] Generate a complete declared-output matrix, including explicit zero-support entries, from effective row/bag/panel supervision after curation and splitting.
- [ ] Count source rows, distinct observations/peptides/alleles, class/species/context, exact/censored labels, and positive/negative or graded labels **per column and split**. Separate real and each generated-label family.
- [ ] Include per-column gradient/update coverage on real training batches; distinguish zero initialization, frozen stages, unused columns and intentional nontrainable quantities.
- [ ] Enforce a prospective supported-endpoint manifest; report unsupported/underpowered outputs explicitly instead of treating any nonzero count as adequate.
- [ ] Audit effective loss multiplicity/weights: MIL currently optimizes both `elution` and its `ms` alias. Document intentional weighting and prevent aliases inflating coverage.
- [ ] Cover default merged, exclusive Hitlist, optional bulk-MS and explicitly supported alternate output configurations. Do not silently union sources.
- [ ] Keep direct measurement support distinct from indirect gradients: presentation versus elution, recognition versus T-cell response, excision/detectability proxies, and organism-derived foreignness require separate validation claims.

This issue supplies measurement and gating infrastructure; it does not require a direct biological label for every internal tensor or close the broader context schema in #46.
