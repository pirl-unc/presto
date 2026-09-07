T-cell observations routed through pathway MIL train the unknown-reference scalar but discard supervision for every observed assay-panel column. #47 correctly changes row-panel losses to response BCE; that correction does not reach these bag observations.

Audited at `1535b5a208b2ab59b70b83d850b1f0a7e1959537`; scoped follow-up to #46/#47.

### Trigger and data flow

[PrestoDataset chooses pathway MIL](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/data/loaders.py#L2328) for supported assays with no explicit class and a candidate allele set spanning class I and class II.

[Collation masks the ordinary T-cell response](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/data/collate.py#L604) and [all six assay-panel objectives](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/data/collate.py#L1533) when `use_tcell_pathway_mil=True`. Masking row supervision is appropriate because a bag response is not a per-molecule label.

The missing replacement is on the [MIL loss path](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/scripts/train_synthetic.py#L2193): it consumes only `tcell_logit` and `immunogenicity_logit`. The per-instance assay selectors are retained in `tcell_mil_context`, but no selected panel prediction is aggregated or supervised.

### Verified reproduction

A positive ELISPOT fixture with two pathway instances, classes I/II:

- `tcell_mil` and `immunogenicity_mil` losses are present.
- `target_masks["tcell_assay_method"] == [0.0]`; no assay-method loss exists.
- After one backward pass with an untrained d32/l2/h4 expanded model, the **ELISPOT embedding row has exactly zero gradient**.
- The unknown-reference method row has nonzero gradient (0.013667 absolute sum for seed 13).
- The held-out collector also produces no T-cell response row for the bag.

These are deterministic code diagnostics, not training-quality measurements. Current full-corpus incidence has not been counted.

### Acceptance criteria

- [ ] Retain the bag's observed output selectors, response and identity through materialization and instance capping.
- [ ] For each supported observed axis, gather the same fixed column at every instance, apply the declared bag aggregation, and train against the measured bag response.
- [ ] Do not re-enable per-instance BCE against a positive bag label; do not send observed descriptors into model inputs.
- [ ] Share the effective selected-bag output with support counting and held-out extraction.
- [ ] Verify positive/negative bags, mixed row/bag batches, unknown selectors and multiple assay categories; measure selected versus unselected column gradients.
- [ ] Count affected source observations and per-axis/column support on real data.
- [ ] If a per-axis bag objective is intentionally unsupported, make that a declared coverage gap with explicit counts rather than silently claiming the response panels are trained.

No new repertoire/context schema is needed to implement this missing supervision path.
