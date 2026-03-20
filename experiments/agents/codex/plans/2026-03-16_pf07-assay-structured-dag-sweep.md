# PF07 Assay-Structured DAG Sweep

## Question

Can an output-side assay-structured DAG beat the current honest seq-only PF07 control on the 2-allele broad-numeric binding contract?

## Why This Experiment

The current honest PF07 baseline is:

- sequence-only on input
- multi-output on the affinity side
- still structurally flat at the assay-output layer

Right now the head family is basically:

- shared `KD` base
- direct residual outputs for:
  - `IC50`
  - `EC50`
  - `KD_proxy_ic50`
  - `KD_proxy_ec50`

That is better than a single-output head, but it is still not the biologically informed output DAG we want. This experiment tests whether adding explicit output-side structure helps without violating the no-assay-input rule.

## Fixed Contract

- Codepath: main `Presto` affinity path via `scripts/focused_binding_probe.py`
- Inputs:
  - `nflank`
  - `peptide`
  - `cflank`
  - `mhc_a`
  - `mhc_b`
- Forbidden inputs:
  - assay type
  - assay method
  - assay prep
  - assay geometry
  - assay readout
  - any `binding_context`-driven predictive input path
- Dataset:
  - source: `data/merged_deduped.tsv`
  - alleles: `HLA-A*02:01`, `HLA-A*24:02`
  - measurement profile: `numeric_no_qualitative`
  - qualifier filter: `all`
  - split policy: peptide-group `80/10/10`
  - split seed: `42`
  - train seed: `43`
- Training:
  - `50` epochs
  - batch size `256`
  - optimizer `AdamW`
  - learning rate `1e-3`
  - weight decay `0.01`
  - no synthetic negatives
  - requested GPU `H100!`

## Variants

### Control

`PF07_control`

- current honest untied PF07 baseline
- mode: `shared_base_factorized_context_plus_segment_residual`

### Variant A

`PF07_dag_family`

Output DAG:

- `KD_base`
- `IC50_family_anchor = f(KD_base, shared_latents)`
- `EC50_family_anchor = g(KD_base, shared_latents)`
- `IC50 = IC50_family_anchor + leaf_delta`
- `KD_proxy_ic50 = IC50_family_anchor + proxy_leaf_delta`
- `EC50 = EC50_family_anchor + leaf_delta`
- `KD_proxy_ec50 = EC50_family_anchor + proxy_leaf_delta`

Interpretation:

- tests whether explicit family anchors help even before assay-condition leaves are introduced

### Variant B

`PF07_dag_method_leaf`

Output DAG:

- same family anchors as Variant A
- `IC50` and `EC50` also get output-side method-specific leaves
- assay labels may choose which leaf is supervised/evaluated
- assay labels must not alter the trunk or affinity input vector

Planned leaf structure:

- method categories with meaningful support on this contract:
  - `PURIFIED_COMPETITIVE_RADIOACTIVITY`
  - `PURIFIED_COMPETITIVE_FLUORESCENCE`
  - `CELLULAR_COMPETITIVE_FLUORESCENCE`
  - `PURIFIED_DIRECT_FLUORESCENCE`
  - `CELLULAR_DIRECT_FLUORESCENCE`
  - plus fallback `OTHER`

Interpretation:

- tests whether explicit assay-method leaves help once they are modeled on the output side only

### Variant C

`PF07_dag_prep_readout`

Output DAG:

- same family anchors as Variant A
- factorized output-side deltas by:
  - prep
  - readout
- geometry remains implicit through assay family support on this contract

Interpretation:

- tests whether a lower-parameter factorized assay-structure variant works better than full method leaves

## Implementation Notes

- Use the existing `affinity_assay_residual_mode` axis to select the new DAG variants.
- The main forward pass should continue producing all affinity outputs from shared sequence-derived latents.
- If method-specific leaves are present:
  - emit the full set of leaf outputs in the `assays` dict
  - select the matched output in training/eval using `batch.binding_context`
  - do not pass `binding_context` into `forward_affinity_only()` as a predictive feature
- Keep evaluation apples-to-apples:
  - the per-example held-out prediction for each row must come from the output leaf appropriate to that row's assay label

## Minimal Verification

- unit tests for:
  - DAG variants construct and emit the expected output keys
  - `forward_affinity_only()` remains invariant to `binding_context`
  - matched output routing chooses the expected leaf for each assay family / method context
- dry-run the experiment launcher locally
- run the sweep on Modal

## Success Criteria

- Any DAG variant beats the honest PF07 control on held-out test Spearman without a clear RMSE collapse

## Failure Criteria

- DAG variants lose to the PF07 control or only improve niche probe behavior while worsening aggregate held-out metrics

## Decision Rule

- Promote a DAG variant only if it clearly beats the PF07 untied control on the primary ranking/regression metrics
- If multiple DAG variants are close:
  - prefer the simpler one
  - prefer the one that preserves better RMSE and calibration
