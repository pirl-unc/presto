# Early supervision and overridable categorical context

This implements the active goal's additional pretraining requirements within the
same encoder and observation/loss path. Existing class/species/lineage tasks stay.

- Add `peptide_source`, `mhc_pairing` and `tcr_pairing` to declared auxiliary tasks.
  Source categories are a frozen vocabulary from the actual Weirdo data. One
  linear readout produces independent membership logits. A label is `present` or
  `absent` for a named category; missing category labels are masked, never absent.
  This does not assert a unique species of origin. Pairing readouts consume the
  two relevant sequence vectors and predict `matched`/`mismatched` according to
  the declared source evidence. Pairing is not pMHC–TCR recognition.
- Keep one identity-observation format: `identity_labels` names categorical
  outcomes. Source keys are `peptide_source:<category>`. A two-logit `[0, logit]`
  view gives each independent membership the existing CE/evaluation contract
  (exactly binary cross entropy), without introducing a second trainer.
- Permit peptide-only identity observations. Source membership, MHC metadata,
  receptor origin and pairing all train the shared encoder. Direct assay
  gradients remain stopped at categorical probabilities, not at rich vectors.
- Add independent `auxiliary_values` and `auxiliary_overrides` to context, never
  as training labels. Preserve predicted/supplied/effective values and origin via
  the existing deterministic resolver. Actual assay-cell coreceptor stays separate
  from the TCR's lineage prior.
- Declare optional downstream categorical slots explicitly in the config:
  `mhc_pairing` can condition pMHC fusion; `tcr_pairing` conditions receptor paths;
  `tcr_lineage` and source memberships condition cellular paths. MS response gets
  none of these metadata slots. Unconfigured or unobserved effects are not silently
  activated. Source and lineage overrides never change purified binding or the
  chemical compatibility of fixed supplied TCR–pMHC sequences.
- Validate task input requirements, categorical vocabularies and training-label
  coverage in preflight. Record each source category's support separately from
  aggregate head support. All new fields must survive checkpoint round trips,
  feature support guards, evaluation and the public prediction adapter.
- Adapters must attest reference coverage and split families, pairing provenance
  and observed versus reconstructed chains. Generated mismatches are only a
  declared representation objective; they are not measured negative compatibility.

Verification: peptide-only/mixed auxiliary losses, independent multilabel outputs,
input gradients and allocation, override invariance and endpoint masks, detached
classifier probabilities, paired input validation, held-out label coverage,
checkpoint/inference round trips and existing modular regression. Real pretraining
is registered separately after actual source admission.
