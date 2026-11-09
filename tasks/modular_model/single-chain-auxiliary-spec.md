# Genuine single-chain annotation admission

Prospective next software/data slice, 2026-10-21. The full goal and frozen issue
remain authoritative. The pinned MHC catalog contains individual class-II
chains; requiring a fabricated partner to supervise per-chain species is a data
contract error. TCR species annotations can likewise exist without a paired
receptor. This slice changes auxiliary input admission, not the assay model or
its requirement for a full resolved molecular pair.

## Minimal input and model change

1. Reuse existing groove and TCR sequence slots, with absent chains represented
   by empty optional fields only on `endpoint=identity` observations. Keep direct
   `MHCInput.validate()` and all assay observation validation strict by default.
   Add an explicit auxiliary-only partial validation mode; at least one actual
   sequence remains required. Do not introduce padding proteins, guessed chains,
   inferred pair labels or new species vocabularies in this software slice.
2. Encode only present sequences. Reuse the existing shared species head on each
   available chain vector. Do not allocate a new encoder or species readout.
   A per-chain target requires that exact slot: `mhc1` needs groove1, `mhc2`
   groove2, `tcr_a` TCR alpha and `tcr_b` TCR beta. Targets for absent chains fail
   before optimization instead of becoming missing loss silently.
3. Class and pairing heads still consume their complete molecular pairs. TCR
   lineage uses the paired receptor representation. Single-chain reference class
   annotations remain provenance until an explicit separate supervision contract
   is adopted; do not claim this slice trains the existing paired MHC-class head
   from one chain. Actual class/pairing cohorts remain a separate preparation task.
4. Auxiliary-only prediction exports retain only available per-chain outcomes;
   partial inputs cannot invoke class routing or pMHC/receptor compatibility.
   Existing full-pair class/species/default/override and assay behavior is
   unchanged. Missing fields do not produce gradients for fictitious partners.
5. Propagate the same admitted semantics through preflight, checkpoint subset
   transfer, retained/frozen dependency validation and prediction support checks.
   Parameter shapes remain unchanged; do not migrate an old head to a new shape
   or treat incidental encoder exposure as directly supervised output support.
6. Across auxiliary train/validation/test partitions, reject a repeated nonempty
   MHC groove or TCR chain even when the partner or slot changes. Full/partial
   views may coexist within one split. This is an exact-sequence exposure check;
   homology grouping and annotation-to-assay exposure require separate audits.

## Verification and follow-on data preparation

- Regressions for alpha-only, beta-only and mixed full/partial auxiliary batches,
  MHC and TCR species, including successful gradient/update closure and reload.
- Reject labels for absent slots, class/pairing/lineage targets without the
  required pair, and every affinity/TCR assay missing its required chain.
- Preserve current complete-pair predictions for identical weights/inputs and
  assert no fabricated sequence enters the encoder batch.
- Exercise the normalized observation index and canonical trainer, rather than
  only a direct forward call. Preserve explicit lineage/grouping for real records.
- Before real pretraining: independently curate catalog quality/taxa, identical-
  sequence aliases and conflicting labels; freeze vocabulary and sequence/group
  splits with dispositions. The raw 56,440-row inventory is not this cohort.
  Register any actual fit with its own baselines, calibration and transfer gates.

The just-completed peptide-source screen failed to establish improvement over
composition. This admission change is needed for truthful use of available
annotations; it is not justified as a proven remedy for affinity underfitting.
