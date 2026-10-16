# Corrected human IC50 fitting diagnostic

Status: prospective recipe, not launched. Agent/model: Codex (GPT-6).

This bounded experiment follows the [pre-counts plan](../agents/codex/plans/corrected-human-fit.md)
and [implementation spec](../../tasks/corrected-human-fit-spec.md). It uses the
[public corrected preflight](https://github.com/pirl-unc/presto/releases/tag/corrected-human-preflight-v1),
released Presto 0.1.13 / Hitlist 1.78.0 and its exact 9,053 selected observations
(7,241/906/906). The unchanged prospective manifest gates the directly measured
IC50 output before model construction. No other output is claimed qualified.

`reproduce/frozen-inputs.json` pins the public manifest, membership and human-readable
source contexts by URL, SHA-256 and size. `reproduce/producer.json` pins the actual
released producer separately from this recipe's repository version. Datacache
stores native inputs, pinned MHC sources/catalog and complete result generations.

Run `reproduce/launch.sh` from a clean, committed repository with the ordinary
Modal/datacache environment. The runner requests eight CPUs, 64 GiB memory,
512 GiB scratch and **no GPU**, with one Torch/BLAS thread and zero loader workers.
The work alarm is 90 minutes and function timeout 95 minutes.

Use a fresh expanded d32/l1/h4 full-profile model, batch 64, shuffled sampling,
AdamW learning rate 0.001 / weight decay 0.01, canonical uncertainty weighting,
two epochs capped at 100 batches each. All augmentation, synthetic ratios and
explicit regularization weights are zero. Complete validation aggregate loss
selects the checkpoint; complete validation/test predictions reload that checkpoint
in a fresh model. IC50 exact values use log10(nM); original operators remain
explicit, with qualifier -1 for left-censored upper bounds and +1 for right-censored
lower bounds. All measured families contribute canonical losses; output claims
remain limited to direct IC50. No pretraining or prior checkpoint is used.

The same 2,074 training IC50 observations are evaluated at initialization and with
the selected checkpoint, preserving training RNG state. Fixed training-only exact
mean and alpha-1 ridge baselines use peptide length/composition and MHC class.
The source-linked train/validation/test tables preserve IDs, raw values, censoring,
original and resolved MHC, citations, cell and condition context. Undefined metrics
are explicit nulls. RMSE uncertainty uses 1,000 paired peptide-cluster bootstrap
resamples, seed 4242. These intervals do not resolve unknown biological identity.

Prospective fitting/quality/selection rules are fixed in the linked plan; failure
will be published without changing the manifest, thresholds or this family's
budget. Test is a final descriptive evaluation, never a selection criterion.
Unknown donor/specimen/acquisition lineage and shared studies prevent an independent
biological generalization claim. Artifact links, hashes, all metrics and plots
will be added after execution, followed by review, merge and release.
