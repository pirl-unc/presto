# Panel-scale fix: first run where the model actually learns

**Date:** 2026-09-02 · **Agent:** claude (Opus 5) · **Commit:** `2688399` (clean)
**Hardware:** local CPU (no GPU) · **hitlist:** 1.55.2 · **Runtime:** ~75 min

## Why this run exists

To answer one question: after correcting the assay-panel loss, does the model
learn on real data at all? It had not been.

`AffinityPredictor.predict_assay_panel` returns a *KD offset* in normalized
log10 space. The panel loss regressed it against `bind_target` **raw** --
nanomolar, up to 50,000. `loss_binding_assay_panel` therefore sat near 142,900,
an order of magnitude above the total loss, and could not fall because no head
output reaches 50,000. It dominated every gradient.

On `presto train synthetic`, before and after:

| | before | after |
|---|---|---|
| val loss | 9924.72 -> 9923.51 over 10 epochs (**0.012%**) | 1.9511 -> 0.9156 over 40 (**53%**) |
| `loss_binding_assay_panel` | 142,888 -> 142,879 (frozen) | 2.82 -> 1.56 |

This run then confirms the same on real data.

## Contract

- **Dataset:** hitlist 1.55.2, MS + binding evidence, `map_source_proteins=True`
  (n_flank / c_flank at 100% coverage on both families), restricted to
  `HLA-A*02:01`. T-cell and TCR rows come from the IEDB/VDJdb path, which
  hitlist does not carry.
- **Caps:** binding 2500, elution 2500, t-cell 800, VDJdb 800, stability 400 ->
  **21,189 total training samples** after synthetic negatives and MHC
  augmentation. Caps are per-modality; without capping t-cell and VDJdb the
  epoch is 250k samples (they contribute 208k and 166k uncapped).
- **Synthetic negatives:** pmhc 1.000, processing 0.500, derived_elution 0.500,
  derived cascade 0.500. 10,000 binding + 4,000 elution negatives added.
- **Model:** d_model 64, 2 layers, 4 heads, expanded latent topology.
- **Split:** peptide-grouped, val_frac default.

## Held-out results

Best val loss **0.6708** at epoch 5; epochs 6-8 overfit (train 0.452 -> 0.342
while val rose 0.694 -> 0.706), which is expected at 21k samples.

**Binding regression -- genuinely good:**

| task | n | Spearman | Pearson | RMSE |
|---|---:|---:|---:|---:|
| binding | 1137 | **0.823** | 0.853 | 0.812 |
| binding_affinity_probe | 1137 | **0.828** | 0.857 | 0.759 |
| binding_kd | 70 | **0.878** | 0.815 | 0.956 |
| binding_ic50 | 236 | **0.774** | 0.745 | 1.077 |
| t_half | 73 | 0.736 | 0.418 | 1.845 |

**Elution / presentation -- the AUROC is not what it looks like:**

| task | n | AUROC | AUPRC |
|---|---:|---:|---:|
| elution | 961 | 0.996 | 0.996 |
| presentation | 961 | 0.981 | 0.983 |

`elution_real_only_n_negatives = 0` and `presentation_real_only_n_negatives
= 0`. **The corpus supplies no real negatives for these tasks**, so every
negative in that AUROC is a synthetic decoy and the number measures whether a
peptide looks real, not whether it is presented. The decoy-stratified metrics
exist to say this out loud rather than report 0.996 as a result.

**T-cell / immunogenicity -- at chance on real data:**

| task | n | AUROC | balanced acc |
|---|---:|---:|---:|
| immunogenicity_real_only | 112 | **0.512** | 0.500 |
| tcell_real_only | 112 | **0.519** | 0.536 |
| foreignness | 1140 | 0.738 | 0.668 |

Against decoys these same heads read 0.74-0.82 AUROC; against real negatives
they are indistinguishable from a coin. With 800 T-cell records capped and 112
real-only validation rows this is under-powered rather than proof of a defect,
but it is not evidence of skill either.

## Takeaway

The pipeline trains end to end on real data and **binding is genuinely
learned** (Spearman 0.82-0.88 held out). The panel-scale bug was blocking all
of it.

What this run does **not** show: presentation skill (no real negatives exist in
the corpus for it) or T-cell skill (chance on real negatives at this data
scale). Both need a larger run, which needs a GPU.

## Artifacts

- `val_metrics.csv` -- all 162 held-out metrics
- `summary.json`, `training_curve.json`
- `val_predictions_sample.csv` -- first 400 of 6,532 per-example rows; the
  full dump stays out of git per repo convention
- `reproduce/launch.sh`, `reproduce/launch.json`
