# Training-only temporal signal before kinetic fitting

Freeze before inspecting response variation. Consume the source-linked shared
contiguous recipe qualified by1123, including reviewed2.0.8 source compatibility.
No new data curation, source edits, split changes, fitting or held-out metrics.
Use training curves only. All conditions of each source construct stay grouped.

For each peptide/condition, use the common positive times across its admitted
source biological labels. Reference is the earliest of these physical times.
For every later common time, response is log10(signal(t)/signal(reference)).
Compare the response in each biological label against the mean response of the
other labels for the same peptide/condition. No-change predicts zero. This
uses a measured baseline and a matched biological measurement; it is an
empirical repeatability check, not zero-shot sequence prediction.

Keep every per-example response, cross-label prediction, zero prediction,
original signal, source curve IDs, reference time and target time. Average
squared errors first over times, then biological labels/peptides within a
condition, then conditions within a construct, then equally over constructs.
Also report pooled RMSE/Pearson/Spearman and target/prediction SD. Preserve
per-construct results to expose heterogeneity and the small parent count.
No significance or biological independence claim from correlated fragments.

Descriptive continuation gate: mean parent-weighted cross-label squared error
must be below the no-change value, with improvement in at least half of the
admitted training constructs. A failure changes the observation-model plan;
it does not authorize reshuffling holdouts or dropping difficult constructs.
Report source positive-selection, forced-zero exclusion and shared source
background processing as limits on what repeatability means. No nM/binder
metrics, no training/pretraining/loss optimization or synthetic labels.

After passing, freeze a small ordered sequence-to-bond-rate function and an
exact expected-fragment temporal-ratio fit with constant-rate/no-change controls.
Before any sampled fit, quantify rare-event variance and nonlinear plug-in bias;
retain exact analytic loss gradients as reference and label hybrids explicitly.
