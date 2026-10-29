# Shared-model retraining after source curation

Status: two H100! fits submitted; source preparation passed. No new predictive result yet.
Codex / GPT-6, 2026-10-29. Source commit `858f5383` plus pinned dirty source.

[Prospective plan and pre-fit amendment](../agents/codex/plans/2026-10-29_curated-data-retrain.md)
· [live comparison](http://127.0.0.1:54503/dashboard.html)
· [Modal fits](https://modal.com/apps/iskandr/main/ap-L2Gu45s9NY8QIyGFBycFZV).

## Frozen data revision

Compared with the completed temporal parent, apply 506 source-scoped inferred
bounds (482 ceilings / 24 floors), withhold 618 unresolved physical TRAIN rows,
add 13,549 source-verified unmodified human MHC-II ligand observations and 13,549
length-matched generated comparisons, and remove 35 old generated TRAIN rows
matching newly verified positive sequences. The expanded positive exclusion pool
adds 292 sequences, including reviewed positives outside architectural admission.

All validation/test files remain byte-identical. One additional recovered positive
was excluded because its sequence occurs in a held-out generated comparison.
Three generated TEST rows matching that sequence are flagged as suspect. Main
comparisons retain them; an additional analysis excludes their IDs for both model
generations. Source ligand evidence is not proof of binding each comparator's MHC.
These previously inspected holdouts support repeated development comparisons,
not a fresh blind evaluation. Test predictions never select the checkpoint.

The initial preparation stopped on the blacklist assertion before any optimizer
update. Its exact source, call and log are preserved in
`attempts/blacklist-assumption/`. An independent audit established affected IDs
and hashes before the revised `source-review-v2` preparation was launched.

## Architecture, loss and training

One canonical observation-manifest trainer and shared-v5 wide model: width 192,
three layers, six attention heads, FF 768, dropout .1. No pretraining, identity
heads, new ranking loss or architecture change. Distinct measured outputs are
IC50, KD, EC50, KD(~IC50), KD(~EC50), half-life, koff, td50, qualitative binding
and elution. Unsupported kon remains absent. A quantity label is not assay-method
verification. TRAIN-only exact-value scalers, unclipped regression, one-sided
inequality residuals and binary cross-entropy are unchanged.

Two fresh seeds 42/43, batch 128, AdamW learning rate 3e-4, weight decay 1e-4,
constant schedule. Sampling: 60% elution, 20% literal IC50, 15% other physical,
5% qualitative; square-root endpoint/class allocation within each family.
Recompute canonical quotas to expose every TRAIN row at least once. This gives
**3,519,577 TRAIN observations**, **6,588,275 weighted examples** and
**51,471 updates per seed** versus the parent's 51,099. It is the same complete-
epoch policy, not exactly matched compute. Checkpoint every 500 updates, validation
every 1,000, select minimum equal-endpoint validation loss, seed breaks ties.

Synthetic comparisons use the original sampler seed 20261028 and pinned Ensembl
112 human proteome. New comparisons: 13,546 same-protein / three reference-proteome.
They are unobserved peptides, not measured nonbinders; original evaluation ratios
stay fixed. Source-reported MS nondetections are evaluated separately.

## Verification, resources and reproduction

172 focused source/admission and checkpoint/reporting/prediction checks pass.
Independent source replay verifies original CSV and PSM chemistry/row evidence;
preparation verifies exact changed/removed row IDs, raw fields, source hashes,
held-out exclusion, numerical values, scalers and complete quotas. A second full
cohort census independently passed; each GPU worker must pass the canonical preflight
before creating its optimizer. CPU preparation: eight CPUs / 64 GiB requested,
246.44 seconds. Requested fit hardware: at most two H100!, eight CPUs / 64 GiB
host RAM and 512 GiB scratch each. Both workers report NVIDIA H100 80GB HBM3, 85,009,367,040 device bytes;
see `results/initial-hardware.json`. Runtime/peak memory remain pending.

Model and trainer source are byte-identical to the parent runtime. Only the new
measurement-curation module, common Hitlist admission and already validated faster
common input reader differ. All 214 source files, dependency versions, proteome,
prepared record and manifests are hash-pinned. No active input mutation is allowed.

`reproduce/launch.json`, `training-contract.json`, `runtime-pins.json`, manifests,
call IDs and `reproduce/source/` preserve exact invocations and executed source.
Preparation receipt: `results/prepared.json`; full source delta and blacklist
receipts are in `results/`. Durable prepared data and run artifacts are on Modal
volume `presto-checkpoints` at this family / `revisions/source-review-v2/`.
Local upload runtime is under `artifacts/2026-10-29_1956_codex_curated-data-retrain/`.
Large collected per-example predictions/checkpoints will remain there, referenced
by hashed collection receipts in the canonical experiment directory.

The running continuation (`code/advance.py`) collects complete receipts, independently
recomputes held-out target/loss and per-output/class/era metrics, writes the full
report, and verifies matched observations before paired parent comparisons.
`analysis/compare_parent.py` also reports sensitivity to the three suspect test
labels. The live report shows parent selected checkpoints against latest new
validation checkpoints, explicitly different training stages. No improvement is
claimed until measured; final scientific writeup and promotion remain pending.
