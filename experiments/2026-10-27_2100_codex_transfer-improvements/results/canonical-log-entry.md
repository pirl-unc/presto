
## 2026-10-27_2100_codex_transfer-improvements — shared residue interactions improve fixed-budget joint affinity

October27,2026; Codex / GPT-6. [Complete report, frozen launch/source, metrics and predictions](2026-10-27_2100_codex_transfer-improvements/README.md).
Clean launch653bbfb96965551b49e53a6a854bd438952b00cc, base4615573.
Reproducibility bundle: experiment reproduce/launch.sh/.json/source; raw artifacts
artifacts/2026-10-27_2100_codex_transfer-improvements, Modalvolume presto-checkpoints.
RequestedH100!, actualNVIDIAH10080GBHBM3/85009367040bytes; max2workers,4CPU,16GiBhost,
threads4/interop1. Python3.12.1,Torch2.14.1+cu130,NumPy2.5.4; noAMP/compile/TF32.
Analysis thread/env overrides and every dependency pin are frozen in reproduce.

Purified human direct IEDB IC50, manifest914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be;
train21636exact(I14375/II7261),val2925,test2803; exactval1968I/758II,test1758I/842II.
Original peptide/measurement-component partitions, shared studies, previously
inspected development holdouts. Verified mhcseqs paired grooves; no guessed partner.
Missing exported qualifier exact policy retained; bounds retained in holdouts.
NoKD/MS/TCR/processing proxies, pretraining, synthetic biology labels or identity loss.
IC50→unclipped log10(nM), train-only standardization, squared lossweight1, one-sided
heldout censor loss. AdamW3e-4,wd1e-4,batch64,dropout.1,min/maxepochs100,patience100.
Joint2000/source-only1000updates; seeds42/43, validation-only checkpoint selection.

Four joint conditions cross pooled-v4 versus common residue-interaction-v5 with
natural versus balanced class sampling; two v5 source-only conditions evaluate
unchanged source weights on the opposite class. Same core/full-context token encoder,
candidate endpoint scorer and normalized pooling; no class-specific heads. Balancing
changes exposure/epoch length and checkpoint cadence; attention adds normalization
and parameters, so mechanisms are not isolated.

| Condition | I val ρ / RMSE | I test ρ / RMSE | II val ρ / RMSE | II test ρ / RMSE |
|---|---:|---:|---:|---:|
| pooled-natural-joint | 0.5859 / 1.1362 | 0.5070 / 1.2030 | 0.5222 / 0.8657 | 0.4893 / 0.8092 |
| pooled-balanced-joint | 0.5733 / 1.1614 | 0.4998 / 1.2282 | 0.5813 / 0.8295 | 0.5524 / 0.7778 |
| interaction-natural-joint | 0.6115 / 1.1122 | 0.5248 / 1.1899 | 0.5832 / 0.8288 | 0.5039 / 0.8059 |
| interaction-balanced-joint | 0.5908 / 1.1475 | 0.5196 / 1.2067 | 0.6045 / 0.8157 | 0.5591 / 0.7823 |
| interaction-I | 0.5896 / 1.1357 | 0.5068 / 1.2112 | 0.2260 / 1.2226 † | 0.2010 / 1.2717 † |
| interaction-II | 0.2388 / 1.6104 † | 0.1702 / 1.6822 † | 0.6496 / 0.7793 | 0.6037 / 0.7352 |


Preferred **interaction-natural-joint**; sealed validation gates and source-only guard
pass. Natural sampling remains default; balancing alone fails allowed class-I drop.
The ordering clarification after partial validation is disclosed in the report;
new tests did not select the winner. MHC-I test ρ delta +0.0178, 95% peptide-bootstrap interval [+0.0026, +0.0329]. MHC-II test ρ delta +0.0146, 95% peptide-bootstrap interval [-0.0108, +0.0381].
I→II test ρ 0.1871→0.2010 (delta +0.0139, interval [-0.0311, +0.0628]). II→I test ρ 0.1652→0.1702 (delta +0.0050, interval [-0.0166, +0.0272]).

| Condition | Fit seconds | Whole worker seconds | Parameters |
|---|---:|---:|---:|
| pooled-natural-joint | 377.0 / 383.3 | 442.5 / 448.1 | 449153 |
| pooled-balanced-joint | 383.5 / 395.8 | 447.6 / 457.0 | 449153 |
| interaction-natural-joint | 573.8 / 554.3 | 637.1 / 615.9 | 515457 |
| interaction-balanced-joint | 556.7 / 577.2 | 620.1 / 643.7 | 515457 |
| interaction-I | 258.3 / 237.0 | 320.0 / 306.0 | 515457 |
| interaction-II | 311.5 / 309.7 | 348.0 / 345.9 | 515457 |


Fit/workerseconds are seed42/43, excludingqueue/build/upload/archive/download.
Memory, all regression/500nM metrics, exposures, per-example val/test predictions,
subgroups and bootstrap comparisons are retained. Two-seed means are not ensembles;
peptide-bootstrap intervals are not training-seed population uncertainty. More
exposure helps the old joint model but does not isolate the residue-interaction gain.
Historical converged models have different budgets; no independent qualification.

All12independent source/loss/metric checks and768fresh-H100training probes pass;
279preclosure software checks,6analysis fixtures and final integration/wheel/docs
evidence retained. Apply winning v5 to the single common training path; archive
v4 for inference. No new labels admitted from exploratory nested-peptide audit.
Takeaway: a useful fixed-budget joint-affinity architecture gain, without a strong
zero-shot-transfer claim. Scientific PR131 and same-protein MS/TCR gates remain open.
