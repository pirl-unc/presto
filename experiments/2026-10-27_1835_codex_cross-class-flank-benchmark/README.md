# Shared candidate CPU reference and runtime profiles

Completed one CPU I-only reference and three training-only profiles. The remaining
nine CPU conditions were superseded by the user-authorized [matched H100 family](../2026-10-27_1930_codex_modal-cross-class/README.md).
Codex / GPT-6, October 27, 2026. No CPU/GPU runtime pooling or ten-CPU-fit claim.

## Contract and reproduction

The [prospective plan](reproduce/plan.md), [launch](reproduce/launch.json),
[bounded revision](reproduce/bounded-resource-revision.md), runtime/code pins,
source snapshots and resource receipts preserve the executed path. Shared-v4
commit0659b43cbd296f9d590a215b6f4358c2780439b1 with recorded experiment/task dirt;
full-v3 f24f7dfd6404c6eb450545daf72769269f55a5bf and three-aa-v2
9dd71348124faa49a4c6a60872385c86895d50b0 frozen controls. Python3.12.6,
Torch2.14.1, macOS arm64 CPU, four numerical threads/one interop, no Modal GPU.

Original purified human IC50 manifest SHA256
914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be.
Exact training21636 (I14375/II7261), validation2925 (I2051/II874), test2803
(I1845/II958). Source-only filtering preserves every original row/split/sequence.
Missing qualifier follows the original exact export policy, with provenance.
log10(nM) IC50 uses training-only standardization and squared residual loss;
held-out bounds use one-sided squared loss. No KD/MS/TCR labels, pretraining,
synthetic biological labels, identity loss, clipping, new split or target-class
calibration. Exact test metrics use1758 class-I and842 class-II observations.
These are previously inspected, shared-study development holdouts.

Seed42 I-only:1000updates, batch64, AdamW3e-4/weight decay1e-4, dropout.1,
min/max epochs100, patience100; select using source-I validation only. Selected
epoch5. Opposite-class evaluation expands routing while preserving every tensor;
it does not make target-class serving qualified. Source checkpoint stays unchanged.

## Results

| Evaluation | Validation Spearman | Test Spearman | Test RMSE log10(nM) |
|---|---:|---:|---:|
| MHC-I, in-class | .572624 | .503549 | 1.205548 |
| MHC-II, zero-shot | .346956 | .235506 | 1.105225 |

Source-I training-mean baseline II test RMSE1.253872. Positive transfer is modest,
not equivalent to II-trained accuracy. Fit elapsed1106.911seconds. All exact-value
Pearson/RMSE/Spearman, loss,500nM classification metrics and secondary strata are
in [independent metrics](results/metrics-independent.csv) and the per-condition
verification JSON. Predictions are retained in results/shared-I-s42, including
source and opposite-class validation/test, source-mean controls and training probes.

| Training-only profile | Parameters | Median seconds/update |
|---|---:|---:|
| three-aa-v2 joint |455041|2.896772|
| full-context-v3 joint |482689|3.410|
| shared-v4 joint |449153|2.013634|

Profiles use three warmup and five timed training batches, not predictive fits.
Raw CPU inputs/checkpoints: ../../artifacts/2026-10-27_1835_codex_cross-class-flank-benchmark/.
All raw paths/hashes and actual resource peaks remain in reproduce/receipts.

## Admission history and closure

Initial6GiB startup admission refused before work during software tests. Admission2
passed after tests ended. A subsequent8GiB/60-second waiter was interrupted after
approximately ten minutes; no updates ran during waiting. The prospective bounded
revision changed startup to5.5GiB, running reserve3GiB, RSS cap2GiB, one-hour fit,
without numerical changes; all three profiles and the I-only fit completed.
The user then explicitly reauthorized Modal and rejected arbitrary local delays.
Remaining workers exited75 before fitting; the original parent records failure
(exit1) because of this deliberate migration. That raw receipt is retained, with
its interpretation in reproduce/modal-steering.json. No fit is silently retried.

Sequence-only audit finds zero exact training peptides shared between classes,
22 shared9mer substrings, and85.0% of II test rows sharing some9mer with II
training (I27.2%). These substrings are not assigned binding cores. Cross-class
shares are1/958 I-to-II and8/1845 II-to-I test rows. Exact-peptide disjointness
therefore does not establish new-core generalization.

Software:272 common-model/trainer tests, four harness fixtures and installed-wheel
smoke passed. Saved CPU predictions independently reproduce all reported metrics.
No biological qualification or PR131 promotion follows. Preferred runtime for the
requested matched comparison is the authorized H100 family; this CPU row remains
an explicitly separate reference.
