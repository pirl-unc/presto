# Common preparation throughput

Status: runtime comparison complete; common implementation retained.
Codex / GPT-6; 2026-10-30. Baseline commit ed9b1b9b18853df171528851c96b39bac1294b0b;
candidate is the dirty five-file implementation frozen in `reproduce/launch.json`.

Compare frozen ed9b1b9b runtime with compact admitted metadata, fused scaler and
capability preflight, and once-per-model dependency resolution. No model, loss,
batch quota, raw observation, source-integrity or scientific-evaluation change.
Existing direct-factorized fits keep their original executable snapshot.

CPU runtime study only: no pretraining, optimization, predictive evaluation or
new synthetic generation. Prediction dumps and validation/test metrics are not
applicable. Read existing measured and generated comparison rows solely to time
preparation; labels/provenance remain byte-identical between implementations.

Use deterministic first-5,000 TRAIN rows of each physical-assay, native-MS and
generated-ligand-background source. These prefixes are a runtime fixture, not a
representative accuracy cohort. Derive the endpoint/class/method registry from
the prefix, retain the original model dimensions and restrict original sampling
weights to present strata. Both variants use the same frozen manifest and bytes.
The manifest explicitly selects no-test diagnostics because no fitting occurs.

Measure admission, preflight, trainable-row filtering and weighted epoch ordering
in fresh processes, three alternating paired repeats. Preserve stage wall times,
normalization counts, memory, platform and exact result digests. Compare scalers,
admission reports, partitions, trainable indices, order and RNG state exactly.
Warm-cache timings include source hashes and every admission check. They do not
establish cold-cache/full-cohort/first-optimizer-update speedup.

Reproduction and source hashes: `reproduce/launch.json`, `reproduce/launch.sh`,
`reproduce/source/`; raw snapshots/inputs under the artifact root named there.
Modal variant requests eight CPU cores and 64 GiB RAM, **no GPU**; actual hardware
is recorded by the worker. Keep the existing maximum two H100! fit allocation.

## Results

Modal app: https://modal.com/apps/iskandr/main/ap-xhgeoVjM1r9gc7mQ9gORqL.
All three paired repeats completed. Exact current-cohort prefixes total **15,000
rows / 116,307,079 bytes**, including 5,000 existing generated-background rows.
They cover IC50, two KD-derived proxies, qualitative binding, half-life and ligand
observations. Endpoint-to-output mapping and descriptor vectors follow shared-v7;
all original reported qualifiers and source-scoped inferred bounds are preserved.
No optimization loss or predictive criterion is measured on these source rows.

| Phase | Baseline median seconds | Candidate median seconds | Speedup |
|---|---:|---:|---:|
| Hashing and complete admission | 3.402 | 3.481 | 0.98× |
| Sampling/capability/scaler preflight | 11.803 | 4.261 | 2.77× |
| Trainable-row filtering | 10.043 | 0.135 | 74.62× |
| Weighted epoch ordering | 4.326 | 0.186 | 23.19× |
| Combined measured preparation | **29.813** | **7.986** | **3.73×** |

Medians are computed separately for each phase and total; phase medians need not
sum to the median total. Raw timings and normalized-row counts are in
`results/timings.csv`; independent extraction/parity checks are in
`results/verification.json`. Full decoding/normalization falls from **90,000 to
30,000 rows** across the four phases. Initial admission retains all checks and
incurs a small metadata cost; most gains come from eliminating subsequent work.
The metadata array uses **60,000 bytes**, with ten interned categories in this
slice. Per-phase RSS is retained in raw results. gVisor reports identical ~3.32 GB
high-water marks for all processes; that does not establish exact peak-memory
equivalence. CPU model is masked as `unknown`; Python 3.12.1 / torch 2.14.1+cu130,
Linux gVisor, CPU-only requested allocation of 8 cores / 65,536 MiB. Host-reported
24 CPUs and 425.7 GB memory describe the visible host, not the requested allocation.

All six executions have identical scalers (including exact floating-point moments),
admission reports, partitions, trainable indices, complete weighted epoch orders
and final RNG states. Every metadata access retains the original per-row
inode/size/mtime check. Full row admission, hashes, lineage, split isolation,
capability checks and actual batch loading remain enabled.

The independent software trajectory experiment runs **eight updates** on fourteen
artificial rows: ten IC50/KD rows spanning both MHC classes and four identity rows,
with weighted sampling, identity replay, dropout and AdamW at the fixture defaults.
The old/new initial, last and selected model tensors, optimizer state, element
coverage and validation/test metrics are exactly identical. These are software
checks, not biological performance estimates. Raw checkpoints, reports and cached
per-example evaluations are in the local `software-baseline/` and
`software-candidate/` artifact directories listed in `results/software-parity.json`.
The exact fixture/command is frozen under `reproduce/source/software_parity.py`
and `reproduce/software-parity.sh`.

## Decision and limits

Use compact metadata and the consolidated scan in the canonical manifest trainer.
Do not add another training path or an unverified persistent cache. The current
two direct-factorized H100 fits already completed their expensive preparation and
continue on their frozen original runtime. They were neither restarted nor
hotpatched; this improvement applies to subsequent launches and epoch preparation
under the new code. Strict source fingerprints continue to reject cross-version
resume unless a new continuation is explicitly declared.

The measured total excludes source staging, model initialization, checkpoint
serialization, GPU work and actual first-batch latency. This is a bounded warm-cache
runtime comparison, **not a measured 3.73× full-30-GB startup speedup**. No larger
corpus or model-quality claim is made. Raw input prefixes and run artifacts are
durable at `presto-checkpoints:/2026-10-30_1725_codex_preparation-throughput/`;
local frozen runtimes are under
`/Users/iskander/code/presto/artifacts/2026-10-30_1725_codex_preparation-throughput/`.
If those runtime directories are absent, `python reproduce/restore_snapshots.py`
reconstructs all 219 files per variant from the pinned Git revision and the five
frozen overlays, checking every recorded hash and refusing to replace different
existing files. Run it from the experiment directory before `reproduce/launch.sh`.
The full-source SHA entries in `results/inputs.json` are inherited declarations
from the already verified active cohort; this bounded benchmark hashes its exact
prefixes, rather than re-reading all 30 GB solely to recalculate those hashes.

## Software verification

**2,929 tests passed, four skipped**, in 389.71 seconds on the unchanged final
implementation; lint and strict MkDocs build also pass. The first complete run
had 2,923 passes and six Hitlist fixture setup errors because free scratch space
fell to 0.341 GiB, below its 1 GiB reserve. After preserving the log and removing
only that run's disposable pytest directory, the clean full rerun passed. The
reserve was not lowered and no test/source fix was needed. Details and copied
logs are in `results/test-environment.json` and `results/full-suite-*.txt`.
The separate eight-update check also confirms byte-identical TRAIN/validation/test
prediction dumps (`results/software-prediction-parity.json`). Its fixture uses
AdamW 3e-4 / weight decay 1e-4, identity-loss weight 0.1, batch size 2, identity
replay batch size 3, IC50-I:KD-II sampling weight 2:1, and dropout 0.3.
