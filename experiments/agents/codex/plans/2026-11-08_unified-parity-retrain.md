# Newer-code unified parity-data retrain

Date: 2026-11-08
Owner: Codex / GPT-5
Status: preregistered before launcher implementation or data access

## Objective

Measure whether the verified NetMHCpan 4.2 / NetMHCIIpan 4.3 BA TRAIN
projection closes the current affinity gap without sacrificing Presto's
presentation or operational-processing performance. Keep initialization,
architecture, optimizer, update budget, development populations, and selection
rule identical between the reference and data-only arms. The only scientific
difference is the added archive-native, unspecified-assay IC50 supervision.

This is a development experiment. It does not create a new TEST and cannot
support an independent superiority claim because the BA archives overlap known
training data and expose no study identities.

## Frozen lineage

- Code starts from successor branch commit
  `c3991699c0f3c3861e698c26c2941aef35a49840`; the executable bundle will pin a
  later clean implementation commit containing only this plan and launcher.
- Both fit arms initialize from the selected Stage A checkpoint, SHA-256
  `e6cc6fbb9950ace32efe418bfa96b119e1b94f0399e6843db0a16b0208937c06`,
  stored at
  `modal-volume://presto-checkpoints/2026-11-07_1915_codex_unified-presentation/stage-a/raw/selected.pt`.
- Do not initialize either comparison arm from selected Stage B. Restarting
  both from Stage A avoids confounding the parity arm with continuation from a
  checkpoint already selected on the exposed development studies.
- The historical Stage B checkpoint, SHA-256
  `a83ac0d80c16798abc33be813dbeb1a6bad42c25dc18c29edbda2716aa801333`,
  remains a scored external reference for code-drift and regression context.

## Frozen data

Both arms use the exact current Stage B sources:

| Source | Role | Rows | SHA-256 |
|---|---|---:|---|
| CARMEN presentation TRAIN | TRAIN | 1,349,378 | `6609ee375f6bb117f19035560daa6cf50a63b9c229fa46ae24e773afebdda2fc` |
| CARMEN presentation development | development | 48,530 | `0313b2d6b394320da7dcf6aa9144cbc44b73abe48b16380e03c1b27a9559fc52` |
| Hitlist v4 concentration TRAIN | TRAIN | 160,326 | `8593fbd416a489b46397c4fae288d5916ad55d2febec7f7798bc4bb3f788ec1c` |
| Hitlist v4 concentration development | development | 2,536 | `08e41a6c006274b5929142fc6bd4687a289a08df9cc5212edd92399f907f2f8a` |

The data-only arm additionally uses:

| Source | Role | Rows | SHA-256 |
|---|---|---:|---|
| NetMHCpan 4.2 BA projection | TRAIN only | 80,744 | `4491918763c1cf1c64825af0228357f43516293d1b6ea73ef8a6a7554d4635df` |
| NetMHCIIpan 4.3 BA projection | TRAIN only | 62,563 | `153a51f8997093c0b8cf724dc8ff649a676ee3ee2e516c52dff810bb18d00ee2` |

The projected BA rows retain `y=0` as `>=50000 nM`, `y=1` as `<=1 nM`,
and interior targets as exact `IC50_nM=50000**(1-y)`. Preparation, mechanism,
and detector are unknown, so these rows train only the all-unknown reference
IC50 route. They do not support any named protocol-specific assay track.

No T-cell row, locked confirmation row, TEST row, current outcome, or model
prediction participates in this first comparison.

## Preflight gate

Before either optimizer runs, execute the canonical `ObservationIndex` and
`training.modular_run.preflight` path on both complete manifests in a Modal CPU
worker with network blocked. Persist:

- source byte counts and independently recomputed SHA-256 values;
- partition, endpoint, MHC-class, qualifier, and assay-track counts;
- every input artifact location;
- resolved model/training configuration;
- admission report and peak RSS/runtime;
- proof that both manifests have zero TEST rows;
- proof that the reference and parity manifests differ only by the two declared
  BA sources and the resulting train census.

Any source drift, unsupported row, undeclared stratum, missing method-specific
track, checkpoint mismatch, or TEST presence stops the family before GPU use.

## Conditions

### R — newer-code reference reproduction

Stage A parent plus the four original Stage B sources.

### P — comparator-parity data only

The same parent, manifest, architecture, and optimization as R, plus the two
verified BA TRAIN files. No model mechanism or loss change is permitted.

Both conditions use the historical Stage B contract exactly:

- shared assay-aware model, `d_model=128`, two layers, four heads, `d_ff=512`;
- both MHC classes and existing normalized class-II 9-mer register
  marginalization;
- nine explicit physical affinity assay tracks;
- staged `elution_observation` downstream of reference IC50 affinity;
- 5,000 updates, batch 128, AdamW, learning rate `3e-4`, weight decay `1e-4`;
- 300-step warmup and cosine decay to 0.05 of the base learning rate;
- seed 42; checkpoint and validation every 500 updates;
- endpoint/MHC-class balancing and the exact historical Stage B stratum weights;
- EL study-mean development selection;
- `H100!` requested explicitly; record observed GPU and memory.

The parity arm's fixed IC50 I/II stratum weights remain 0.25/0.25. Thus added
rows change which unique examples are covered inside those strata, not the
between-class or EL-versus-affinity exposure contract.

## Evaluation

After both checkpoints are frozen, score the exact unchanged development rows.
Retain per-example prediction dumps for R, P, and the historical Stage B
reference. Join comparator predictions only by the frozen observation identity
and require complete matched-row equality before metrics.

Report at minimum:

- MHC-I and MHC-II exact IC50 Spearman, Pearson, RMSE in log10(nM), and row
  counts, pooled and equal-study;
- `<=500 nM` accuracy, balanced accuracy, precision, recall, F1, AUROC, and
  AUPRC by class and assay method cell;
- censor satisfaction and violation magnitude;
- presentation and operational-processing AUROC/AUPRC, pooled, equal-study,
  and per study;
- paired study/peptide-family bootstrap intervals for R versus P and each arm
  versus the already frozen matched comparator predictions;
- runtime, peak RSS, peak GPU allocation/reservation, checkpoint size, and
  parameter/update coverage;
- class-II register posterior entropy and core recovery where labels permit.

No threshold, calibration, architecture, or stopping decision may use a TEST.

## Frozen H100 execution implementation

Use the two exact manifests emitted by the passed primary preflight, with
SHA-256 `e2724c0ba873a98fa15f90c6ccc06013f320e65eb8968af0400bf7722e6db16d`
(reference) and `8259d1ec1b59430d7078174a7718d8b7157e351a0261e70384996692605bbd7c`
(parity). Their absolute scratch paths and complete scientific content remain
byte-identical to preflight. Before either H100 worker loads data, it must verify
the canonical comparison-closure receipt SHA-256
`b60d3f9343808b23fa67a6f636df0be01699f59475f7af596701c1ac00121c16`,
its manifest hash, the Stage A parent, every source byte count/hash, and the
clean source commit/runtime-file hashes.

Launch reference and parity from one frozen app with requested GPU `H100!`, at
most two containers, 4 CPU / 32 GiB / 512 GiB scratch per worker, blocked
external network, TF32/AMP/compile disabled, and the historical 10,800-second
segment bound. The workers may run concurrently, but each writes only to its
own immutable `fits/{condition}` directory. Checkpoint/resume state is committed
at checkpoint boundaries so a platform retry resumes the same frozen condition;
it must never silently restart with a changed manifest or seed.

Preserve selected/last/resume checkpoints, summary/history/config/coverage/
admission, gzipped per-example TRAIN and development predictions, the empty TEST
prediction contract, hardware evidence, resource measurements, and a complete
artifact receipt. Return only a compact receipt pointer from the network-blocked
worker. A condition is complete only at exactly 5,000 steps, passed coverage,
zero TEST rows, a zero-byte raw TEST prediction dump, and a locally hash-verified
receipt. Do not choose between conditions until both receipts and development
prediction dumps are independently collected.

## Preregistered decisions

1. Prefer P as the next affinity/presentation parent only if it improves MHC-II
   exact-IC50 Spearman by at least 0.05 over R, does not reduce MHC-I Spearman by
   more than 0.01, and does not reduce either presentation or processing
   equal-study AUROC by more than 0.01.
2. Regardless of the point estimate, describe any comparator result here as
   development-only. A superiority claim waits for genuinely new
   study-disjoint affinity data.
3. If P remains more than 0.05 Spearman below matched NetMHCIIpan, implement one
   separately frozen mechanism arm: candidate-core attention to alpha and beta
   grooves separately, neutral-start residual fusion, unchanged register
   enumeration/marginalization, and unchanged output assay semantics. Do not
   add another register-temperature or pooling-only ablation.
4. If P fails the MHC-I or presentation/processing regression guards, retain R
   while diagnosing per-source and per-assay cells; do not average the failure
   away in a composite score.

## Unified T-cell continuation

Only after the R/P decision, extend the selected checkpoint with the five fixed
assay-aware T-cell tracks from the frozen 17,176-row bundle, SHA-256
`0ef7bc5b8b6926750fb65261e17a2938f5d1b96539fc980f9227ada6c011d86e`.
Start with the proven binary-only output loss and preserve multimer binding as a
separate report-only track. The old development cohort may guide software and
regression checks but cannot establish promotion.

Before a final T-cell fit, acquire a newly disjoint whole-study cohort with
observed negatives, mixed-outcome studies, and a dedicated train-only or nested
calibration partition. Freeze track/study/outcome sampling and nondegenerate
threshold gates. Run PRIME 2.1 through `mhctools.PRIME` only after installing and
pinning a licensed MixMHCpred 3.x executable and only on the new class-I TEST.

## Closure

The family is complete only when the experiment directory contains frozen
launch manifests, source snapshots, call receipts, complete admission and
training summaries, per-example development predictions, comparison tables,
deterministic figure-generation commands, artifact hashes, and a canonical
`experiments/experiment_log.md` entry. Large data and checkpoints remain on the
Modal volume with byte counts and hashes; they are not committed wholesale.
