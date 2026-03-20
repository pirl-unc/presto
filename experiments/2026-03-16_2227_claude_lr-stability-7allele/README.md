# LR/Schedule Stability Sweep (d=128 full loss, 7-allele, 50ep)

- Agent: `claude`
- Source script: `scripts/benchmark_lr_stability_7allele.py`
- Created: `2026-03-16T22:27`
- Git commit: `bf7fcbf`
- GPU: H100!
- Prior experiment: `2026-03-16_1454_claude_factorized-ablation-7allele`

## Question

Can we stabilize d=128 + `full` loss (which diverged at lr=1e-3) by lowering lr or adding warmup? And once stable, does `full` loss beat `assay_heads_only`?

## Dataset Contract

- 7 alleles: HLA-A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02
- `numeric_no_qualitative`, qualifier_filter=all, split: peptide_group 80/10/10 seed 42
- train=32,805, val=4,184, test=4,060

## Training

- d_model=128, n_layers=2, n_heads=4, batch_size=256, 50 epochs
- All pretrained from mhc-pretrain-20260308b, mhcflurry target encoding
- split_kd_proxy, max_affinity_nM=100,000, no synthetics

## Results

### Stability and Loss

| ID | Description | Diverged | Best Epoch | Best Val Loss | Probe Rank Corr |
|----|-------------|----------|------------|---------------|-----------------|
| **S1** | **A07 full lr=3e-4 warmup_cosine** | **No** | **17** | **0.0306** | **0.8839** |
| S2 | A07 full lr=3e-4 onecycle | No | 10 | 0.0320 | 0.8393 |
| S3 | A07 full lr=1e-4 warmup_cosine | No | 20 | 0.0330 | 0.8929 |
| S4 | A07 full lr=1e-3 warmup_cosine | Yes (ep14) | 5 | 0.0376 | 0.1518 |
| **S5** | **A07 heads_only lr=1e-3 constant [C4]** | **No** | **40** | **0.0294** | 0.5536 |
| S6 | A03 full lr=3e-4 warmup_cosine | No | 17 | 0.0310 | 0.8393 |

### Probe Discrimination (SLLQHLIGL at best val_loss epoch)

| ID | A\*02:01 IC50 (nM) | A\*02:01 Kd (nM) | A\*02:01 Bind | A\*24:02 Bind | Discrimination |
|----|-------|------|-----------|-----------|----------------|
| S1 | 32.4 | 26.9 | 0.9676 | 0.0084 | 115.5x |
| S2 | 40.2 | 28.3 | 0.9580 | 0.0359 | 26.7x |
| S3 | 18.3 | 10.9 | 0.9838 | 0.0065 | 151.1x |
| S4 | 17.7 | 15.8 | 0.9844 | 0.0367 | 26.8x |
| **S5** | **42.3** | **5.4** | **0.9555** | **0.0027** | **358.7x** |
| S6 | 19.6 | 10.9 | 0.9823 | 0.0443 | 22.2x |

## Pairwise Comparisons

### Does lr=3e-4 + warmup_cosine stabilize? (S1 vs prior C1)

**Yes.** S1 completed all 50 epochs without divergence (prior C1 diverged at epoch 49 with lr=1e-3 constant). Val loss improved from 0.0392 to 0.0306.

### warmup_cosine vs onecycle? (S1 vs S2)

**warmup_cosine wins.** S1 val_loss=0.0306 vs S2 val_loss=0.0320. S1 also has much better discrimination (115.5x vs 26.7x).

### Is lr=1e-4 too conservative? (S3 vs S1)

**Slightly.** S3 val_loss=0.0330 is worse than S1's 0.0306, but S3 has better probe rank corr (0.8929 vs 0.8839) and better discrimination (151.1x vs 115.5x). Trade-off: S3 learns slower but may generalize better.

### Does schedule alone fix lr=1e-3? (S4 vs prior C1)

**No.** S4 diverged at epoch 14 even with warmup_cosine. lr=1e-3 is too high for `full` loss at d=128 regardless of schedule.

### full vs assay_heads_only once stable? (S1 vs S5)

**S5 (heads_only) still wins on val_loss** (0.0294 vs 0.0306) and discrimination (358.7x vs 115.5x). But S1 has much better probe rank corr (0.8839 vs 0.5536), suggesting better multi-head coherence. S1's IC50/Kd predictions are biologically reasonable (32.4 / 26.9 nM for a ~15 nM binder).

### A07 vs A03 once stable? (S1 vs S6)

**A07 slightly wins.** Val_loss 0.0306 vs 0.0310, discrimination 115.5x vs 22.2x. Once stability is controlled for, factorized context provides modest benefit.

## Key Findings

1. **lr=3e-4 + warmup_cosine stabilizes d=128 + full loss.** This is the primary result.

2. **lr=1e-3 is too high for `full` loss at d=128** — even warmup_cosine can't save it (S4 diverged ep14).

3. **S5 (assay_heads_only) has lowest val_loss (0.0294) and highest discrimination (358.7x)** but poor probe rank correlation (0.5536), meaning its different assay heads are less coherent.

4. **S1 (full loss, lr=3e-4, warmup_cosine) is the best balanced condition** — stable, good val_loss, excellent probe rank corr, reasonable probe predictions.

5. **S3 (lr=1e-4) is a viable conservative alternative** with slightly better discrimination but slower learning.

6. **Next steps:** S1 or S3 as the 7-allele d=128 baseline; consider longer training (100ep) at lr=1e-4 or lr=3e-4 with cosine decay.

## Artifacts

- Experiment dir: `experiments/2026-03-16_2227_claude_lr-stability-7allele/`
- Results: `results/runs/lr-stab-{s1..s6}-20260316b/`
- Manifest: `manifest.json`
- Probe CSVs: per-run `probe_affinity_over_epochs.csv`
- Reproducibility: `reproduce/`
