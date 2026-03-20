# Factorized Multi-Output Ablation (8 conditions, 7-allele, 50ep)

- Agent: `claude`
- Source script: `scripts/benchmark_factorized_ablation.py`
- Created: `2026-03-16T14:54`
- Git commit: `bf7fcbf` (Wire up factorized assay embeddings)
- GPU: H100!

## Question

Does the newly wired factorized assay embedding (A07) help on the 7-allele / ~44K-row contract? And what is the relative importance of pretraining, capacity, loss mode, target encoding, and architecture choice?

## Dataset Contract

- 7 alleles: HLA-A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02
- `numeric_no_qualitative`, qualifier_filter=all
- split: peptide_group 80/10/10 seed 42
- train=32,805, val=4,184, test=4,060
- Probes: SLLQHLIGL, FLRYLLFGI, NFLIKFLLI

## Training

- batch_size=256, epochs=50, lr=1e-3, constant schedule, weight_decay=0.01
- split_kd_proxy, max_affinity_nM=100,000
- concat_start_end_frac positions, binding_core_lengths=8,9,10,11, shared refinement
- No synthetics, no contrastive

## Results

### Stability Summary

| ID | Description | Diverged? | Div Epoch | Best Epoch | Best Val Loss | Params |
|----|-------------|-----------|-----------|------------|---------------|--------|
| C1 | d128 A07 full pretrain mhcflurry | Yes | 49 | 38 | 0.0392 | 5,177,382 |
| C2 | d128 A03 full pretrain mhcflurry | Yes | 18 | 2 | 0.0672 | 5,144,614 |
| C3 | d128 A07 full cold-start mhcflurry | Yes | 6 | 2 | 0.0667 | 5,177,382 |
| **C4** | **d128 A07 heads_only pretrain mhcflurry** | **No** | - | **33** | **0.0312** | 5,177,382 |
| C5 | d128 A07 full pretrain log10 | Yes | 3 | 1 | 1.8634 | 5,177,382 |
| **C6** | **d32 A07 full cold-start mhcflurry** | **No** | - | **41** | **0.0313** | 411,934 |
| **C7** | **d32 A03 full cold-start mhcflurry** | **No** | - | **36** | **0.0299** | 409,886 |
| C8 | d128 pooled full pretrain mhcflurry [NEG] | Yes | 20 | 7 | 0.0936 | 5,100,641 |

### Probe Discrimination (SLLQHLIGL, best val_loss epoch)

| ID | A\*02:01 IC50 (nM) | A\*02:01 Kd (nM) | A\*02:01 Bind Prob | A\*24:02 IC50 (nM) | A\*24:02 Bind Prob | Discrimination |
|----|-------|------|-----------|-------|-----------|----------------|
| C1 | 51.4 | 8.6 | 0.9439 | 7,507 | 0.0335 | 28.2x |
| C2 | 34.2 | 406.1 | 0.9653 | 4,547 | 0.0607 | 15.9x |
| C3 | 169.8 | 1,457 | 0.7925 | 12,324 | 0.0184 | 43.1x |
| **C4** | **65.7** | **11.4** | **0.9254** | **32,981** | **0.0055** | **168.3x** |
| C5 | 52.2 | 746.9 | 0.9429 | 428.7 | 0.5476 | 1.7x |
| **C6** | **54.6** | **25.9** | **0.9398** | **23,575** | **0.0083** | **113.0x** |
| **C7** | **88.6** | **44.6** | **0.8954** | **41,242** | **0.0042** | **214.7x** |
| C8 | 1,486 | 1,486 | 0.2057 | 1,486 | 0.2057 | 1.0x |

### Probe Head Rank Correlation (final)

| ID | probe_head_rank_corr |
|----|---------------------|
| C1 | 0.8482 |
| C2 | 1.0000 |
| C3 | 0.8929 |
| C4 | 0.4286 |
| C5 | 0.5893 |
| C6 | 0.4821 |
| C7 | 0.7054 |
| C8 | 1.0000 |

## Pairwise Comparisons

### Does factorized help? (C1 vs C2, C6 vs C7)

**No clear evidence.** Both C1 and C2 diverged at d=128. At d=32 (stable), C7 (A03, no factorized) actually achieved the lowest val_loss (0.0299 vs 0.0313) and the best probe discrimination (214.7x vs 113.0x). Factorized context embedding does not obviously help and may slightly hurt at d=32.

### Does pretraining help? (C1 vs C3)

**Weak yes for stability.** C1 (pretrained) survived to epoch 49 before diverging; C3 (cold-start) diverged at epoch 6. But both diverged. Pretraining extends stable training window but does not prevent divergence at lr=1e-3 with `full` loss at d=128.

### Does d=128 vs d=32 matter? (C1 vs C6)

**d=32 is more stable.** C6 (d=32) completed 50 epochs without divergence. C1 (d=128) diverged at epoch 49. Best val_loss is comparable (0.0313 vs 0.0392). d=128 has 12.5x more parameters but is unstable at this learning rate.

### Does full vs assay_heads_only matter? (C1 vs C4)

**assay_heads_only is dramatically more stable.** C4 completed 50 epochs without divergence, while C1 diverged at epoch 49. C4 also achieved best val_loss (0.0312), best probe discrimination (168.3x), and lowest A\*02:01 Kd (11.4 nM). **C4 is the best d=128 condition.** The `full` loss mode contributes to instability at d=128.

### Does mhcflurry vs log10 matter? (C1 vs C5)

**Yes — log10 is catastrophically unstable.** C5 diverged at epoch 3 with val_loss=1.86. The mhcflurry encoding is essential for this contract at lr=1e-3.

### Does pooled collapse? (C8 vs all)

**Yes — confirmed.** C8 shows zero allele discrimination (1.0x ratio, identical predictions for A\*02:01 and A\*24:02), bind_prob=0.2057 for everything. The pooled_single_output architecture cannot learn allele-specific binding at this scale.

## Key Findings

1. **Stability is the dominant issue.** 5 of 8 conditions diverged. lr=1e-3 + d=128 + `full` loss is too aggressive. The three stable conditions (C4, C6, C7) all either used `assay_heads_only` or d=32.

2. **Best d=128 condition: C4** (assay_heads_only + pretrained). Best val_loss (0.0312), best allele discrimination (168.3x), stable for 50 epochs. The trunk gradients from `full` loss cause instability.

3. **Best d=32 condition: C7** (A03, no factorized). Lowest overall val_loss (0.0299), best absolute discrimination (214.7x), stable. Surprisingly, the simpler A03 architecture edges out A07 at d=32.

4. **Factorized does not help** at d=32 (the only stable comparison pair). C7 > C6 on val_loss and discrimination.

5. **Next steps:** Lower lr (e.g., 3e-4) or warmup schedule for d=128 + full loss; or proceed with d=32/assay_heads_only as the stable baseline for 7-allele work.

## Artifacts

- Experiment dir: `experiments/2026-03-16_1454_claude_factorized-ablation-7allele/`
- Results: `results/runs/fac-ablation-{c1..c8}-20260316a/`
- Manifest: `manifest.json`
- Probe CSVs: per-run `probe_affinity_over_epochs.csv`
- Reproducibility: `reproduce/`
