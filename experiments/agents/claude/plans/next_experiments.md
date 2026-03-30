# Next Experiments

Experiments to run with the curriculum architecture changes from PR #3.
All can be run locally on MPS or on Modal with `scripts/train_class1.py`.

## Experiment 1: Regression check — verify no class I performance loss

**Purpose**: Confirm the architecture changes (core/PFR separation, processing enrichment) don't hurt the current best class I binding performance.

**Recipe**: L2 (DAG + heads_only + lr=3e-4 warmup_cosine + d128 + pretrained), 7-allele panel, 50 epochs, 3 seeds.

**Compare to**: 7-allele bakeoff L2 baseline (Spearman 0.806±0.006, F1 0.820±0.006).

**Run**:
```bash
# MPS
python scripts/train_class1.py --backend mps --seed 42
python scripts/train_class1.py --backend mps --seed 43
python scripts/train_class1.py --backend mps --seed 44

# Or Modal
python scripts/train_class1.py --backend modal --seed 42
python scripts/train_class1.py --backend modal --seed 43
python scripts/train_class1.py --backend modal --seed 44
```

**Pass criteria**: Spearman within 0.01 of baseline (0.796+). Probe discrimination ratio > 1000x.

## Experiment 2: Full class I best-hits (complete the bakeoff)

**Purpose**: Complete the 6-condition comparison at full class I scale (~105 alleles, ~250K rows) with mhcseqs groove sequences.

**Conditions**:

| ID | Residual | Loss | Pretrain | Question |
|----|----------|------|----------|----------|
| F1 | DAG | heads_only | yes | Control (L2 at scale) |
| F2 | A07 | heads_only | yes | DAG vs A07 at scale? |
| F3 | DAG | full | yes | full loss with more data? |
| F4 | DAG | heads_only | no | Pretraining at 105 alleles? |
| F5 | DAG | full | no | Cold-start stress test |
| F6 | A03 | heads_only | yes | 3-way architecture |

**Run**: `python experiments/2026-03-28_claude_class1-best-hits/launch.py --skip-launched`

Or individually on MPS:
```bash
# F1 (control)
python scripts/train_class1.py --backend mps --design-id F1

# F3 (full loss)
python scripts/train_class1.py --backend mps --loss-mode full --design-id F3

# F4 (no pretrain)
python scripts/train_class1.py --backend mps --no-pretrain --design-id F4

# F5 (full + no pretrain)
python scripts/train_class1.py --backend mps --loss-mode full --no-pretrain --design-id F5

# F2 (A07)
python scripts/train_class1.py --backend mps --residual-mode shared_base_factorized_context_plus_segment_residual --design-id F2

# F6 (A03)
python scripts/train_class1.py --backend mps --residual-mode shared_base_segment_residual --design-id F6
```

**Partial results** (from Modal, 5 of 18 runs):
- F1 (DAG): Spearman 0.767±0.002 (2 seeds)
- F2 (A07): Spearman 0.744 (1 seed)
- DAG is +0.023 ahead, consistent with 7-allele finding

## Experiment 3: Curriculum stage 2 — binding with contrastive + synthetics

**Purpose**: Test the curriculum sub-stages (2a → 2b → 2c) on full class I data. Does contrastive ranking help at 105 alleles? Do synthetic negatives help rare alleles?

**Run sequence**:
```bash
# Stage 2a: clean affinity (no contrastive, no synthetics) — the baseline
python scripts/train_class1.py --backend mps --design-id stage2a

# Stage 2b: add contrastive (init from 2a checkpoint)
python scripts/train_class1.py --backend mps --design-id stage2b \
    --init-checkpoint <stage2a_checkpoint>
    # TODO: add --binding-contrastive-weight and --binding-peptide-contrastive-weight flags

# Stage 2c: add synthetics (init from 2b or 2a)
python scripts/train_class1.py --backend mps --design-id stage2c \
    --init-checkpoint <stage2b_checkpoint>
    # TODO: add --synthetic-modes peptide_scramble --negative-ratio 0.1
```

**Note**: train_class1.py doesn't yet expose contrastive/synthetic flags. These would need to be passed as extra args or the script extended.

## Experiment 4: Curriculum param groups validation

**Purpose**: Verify that `curriculum_param_groups()` produces better results than training everything at once, by comparing:
- Full model training (all params at same LR) on class I affinity
- Stage-gated training (binding_class1 params only, trunk at 0.1x LR)

**Hypothesis**: Stage-gated training should match or beat full training because the frozen heads don't receive noisy gradients from an untrained affinity loss.

## Experiment 5: Class II binding (when class II data is available)

**Purpose**: Test the class II core scanning mechanism on class II affinity data.

**Prerequisites**:
- Class II alleles in merged_deduped.tsv (check with `--train-mhc-class-filter all`)
- Remove `--train-mhc-class-filter I` from train_class1.py

**Key metrics**:
- Class II binding Spearman
- Core posterior entropy (should be lower than random)
- Core accuracy on peptides with known cores (if available)

**Training**:
1. Train binding_class1 on class I affinity (stage 2a)
2. Switch to binding_class2, add class II data, init PFR module from checkpoint
3. Compare class II Spearman with and without PFR module

## Priority order

1. **Experiment 1** (regression check) — must pass before anything else
2. **Experiment 2** (complete best-hits) — answers the key architectural questions at scale
3. **Experiment 3** (curriculum stages) — tests the sub-stage progression
4. **Experiment 4** (param groups) — validates the curriculum infrastructure
5. **Experiment 5** (class II) — first real class II test, needs data curation
