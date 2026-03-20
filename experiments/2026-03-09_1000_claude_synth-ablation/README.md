# Synthetic Negative Ablation

**Date**: 2026-03-09
**Agent**: Claude Code (claude-opus-4-6)
**Script**: `scripts/groove_baseline_probe.py`
**Raw data**: `modal_runs/synth_ablate/`

## Question

Which synthetic negative generation modes (if any) improve binding prediction? Are any structurally unsafe?

## Dataset

- **Source**: IEDB, IC50-only, exact qualifier
- **Alleles (2)**: HLA-A\*02:01, HLA-A\*24:02
- **Rows**: 7,793 total (6,234 train / 1,559 val)
- **Allele distribution**: A\*02:01=6,989; A\*24:02=804 (highly imbalanced)
- **Synthetic ratio**: 0.25 (1,558 synthetic rows added when active)
- **Probes**: SLLQHLIGL, FLRYLLFGI, NFLIKFLLI

## Training

- **Model**: GrooveTransformerModel (transformer, embed=64, hidden=128)
- **Warm start**: `mhc-pretrain-20260308b` (1 epoch)
- **Epochs**: 10, **Batch size**: 128, **Seed**: 42
- **Loss mode**: ic50_only, affinity_loss_mode=ic50_only

## Conditions (8 variants, 5 completed)

| Variant | Synthetic mode | Completed |
|---------|---------------|-----------|
| none | No synthetics (baseline) | Yes |
| all | All 6 modes combined | Yes |
| peptide_scramble | Scramble peptide sequence | Yes |
| peptide_random | Random peptide sequence | Yes |
| mhc_scramble | Scramble MHC sequence | Yes |
| mhc_random | Random MHC sequence | Pending |
| no_mhc_alpha | Remove MHC alpha chain | Pending |
| no_mhc_beta | Remove MHC beta chain | Pending |

## Results

### Val Loss

| Variant | Best Val Loss | @ Epoch |
|---------|--------------|---------|
| peptide_scramble | **0.660** | 9 |
| all | 0.677 | 9 |
| **none (baseline)** | **0.688** | 9 |
| peptide_random | 0.696 | 8 |
| mhc_scramble | 0.714 | 5 |

### Probe IC50 (at best epoch)

| Variant | SLLQHLIGL A02 | A24 | Correct? | NFLIKFLLI A24 | A02 | Correct? |
|---------|--------------|-----|---------|--------------|-----|---------|
| **none** | **110 nM** | 2,382 | Yes | **8 nM** | 69 | Yes |
| all | 20,868 | 43,182 | Yes (collapsed) | 4,857 | 33,226 | Yes (collapsed) |
| peptide_scramble | 26,647 | 27,514 | **Barely** | 8,975 | 41,394 | Yes (collapsed) |
| peptide_random | 32,915 | 46,817 | Yes (collapsed) | 40,345 | 44,217 | **WRONG** |
| mhc_scramble | 619 | 1,104 | Yes | 269 | 210 | **WRONG** |

### Structural Safety

- **mhc_scramble is structurally unsafe**: 54/128 synthetic rows used fallback groove parsing. Aberrant groove lengths (82:81, 59:86, etc.) vs canonical 91:93.
- All other modes preserve groove structure.

## Takeaways

1. **Val loss is anti-correlated with biological correctness**: peptide_scramble has the lowest val loss (0.660) but worst allele discrimination. The baseline has the best probe behavior.
2. **All synthetic modes destroy IC50 calibration**: Predictions jump from ~100 nM range (baseline) to ~20,000-40,000 nM. The model learns "everything is a non-binder."
3. **mhc_scramble is structurally dangerous**: produces noncanonical groove lengths.
4. **peptide_random inverts NFLIKFLLI**: predicts A\*02:01 as stronger binder (wrong).
5. **Recommendation**: Do not use synthetic negatives at this data scale. If re-introducing, use ratios << 0.25 and validate probe discrimination, not just val loss.
