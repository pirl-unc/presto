# Groove-corrected source-mapping baseline

- Experiment id: `2026-09-04_1121_claude_groove-corrected-baseline`
- Contributors: Claude (initial design); Codex / GPT-5 (data-trust remediation and launch gate)
- Status: **not launched**; all GPU conditions remain behind the mandatory exact preflight
- Base commit: `d79853d`; the remediation is carried on `claude/restore-unmapped-masking`
- Reproducibility bundle: [`reproduce/`](reproduce/)

## Question and conditions

This is a paired 2x3 comparison of source-junction handling after the corrected
`mhcseqs` groove parser:

| Policy | Seeds | Only intended input difference |
|---|---|---|
| `legacy_global_canonical` | 42, 43, 44 | Keep the deterministic selected junction for unresolved mappings. |
| `mask_unresolved` | 42, 43, 44 | Remove unresolved flank input while preserving the selected mapping as diagnostics-only lineage. |

It is deliberately a narrow HLA-A\*02:01 affinity/stability and
elution-decoy experiment. It is not a complete unified Presto corpus.

## Dataset and curation contract

- Source package: `hitlist==1.55.8`.
- Frozen Hitlist artifacts and `data/mhc_index.csv` are checked against the
  SHA256 values in [`code/launch.py`](code/launch.py) before every preflight and
  training condition.
- MHC parsing: `mhcseqs==2.6.10`; allele parsing: `mhcgnomes==3.41.0`.
- Allele filter: `HLA-A*02:01`; binding is single-allele, while selected
  elution samples retain their complete co-expressed allele sets. This is not
  a single-allele elution corpus. Peptide length uses the Hitlist loader
  default, 7--30 aa.
- Binding: all qualifying numeric IC50, EC50, and KD-family observations.
- Elution: reservoir-capped at 20,000 real positive observations before
  synthetic augmentation.
- Stability: reservoir-capped at 1,000 combined stability observations, then
  Tm is explicitly removed. The exact retained half-life count is a mandatory
  preflight output rather than assumed to be 1,000.
- Kinetics is loaded with a 500-row cap, then `kon` and `koff` are explicitly
  removed. Rows emptied by target exclusion are dropped.
- Absent modalities: processing, observed T-cell, TCR evidence, bulk MS, MHC-only
  augmentation, and UniProt negatives.
- Data seed base: 42. Hitlist reservoir sampling uses the trainer's documented
  `data_seed + 17` stream (59); synthetic generation uses the fixed data seed.
  Model/split seeds do not change capped or synthetic dataset membership.
- Split: peptide-disjoint 80/10/10 train/validation/test for each model seed.

Synthetic binding negatives are disabled: random/scrambled MHC or peptide inputs
would add a large second intervention to a flank-policy experiment. The explicit
elution-decoy ratio is 1.0, while both binding-to-elution and binding-to-T-cell
cascades are zero. These elution negatives are generated decoys, not observed
biological negatives. Their three generation modes (random peptide/real MHC,
real peptide/random MHC, and random peptide/random MHC) retain distinct source
labels, and their discrimination metrics must be described as real-positive
versus generated-decoy performance.

## Training and loss contract

- No pretraining checkpoint.
- Expanded Presto topology, `d_model=128`, 2 layers, 4 heads.
- 10 epochs, batch size 256, AdamW, learning rate `2.8e-4`, weight decay 0.01.
- Task-mean supervised-loss aggregation with learned uncertainty weighting;
  ordinary task base weight 1.0 and the existing MHC class/species/fine-type
  auxiliary weights 0.1.
- Existing consistency weights: cascade 0.2, assay-affinity 0.1,
  assay-presentation 0.1, no-beta2m 0.5, T-cell-context 0.05, and
  T-cell-upstream 0.2.
- Requested Modal GPU: `H100!`. Actual GPU name, memory, and driver are written
  to each run's `hardware.json`.

Observed-label routing is:

| Observation | Output / loss |
|---|---|
| Affinity value + qualifier | Censor-aware normalized `KD_nM`, plus the assay-specific `IC50_nM`, `EC50_nM`, or `KD_nM` output. |
| Half-life + qualifier | Censor-aware `t_half`. |
| Elution positive or generated decoy | BCE on `elution_logit` and `presentation_logit`. |

## Mandatory launch gate

`modal run code/launch.py` always runs all six exact CPU preflights before it
can spawn a GPU condition. Each preflight uses the production caps, exclusions,
synthetic data, MHC resolution, and split seed. It emits:

- `data_funnel.{json,csv}` with pre-cap/post-cap counts, explicit drop reasons,
  augmentation counts, and final source counts;
- `split_support.{json,csv}` from collated target masks, including censor and
  binary-class counts;
- stable source-observation/mapping lineage checks and a deterministic contract
  fingerprint.

The gate rejects missing split support for any active target, any active
one-class binary target, incomplete/duplicate lineage, fake null-derived
optional sequences, package/artifact drift, policy-dependent supervision or
split membership, and data membership that changes with the model seed. Each
GPU condition must reproduce its own preflight fingerprint exactly. A non-empty
remote run directory is rejected instead of reused.

The source-mapping policy is allowlisted to change only flank strings and flank
terminus state. Observation identity, labels, qualifiers, source mapping,
synthetic membership, and split assignment must match between policies.

All six exact local CPU preflights passed. The machine-readable summary is
[`results/local_preflight_summary.json`](results/local_preflight_summary.json),
and the seed-42 active-target table is
[`results/local_preflight_seed42_support.csv`](results/local_preflight_seed42_support.csv).
Dataset supervision membership is identical across all policies and seeds, and
paired split supervision is identical within each seed. The local environment
has editable Hitlist/mhcseqs checkouts shadowing two distribution versions, so
this evidence does not replace the launcher's exact pinned-package check.
The checked-in fingerprints were refreshed after lineage was split into the
complete source MHC allele set and the resolved model-input allele set; target
counts and the representative active-target support table are unchanged.

## Results and closure

No GPU runs have been launched. After a successful explicit launch, closure
must collect every run bundle, preserve per-example validation/test prediction
dumps with lineage, report held-out loss and binding Spearman/Pearson/RMSE plus
500 nM classification metrics, report generated-decoy discrimination separately
in `results/elution_decoy_metrics.csv`, record observed H100 hardware/runtime,
update this README, and add the completed
family to `experiments/experiment_log.md`.

The independent [`analysis/data_audit.py`](analysis/data_audit.py) is a
pre-augmentation real-source audit only. It is not the launch gate.
