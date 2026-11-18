# Current model and workflow

## One supported runtime

| Responsibility | Canonical implementation |
| --- | --- |
| Model and configuration | `presto.models.Presto`, `PrestoConfig` (`presto.shared.v7`) |
| Typed observations | `presto.data.modular.Observation` |
| Source admission | `data/modular_hitlist.py`, `data/modular_hitlist_ms.py` |
| Protected presentation preparation | `data/presentation_preparation.py` |
| Training, selection and resume | `training/modular_run.py` |
| Measured endpoint evaluation | `training/modular_evaluation.py` |
| Complete-pair presentation metrics | `evaluation/presentation.py` |
| Public prediction | `presto.inference.ObservationPredictor` |
| Installed commands | `presto train`, `presto evaluate observations`, `presto predict observations` |
| Modal execution | `presto.scripts.modal_workflow` with a pinned wheel/request |

`ModularPresto` and `ModularConfig` are compatibility aliases for the same classes.
Model options are configurations and ablations within this implementation. New
training never selects a second class-specific trainer or a legacy model backend.

## Information flow

```text
Peptide + MHC sequences ── shared sequence encoder / pMHC interaction
                                  │
                    class I: whole peptide
                    class II: candidate 9-residue registers
                                  │
                     shared affinity decoder ── measured assay outputs
                                  │
Peptide / source flanks ── processing supply
APC cell / expression / interventions ── shared APC state
                                  │
               class-aware supply and register-selection effects
                                  │
                    presentation aggregation
                                  │
                   MS observation contribution ── observed elution score
```

The retained checkpoint uses a 64-dimensional harmonized APC state. An optional
shared junction module reads ordered N/C boundary residues and this state; it is
enabled in the provisional processing candidate. Known protein termini differ
from missing flanks. Unknown interventions differ from explicit controls.
Affinity excludes APC/source-flank inputs, but joint updates to the shared encoder
can change affinity, requiring retention checks. External NetMHC BA predictors
construct matched backgrounds; inference does not require them.

## Checkpoint decision

| Role | SHA-256 | Evidence |
| --- | --- | --- |
| Retained development default | `1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68` | Broad both-class training and validation; APC increment strongest in class I |
| Provisional junction candidate | `e4898af3a7a442a999177273687ea4e4ce5d4216e908f2e435b2d73226e56963` | Replicated pooled affinity-matched gains; class-II paired ordering uncertain |

Default location on Modal volume `presto-checkpoints`:
`2026-11-17_1155_codex_shared-presentation-expansion/fits/expanded-state-sqrt-s42-n6000/raw/selected.pt`.
The candidate is under
`2026-11-17_2330_codex_affinity-conditioned-processing/fits/paired-junction-s42-n2000/raw/selected.pt`.
Downloaded tensors remain outside version control; verify the hash above.
The new corpus does not change these checkpoint/data contracts retrospectively.

## Data to training

The [preparation contract](presentation_data.md) emits a source manifest. Add
`model`, `training` and, for a warm start, hash-pinned `pretrained` fields to a copy
of that manifest. Relative source paths remain relative to the manifest directory.
Use the exact parent configuration when retaining its architecture; expansions
require their explicit compatibility option. Preserve measured-affinity replay
when training presentation. Source preparation alone does not supply assay replay.

For broad presentation, the established selection contract is
`selection_endpoint: elution_observation`, `selection_reduction: class_study_mean`.
Use `sampling_mode: weighted_steps` with declared source/stratum weights and
`balance_by: [endpoint, mhc_class, study]`. Compute explicit stratum weights
from square-root study counts when reproducing that recipe; there is no implicit
study-power configuration switch.
No pilot caps or intervention-resolution whitelist is implied.
The manifest trainer supports exact-state resume; archived runs retain their
original runtime because changing code changes the resume contract.

Use `test_policy: no_test_training_diagnostic` with a recorded `no_test_reason`
for development while TEST stays sealed. Coverage remains required; the diagnostic
coverage exception is only for explicitly labeled software checks.

## Installed prediction and evaluation

Prediction JSONL records have `split: predict` and no measured `value`. Explicit
MHC groove sequences and candidate sets belong in the observation schema; allele
names are not residue sequences. `--unsupported-apc-policy unknown` deliberately
projects unsupported categorical annotations to unknown while retaining provenance.
The default is `error`. Processing diagnostics are already included in the score.

Every source in an evaluation manifest declares `split: train`, `validation` or
`test`. The command selects sources before opening payloads, verifies their hashes,
and rejects content that contradicts the declared split. Its default is validation.
It writes prediction dumps, endpoint/stratum metrics, complete-pair metrics and
a receipt. It refuses incomplete/duplicate pairs. TEST access requires explicitly
selecting TEST; this cleanup never does so.

## Historical boundaries

`legacy/` and older high-level prediction wrappers support historical checkpoints.
`predict presentation`, `predict tile`, `predict recognition` and `evaluate synthetic`
are historical interfaces, distinct from the current observation workflow. The
older `Predictor` import remains compatible; new callers use `ObservationPredictor`.
Frozen experiment snapshots are archival evidence, never dependencies of current
training, evaluation or prediction.

## Experiment workspace and installation checks

Python 3.11 or newer is supported. Training writes to `experiments/` in the current
working directory, or in the explicit `PRESTO_WORKSPACE` directory. The installed
package location only supplies code; it is never the output destination.
`--run-dir` must name a timestamped direct child of that workspace's `experiments/`.
The frozen source manifest and source snapshots remain attached to each run.

`python scripts/check_wheel.py dist/presto-*.whl` checks the actual wheel from a
neutral directory, including one artificial optimizer update, selected-checkpoint
reload, validation-only evaluation and label-free predictions for both classes.
These fixtures test software integration, not biological accuracy.
