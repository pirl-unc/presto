# Presto end-to-end on Brev — class I, hitlist-sourced

- **Date:** 2026-08-29
- **Agent/model:** claude (Opus 5), interactive session
- **Commit:** `42c8698` (branch `feat/gap2-and-cleanups`), clean except the launcher edit committed alongside
- **Run dir on box:** `/root/runplz-runs/20260829T195356Z-rc14-gcp-provision-train-dcb1b1bc/out`
- **Collected locally:** `brev_runs/presto-e2e-06/`

## Purpose

Validate that presto trains end to end on remote hardware and emits the
artifacts the experiment contract requires. **This is a plumbing run, not a
model worth using** — 2 epochs, `d_model=128`, 2 layers, 5.6M parameters.

Read the "What the metrics actually say" section before quoting any number
from it. Three of the five binary metrics do not mean what they appear to.

## Hardware

| | |
|---|---|
| requested | `min_gpus=1`, no pinned model |
| instance | `rc14-gcp-provision-full-training-3`, shared org box, `a2-highgpu-4g` (4x A100-SXM4-40GB) |
| observed | `NVIDIA A100-SXM4-40GB`, 39.49 GiB usable, ~27.3 GB peak |
| pinning | `CUDA_VISIBLE_DEVICES=0` — one device, other three left free |

Not a new instance: the box was already running and idle (mhcflurry release work
finished; heartbeat 46 h stale, all four GPUs at 0%). Taking one device of a box
already billing adds no marginal cost. AGENTS.md's `H100!` default applies to
Modal; this is Brev, where the org has only 4xA100 boxes.

## Data and curation contract

hitlist 1.45.0, built parquets staged at `/root/.hitlist`. `PRESTO_HITLIST_BUILD=0`
— the raw IEDB/CEDAR exports (14.7 GB) and proteome index cache (16 GB) are
build-time only and were not shipped; the 244 MB of built tables are enough.

| modality | rows |
|---|---|
| binding | 40,000 |
| elution | 60,000 |
| stability | 9,901 |
| kinetics | 47 |
| bulk MS (shotgun) | 40,000 |
| processing / tcell / tcr_evidence | **0** |

**The last row matters.** hitlist exposes only `ms` and `binding` evidence —
there is no T-cell or receptor table in it. So the immunogenicity, recognition
and TCR heads received **zero gradient for this entire run**. Supplying them
needs the merged TSV; `reproduce/launch_full_modality.sh` does that and the
1.0 GB file is staged at `/root/presto_data/merged_deduped.tsv`.

- MHC sequences: mhcseqs 2.5.12, 56,276 sequences. Coverage **95.25%**
  (212,992 / 223,621 rows).
- Split: **peptide-grouped**, 230,940 train rows / 106,547 peptides vs
  57,735 val rows / 26,733 peptides. Gap 1's fix is active in the real run,
  not only in tests — the previous row-level split leaked 41.7% of peptides.
- Synthetic negatives added: binding 49,941, elution 24,830, processing 0;
  cascaded downstream elution 17,199, tcell 17,235.
- No test split. Only validation metrics are reported here; a plumbing run does
  not justify burning the test split.

## Training

`epochs=2, batch_size=128, d_model=128, n_layers=2, n_heads=4, lr=3e-4,
seed=42, latent_topology=expanded, max_mil_instances=16, num_workers=8`,
bf16 autocast. 1,805 train / 452 val batches per epoch. ~48 min/epoch.

```
Epoch 1/2: train_loss=1.5580, val_loss=0.6319, lr=1.76e-4
Epoch 2/2: train_loss=0.9867, val_loss=0.6874, lr=3.0e-5
Best val_loss: 0.6319  (epoch 1)
```

**Validation loss rose in epoch 2 while training loss fell** -- 0.632 to 0.687
against 1.558 to 0.987. Two epochs is too few to call it, but the reported best
is epoch 1, so the second epoch made the model worse on held-out data and every
metric below comes from the epoch-1 checkpoint. A longer run needs early
stopping or a smaller LR, and should not assume more epochs help.

## What the metrics actually say

Held-out metrics were written for 16 tasks. Comparing each AUPRC against its
**base rate** is what separates a result from an artifact:

| task | AUPRC | base rate | lift | reading |
|---|---|---|---|---|
| elution | 1.0000 | 0.5390 | +0.461 | **artifact** |
| presentation | 1.0000 | 0.5390 | +0.461 | **artifact** |
| excision | 0.9998 | 0.5000 | +0.500 | **artifact** |
| ms_detectability | 0.9071 | 0.9042 | **+0.003** | **learned nothing** |
| foreignness | 0.6798 | 0.4135 | +0.266 | genuinely informative |

**elution / presentation / excision are not 1.0 because the model solved
presentation.** Positive and negative predictions do not overlap *at all*:
elution positives span [-0.74, 3.95] and negatives [-7.38, -1.55], with zero
negatives scoring above the lowest positive. Roughly half the elution
validation set is synthetic negatives — scrambled peptides, randomized MHC,
stripped alpha/beta chains — and those are separable on surface features
without any presentation biology. The metric measures "is this a real peptide
paired with a real MHC", which the model should of course ace.

**ms_detectability at 0.9071 is the more dangerous number**, because it looks
like a good score. The base rate is 0.9042: lift is +0.003. Positive and
negative predictions have nearly identical means (0.663 vs 0.671). The
detectability arm — the thing the excision/detectability design exists for —
learned nothing here.

Regression tasks, which have no base-rate trap:

| task | Spearman | n |
|---|---|---|
| binding_kd | 0.8417 | 2,321 |
| binding_affinity_probe | 0.7964 | 17,885 |
| binding | 0.7959 | 17,885 |
| binding_ic50 | 0.4747 | 1,445 |
| t_half | 0.1660 | 1,999 |
| tm | 0.1399 | 26 |
| binding_ec50 | 0.0882 | 49 |
| koff / kon | 0.0714 / -0.1071 | 7 / 7 |

Binding is the real result: **Spearman 0.796 on 17,885 held-out peptides** from
a peptide-disjoint split. This also settles an open question — an earlier toy
run showed binding Spearman −0.656 at n=77, and that was noise, not a sign
error.

`immunogenicity` and `tcell` report n=3,419 with no metric: their targets come
from cascaded synthetic negatives only, since hitlist supplied no T-cell rows.

## Takeaway

The remote path works and the artifact contract is met. The science is not
established by this run, and two of its headline numbers are traps:

1. **Synthetic negatives are too easy.** Any binary metric mixing them with real
   negatives is measuring peptide realism, not presentation. Either report
   metrics on real negatives only, or make the synthetic negatives hard.
2. **ms_detectability needs a real evaluation.** At a 90% base rate, AUPRC is
   nearly uninformative. `dual_corpus_transfer_set` (24,125 peptides) exists for
   exactly this and has still never been measured.
3. A full-modality run is needed before anything about immunogenicity, T-cell
   response or TCR evidence can be said at all.

## Artifacts

| file | what |
|---|---|
| `summary.json` | per-task held-out metrics |
| `val_task_summary.json` | per-task n, base rate, AUPRC, lift, and positive/negative score separation — the distilled form of the per-example dump, and what the AUPRC-vs-base-rate reading above is computed from |
| `val_metrics.csv` | flat metric table |
| `training_curve.json` | loss and LR per epoch |
| `config.json`, `launch_argv.json` | resolved config and exact argv |
| `mhc_sequence_coverage.csv` | resolution audit |
| `probe_affinity_over_epochs.png` | probe tracking |
| `reproduce/launch.sh` | frozen invocation for this run |
| `reproduce/launch_full_modality.sh` | variant adding the merged TSV |
| `reproduce/source/train_remote.py` | launcher snapshot |

**Raw artifacts, deliberately not committed** (gitignored, under
`brev_runs/presto-e2e-06/`, and on the box at
`/root/runplz-runs/20260829T195356Z-rc14-gcp-provision-train-dcb1b1bc/out`):

| file | size | why it is not here |
|---|---|---|
| `val_predictions.csv` | 6.7 MB, 127,101 rows | Everything this experiment concluded from it is in `val_task_summary.json`. Committing it added 127k lines to a pull request whose actual change is ~5k. |
| `metrics.csv` | 2,967 rows | Per-step logging; the per-epoch shape is in `training_curve.json`. |
| `model.pt` | 67 MB | A plumbing run's checkpoint, superseded by five later architectural changes. |

Regenerate any of them by re-running `reproduce/launch.sh`. Keep a raw dump
only when a future experiment will actually re-read it -- per-example
predictions earn their place when two runs are being compared example by
example, which is not the case for a single plumbing validation.

## What it took to get here

Six attempts, each a distinct real defect, five of which would have looked fine
from the outside:

| # | failure | fix |
|---|---|---|
| 1 | hitlist had no registered data | stage 244 MB of built parquets |
| 2 | **trained on 0/88,797 MHC, silently** | ship mhcseqs catalog; guard now refuses to start at zero coverage |
| 3 | CUDA OOM at 36 GB | effective batch is `batch_size x candidates`, not `batch_size` |
| 4 | 28 h/epoch | `PRESTO_MAX_BULK_MS=0` means *unlimited* |
| 5 | dead at batch 1, `received 0 items of ancdata` | FD exhaustion — ~50 tensors/batch; `file_system` sharing |
| 6 | — | completed |

Only #3 announced itself honestly.
