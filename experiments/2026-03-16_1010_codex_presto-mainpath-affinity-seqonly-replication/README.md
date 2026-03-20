# Presto Main-Path Seq-Only Affinity Replication

- Agent: `codex`
- Source script: `code/launch.py`
- Status: `completed`
- Result: `failed replication`

## Goal

Rebuild the current best 2-allele broad-numeric binding contract on top of the main `Presto` model path while enforcing a strict sequence-only input contract:

- allowed inputs: `peptide`, `nflank`, `cflank`, `mhc_a`, `mhc_b`
- forbidden model inputs: binding assay type / method / prep / geometry / readout

The source configuration to beat was the EXP-21 winner:

- `groove c02`
- `d=32`, `n_layers=2`, `n_heads=4`
- `mhcflurry`
- `max_nM=100k`
- `50` epochs
- no warm start

## Dataset Contract

- Source: `data/merged_deduped.tsv`
- Alleles: `HLA-A*02:01`, `HLA-A*24:02`
- Included assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Measurement profile: `numeric_no_qualitative`
- Qualifier policy: `all`
- Split: peptide-group `80/10/10`, seed `42`
- Held-out artifacts: `val_predictions.csv`, `test_predictions.csv`

## Training Contract

- Model path: main `Presto` model via `Presto.forward_affinity_only(...)`
- Assay-input setting: `affinity_assay_mode=none`
- Assay residual mode: `pooled_single_output`
- Output supervision: `assay_heads_only`
- Target encoding: `mhcflurry`
- `max_affinity_nM=100000`
- `d_model=32`, `n_layers=2`, `n_heads=4`
- Optimizer: `AdamW(lr=1e-3, weight_decay=0.01)`
- Batch size: `256`
- Epochs: `50`
- Synthetic negatives: `off`
- Binding ranking losses: `off`
- Requested Modal GPU: `H100!`
- Observed peak reserved memory: `7.86 GiB`

## Outcome

This configuration did not reproduce the benchmark winner.

Final held-out metrics:

| split | loss | Spearman | Pearson | RMSE log10 | AUROC | AUPRC | accuracy | balanced acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | `0.1024` | `0.0214` | `-0.0079` | `1.5590` | `0.5004` | `0.4097` | `0.6019` | `0.5000` |
| test | `0.1088` | `0.0210` | `-0.0253` | `1.5459` | `0.5188` | `0.4324` | `0.6026` | `0.5000` |

Comparison to the source benchmark:

- EXP-21 best single run (`groove c02`, 50 epochs): test Spearman `0.8541`
- This main-path seq-only replication: test Spearman `0.0210`
- Gap: `-0.8331`

## Failure Mode

The run converged to a near-constant prediction band instead of learning useful rank structure.

Evidence:

- test predictions are concentrated in roughly `1094 nM` to `1733 nM`
- true held-out targets span roughly `0.1 nM` to `824,904 nM`
- every held-out prediction is above the `<=500 nM` binding threshold, so:
  - precision = `0`
  - recall = `0`
  - balanced accuracy = `0.5`
- the probe panel also collapses:
  - at epoch `50`, `SLLQHLIGL`, `FLRYLLFGI`, and `NFLIKFLLI` all land at about `1094 nM` on both alleles

Training also drifted in the wrong direction over time:

- best validation loss was at epoch `9`: `0.0875`
- final validation loss at epoch `50`: `0.1024`

So this is not just a mild underperformance. Under this contract, the main seq-only Presto path is effectively non-competitive with the current binding baseline.

## Interpretation

What this experiment establishes:

- the main `Presto` codebase can now run a strict assay-free affinity benchmark with the intended sequence-only inputs
- the new closeout path writes held-out prediction dumps and final metrics correctly
- this exact seq-only main-path configuration is not a viable replacement for the existing EXP-21 groove baseline

What it does not establish:

- that seq-only main-path Presto cannot work at all
- that the backbone is the only problem

The most plausible immediate issue is contract mismatch in the affinity training/output path:

- the shared main-path affinity head stack under `assay_heads_only + pooled_single_output + no assay input`
  appears to collapse to a weak pooled calibration solution
- removing assay inputs alone is not enough to make the current main-path head behave like the benchmark groove model

## Artifacts

- Run dir: `results/runs/presto-mainpath-affinity-seqonly-e050-s42/`
- Summary: `results/runs/presto-mainpath-affinity-seqonly-e050-s42/summary.json`
- Validation predictions: `results/runs/presto-mainpath-affinity-seqonly-e050-s42/val_predictions.csv`
- Test predictions: `results/runs/presto-mainpath-affinity-seqonly-e050-s42/test_predictions.csv`
- Condition summary: `results/condition_summary.csv`
- Epoch summary: `results/epoch_summary.csv`
- Fetch status: `results/fetch_status.json`
- Repro bundle: `reproduce/`

## Operational Note

During collection, `scripts/fetch_experiment_modal_runs.py` initially misclassified the run as complete because this trainer writes `summary.json` every epoch. That collector has now been fixed to honor per-run `required_files`, and this experiment manifest now requires:

- `summary.json`
- `val_predictions.csv`
- `test_predictions.csv`

## Takeaway

Keep EXP-21 `groove c02` as the model to beat for this contract. Do not promote this main-path seq-only Presto run as a baseline.
