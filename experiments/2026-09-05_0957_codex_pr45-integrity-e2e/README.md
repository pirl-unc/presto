# PR #45 data-integrity end-to-end smoke

- Date: 2026-09-05
- Agent/model: Codex / GPT-5
- Launch commit: `c4dc1cf6c60ccca68eb3e467fffd70b3067945df` (clean)
- Hardware: local CPU; no GPU requested
- Status: complete; all acceptance gates passed

## Question and scope

Can the reviewed canonical paths load the default merged corpus and an exact
Hitlist-only corpus, resolve model-facing MHC sequences directly from
`mhcseqs`, train a model, reload the selected checkpoint, and emit complete,
traceable validation/test predictions?

This is a functional integrity smoke, not a model-quality comparison. The tiny
model, one epoch, and two training batches deliberately make its predictive
metrics unsuitable for scientific interpretation.

## Dataset and curation contract

The first condition preflighted `data_source=merged_tsv` with reservoir caps of
96 binding, 24 kinetics, 24 stability, 24 processing, 96 elution, 96 T-cell,
and 24 TCR-evidence rows. It scanned 3,266,972 source rows and recorded the
complete pre-cap, cap-loss, ingest-drop, MHC-filter, and final-sample funnel in
[`results/merged_preflight/data_funnel.json`](results/merged_preflight/data_funnel.json).
After MHC filtering it produced 287 samples and a peptide-grouped 173/57/57
train/validation/test split.

The training condition used the unique `data_source=hitlist` path for the
HLA-A*02:01 slice, even though the merged TSV was present. It reservoir-sampled
96 of 16,721 binding, both kinetics, 24 of 2,150 stability, and 96 of 726,766
elution records. `kon`, `koff`, and `tm` were excluded, removing the two sparse
`koff` observations and retaining 96 binding, 24 half-life, and 96 elution
samples. Every synthetic-data ratio and MHC augmentation was zero. The
peptide-grouped split, with separate data and split seeds fixed at 42, contained
130/43/43 samples. Required binding/half-life/elution support was 59/16/55 in
train, 18/4/21 in validation, and 19/4/20 in test.

Both conditions used strict, resolved-only MHC inputs. No index CSV was
provided. The merged preflight resolved 232 of 269 reported alleles through
`mhcseqs`, explicitly filtered unresolved model inputs, and then had 296/296
resolved model-facing rows. The Hitlist condition resolved 63/63 alleles and
717/717 row-wise MHC inputs through `mhcseqs`; index fallback use was zero.

All split audits report zero lineage issues, zero duplicate sample IDs, and
zero fake-null optional sequences. The Hitlist support, dataset, and supervision
hashes are respectively:

- `2587f4cc26fa531afb71fd999bc4d9517495a798ac7e78452835de62687f3cb2`
- `e1c7012b94ca6b787b0beab642247eb18140dca11db5b9e3b4a847b8ab0267f1`
- `4a1d280e5f23b2ce3c93d463215b79905c21d4b8cacae3b0728bc963bb873186`

## Training and output contract

No pretraining checkpoint was used. The model used the canonical expanded
latent topology with `d_model=32`, one layer, four heads, 411,722 parameters,
AdamW at `2.8e-4` with weight decay `0.01`, batch size 16, and a 16-instance MIL
cap. It ran one epoch on CPU with two capped training batches and one capped
in-loop validation batch. Learned uncertainty weighting was disabled. Standard
losses and consistency terms were retained; synthetic-data losses had no
support because synthetic generation was disabled.

Assay supervision maps as follows: binding affinity observations supervise the
binding/affinity-family outputs and their assay-specific IC50/KD variants;
half-life observations supervise `assays.t_half`; elution detections supervise
elution/presentation outputs. Source organism and MHC annotations supervise the
corresponding auxiliary species/class/type outputs where available.

After training, the runner reconstructed the model from the epoch-1
best-validation checkpoint, including its expanded topology, and evaluated all
validation and test batches. It wrote full loss terms and per-example dumps:

- [`val_predictions.csv`](results/run/val_predictions.csv): 132 task rows for
  all 43 validation examples
- [`test_predictions.csv`](results/run/test_predictions.csv): 130 task rows for
  all 43 test examples
- [`val_summary.json`](results/run/val_summary.json)
- [`test_summary.json`](results/run/test_summary.json)

No `holdout_error.json` was produced.

## Held-out metrics

RMSE is in `log10(nM)`. “Exact” excludes censored affinity observations.

| Split | Overall loss | Binding n | Exact n | Exact Spearman | Exact Pearson | Exact RMSE | <=500 nM bal. acc. | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 3.3841 | 18 | 14 | -0.2044 | -0.2718 | 1.6213 | 0.5000 | 0.3889 | 0.3433 |
| Test | 3.2907 | 19 | 13 | -0.1593 | 0.0877 | 1.6278 | 0.5000 | 0.4167 | 0.2420 |

Half-life support was four observations per held-out split: validation
Spearman 0.4000, Pearson 0.6790, RMSE 3.3849; test Spearman -0.4000, Pearson
-0.4702, RMSE 2.6119. Elution was all-positive in this intentionally
synthetic-free Hitlist smoke (21 validation, 20 test), so AUROC/AUPRC are
undefined and the classification metrics cannot establish discrimination.

## Result and limitations

The end-to-end integrity contract passed. The two named data sources take
distinct loaders; the merged loader exposes its full curation funnel; the
Hitlist loader does not absorb merged data; `mhcseqs` supplies model inputs
without an index CSV; source and resolved allele lineage remain separate; the
selected checkpoint reloads exactly; and complete held-out artifacts are
present.

There is no model winner in this experiment. Its low predictive metrics are
expected from two optimizer steps and should not be compared with full training
runs. The Hitlist slice remains narrow and its elution labels remain one-class;
this run proves plumbing and integrity, not corpus completeness, balance, or
scientific model quality. Plots would not add useful information at this scale.

## Runtime, artifacts, and reproduction

The merged preflight took 59 seconds and the Hitlist train/evaluate path took
53 seconds (112 seconds total wall time). Structured outputs, summary tables,
and prediction dumps are committed under `results/`; the compact digest is
[`results/experiment_summary.json`](results/experiment_summary.json).

Large/raw artifacts remain outside the experiment directory:

- `artifacts/2026-09-05_0957_codex_pr45-integrity-e2e/model-c4dc1cf.pt`
  (`sha256=eabcd8954190d4638aa7529cab519f4c562cfea17c5292a37cb3b9d78ca3d7d9`)
- `artifacts/2026-09-05_0957_codex_pr45-integrity-e2e/train-c4dc1cf.log`
  (`sha256=933527177c3009328b7488bb3dd940e66bd922efa75ae64ff4958bbedc3d21bd`)
- `artifacts/2026-09-05_0957_codex_pr45-integrity-e2e/merged-preflight-c4dc1cf.log`
  (`sha256=2b13de87d9fe5312051c42aca8c93ff652cec25d3d56cd7db85662615a55cd14`)

Run `reproduce/launch.sh` from a clean checkout at the launch commit with
`HITLIST_DATA_DIR=/Users/iskander/.hitlist`. Exact commands are frozen in
`reproduce/source/`; all non-default parameters are recorded in
`reproduce/launch.json`.
