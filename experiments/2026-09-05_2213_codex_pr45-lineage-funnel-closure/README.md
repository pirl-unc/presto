# PR #45 class-II lineage and Hitlist funnel closure

- Date: 2026-09-05
- Agent/model: Codex / GPT-5
- Code commit: `15003dacf310de3534325cb2db7fd6f3e2e89481`
- Hardware: local CPU; no GPU requested
- Status: complete; all acceptance gates passed

## Purpose

This deterministic data preflight closes two data-audit findings: standalone
DQ/DP beta alleles must be recorded as resolved when `mhcseqs` supplies their
model-facing `groove2`, and Hitlist rows removed by unresolved-flank filtering
must be exported in the canonical funnel JSON and CSV.

This is not predictive evaluation and performs no optimization. Validation and
test predictions, model metrics, loss weights, and assay-output mappings are
therefore not applicable. The fixed dataset and curation contract is recorded
in [`reproduce/launch.json`](reproduce/launch.json), and the exact executable
checks are frozen under `reproduce/source/`.

## Dataset and curation contract

The Hitlist condition used the unique `data_source=hitlist` path with flanks
enabled, HLA-A*02:01, `source_mapping_policy=mask_unresolved`, and
`HITLIST_DATA_DIR=/Users/iskander/.hitlist`. It loaded 16,721 binding, 2
kinetics, 2,150 stability, and 726,766 elution observations before caps, then
reservoir-sampled 96/2/24/96 respectively. Data and split seeds were fixed at
42; the loader's derived sampling seed was 59. `kon`, `koff`, and `tm` were
excluded, every synthetic ratio was zero, and MHC augmentation was disabled.

MHC inputs used the default `mhcseqs` source with no index CSV. Resolution was
63/63 alleles and 717/717 row-wise MHC inputs, with zero index fallback,
noncanonical sequences, or X residues. The resulting 216 samples split by
peptide into 130 train, 43 validation, and 43 test rows. The split support,
dataset, and supervision hashes are:

- `2587f4cc26fa531afb71fd999bc4d9517495a798ac7e78452835de62687f3cb2`
- `e1c7012b94ca6b787b0beab642247eb18140dca11db5b9e3b4a847b8ab0267f1`
- `4a1d280e5f23b2ce3c93d463215b79905c21d4b8cacae3b0728bc963bb873186`

## Results

| Check | Result |
|---|---|
| Real `HLA-DQB1*06:02` | beta; `groove1=0`, `groove2=93`; sample/batch lineage resolved |
| Real `HLA-DPB1*02:01` | beta; `groove1=0`, `groove2=91`; sample/batch lineage resolved |
| Hitlist binding unresolved-flank drops | 4 in JSON and CSV |
| Hitlist MS unresolved-flank drops | 235 in JSON and CSV |
| Lineage / duplicate-ID issues | 0 / 0 |
| Fake-null optional sequences | 0 |

[`data_funnel.json`](results/hitlist_preflight/data_funnel.json) retains the
complete source-loader statistics, including pre/post-collapse counts and the
nested mapping diagnostics, and also promotes the flank removals to
`drop_reasons.unresolved_flank`. The same two normalized rows appear in
[`data_funnel.csv`](results/hitlist_preflight/data_funnel.csv). The real
chain-placement evidence is in
[`class2_lineage.json`](results/class2_lineage.json), and artifact hashes are
collected in [`verification_summary.json`](results/verification_summary.json).

This was a data preflight, so no model was initialized, no pretraining or
training parameters/losses applied, no assay labels were mapped to outputs,
and no predictive validation/test metrics or prediction dumps were expected.
There is no winning model condition. The result is that both reviewed audit
contracts now hold on production loaders and real external data.

The local CPU run took approximately 73 seconds. No GPU was requested. There
are no external raw artifacts: the complete logs, configs, funnel, coverage,
split audit, and summaries are committed beneath `results/`. Reproduce with
[`reproduce/launch.sh`](reproduce/launch.sh) at the recorded code commit.
