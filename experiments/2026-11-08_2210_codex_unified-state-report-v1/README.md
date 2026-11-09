# Unified state report v1

Date: 2026-11-08  
Agent/model: Codex / GPT-5  
Status: generated development report; no new model training or TEST access

This experiment is the first execution of the versioned `presto-report`
contract. It summarizes the selected Stage B affinity/presentation checkpoint
plus the preferred binary-only assay-aware T-cell development head. It does not
change a checkpoint or promote a model.

The canonical input is `report_manifest.json`. The generated timestamped bundle
is under `reports/2026-11-08_221000_4935bd7f/`. Large or restricted row bundles,
checkpoints, comparator receipts, and per-example predictions remain on the
`presto-checkpoints` Modal volume at the exact paths, sizes, and SHA-256 hashes
recorded in the manifest and report.

The report preserves the current evidence boundary:

- Stage B processing decisively exceeds MHCflurry on the matched three-study
  development cohort;
- presentation is statistically similar, not superior;
- MHC-I and especially MHC-II exact-affinity correlation remain below the
  matched NetMHC tools;
- binary-only is preferred to binary+ordinal for exposed T-cell development,
  but fixed-threshold behavior is degenerate and PRIME/TEST remain unevaluated.

No current confirmation, locked T-cell cohort, or independent TEST was opened.
The report generator ran locally on CPU; the source training runs explicitly
requested and observed H100 80GB hardware.

## Frozen data and training contract

| Family | TRAIN rows | Development rows | Split/curation contract |
|---|---:|---:|---|
| presentation/processing | 1,349,378 | 48,530 | whole-study plus exact/core8/Hamming-1 embargo; reported ligands paired 1:1 with source-matched generated controls |
| affinity | 160,326 | 2,536 | whole-study plus cross-endpoint core9 embargo; literal endpoint/unit/qualifier and assay method retained |
| assay-aware T-cell | 15,646 | 1,530 | whole study/group chronology; five literal family/response/method tracks; observed labels only |

Stage B used 5,000 updates, batch 128, seed 42, a 50:50 EL/concentration
replay, and equal-study presentation selection. The fixed-parent T-cell run used
3,000 updates, batch 128, seed 42, and selected binary-only step 100 by
equal-functional-track/equal-study loss. No synthetic affinity rows, random
T-cell negatives, PRIME rows, or TEST rows were used. Generated presentation
controls remain a separately identified EL-background contract rather than
affinity negatives.

Assay label to output mapping remains explicit: affinity methods retain their
separate output tracks; presentation and operational processing are distinct;
T-cell activation, IFNg ELISPOT, IFNg ICS, TNFa ICS, and multimer binding are
five separate binary outputs. Conditional ordinal intensity was tested only as
an auxiliary T-cell condition and rejected.

## Current development metrics

| Task | Presto | Comparator/condition | Decision |
|---|---:|---:|---|
| MHC-I exact-IC50 Spearman (n=169) | 0.668853 | NetMHCpan 4.2c: 0.758445 | fail |
| MHC-II exact-IC50 Spearman (n=1,170) | 0.499492 | NetMHCIIpan 4.3e: 0.777863 | fail |
| presentation equal-study AUROC/AUPRC | 0.952049 / 0.952668 | MHCflurry: 0.948433 / 0.954281 | intervals cross zero |
| operational-processing equal-study AUROC | 0.905777 | MHCflurry processing/AP: 0.805672 | pass; paired delta 0.100104 [0.070592, 0.145770] |
| T-cell functional equal-track/equal-study AUROC | binary-only 0.630998 | binary+ordinal 0.491024 | binary-only preferred, not promoted |

There is no held-out TEST metric. PRIME 2.1 was audited but not executed because
the installed MixMHCpred 2.0.2 does not meet its 3.x requirement. Comparator
receipts pin NetMHCpan 4.2c, NetMHCIIpan 4.3e, MHCflurry package 2.3.16 with
weights 2.3.0, and the blocked PRIME 2.1 contract.

## Runtime and artifacts

Stage B ran 9,637.90 seconds; the binary-only T-cell run ran 1,842.35 seconds.
Both requested `H100!` and observed NVIDIA H100 80GB HBM3. The report itself was
generated locally on CPU. The selected T-cell checkpoint is 4,433,515 bytes,
SHA-256 `0c8eab7381631e14bc1982439bdf639a5adb6f89a4bbdbda23111bb444e5e145`.

All six frozen row bundles, four per-example development prediction artifacts,
the checkpoint, and four comparator receipts are recorded by immutable Modal
volume URI, byte count, license note, and SHA-256 in `report_manifest.json` and
the generated `data_manifest.json` / `predictions/index.json`. The report's
`checksums.sha256` covers every committed report file. No licensed binary or
large raw bundle is committed.

## Reproduction

Run `reproduce/launch.sh` from any source checkout containing commit
`4935bd7f6676a6a3fa003b8411062eb3c70ae010` or a reviewed descendant. By
default it writes a separate `reproduced-reports/` directory and therefore does
not overwrite the canonical report.
