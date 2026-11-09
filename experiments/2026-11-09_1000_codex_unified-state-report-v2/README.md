# Unified state report v2

Date: 2026-11-09  
Agent/model: Codex / GPT-5  
Status: closed; two independent renders are byte-identical and all report
checksums pass

This registered report closes the newer-code comparator-parity retrain at clean
source commit `1cca535e99a20e58c0ce00bc27edcbefbba3ad7d`. It selects the reference
checkpoint after the experimental BA parity arm failed the preregistered MHC-II
parent gate.

The frozen builder is `code/build_manifest.py`; manifest SHA-256 is
`f7a862fb83056e911805ee19b9a2d0c0e38d105c3c39a2f19e2be6652fea9e64`.
The canonical bundle is
`reports/2026-11-09_150000_1cca535e/`. Two independent renders contain 41 files
and are byte-identical. The committed bundle is 4.9 MB; every entry in
`checksums.sha256` verifies. Large row bundles, checkpoints, comparator outputs,
and per-example predictions remain on the hash-pinned `presto-checkpoints`
Modal volume.

This remains development evidence. No TEST or locked confirmation cohort was
accessed, and the operational PRIME smoke is not an immunogenicity benchmark.

## Model, data, and assay mapping

The retained checkpoint is the reference arm, 8,270,787 bytes, SHA-256
`1d0d4122abf022830ea97fcc247b1d28c0645d7d42d206130cdf8a5d7762815a`.
It restarted from Stage A SHA-256
`e6cc6fbb9950ace32efe418bfa96b119e1b94f0399e6843db0a16b0208937c06`
and ran 5,000 updates / 640,000 examples. Requested `H100!` resolved to an
NVIDIA H100 80GB HBM3. The model retains separate EC50, IC50, KD-via-EC50,
KD-via-IC50, presentation, and operational-processing outputs.

| Dataset role | Reference rows | Experimental-parity addition | Contract |
|---|---:|---:|---|
| presentation/processing TRAIN | 1,349,378 | 0 | source-matched paired controls; whole-study and exact/core8/Hamming-1 embargo |
| affinity TRAIN | 160,326 | 143,213 | literal endpoint/unit/qualifier/method; parity BA rows use unknown-method IC50 routing and complete development embargo |
| development | 51,066 | same rows | 48,530 presentation/processing plus 2,536 physical affinity; exposed, not TEST |
| historical T-cell context | 15,646 TRAIN / 1,530 development | not used by selected checkpoint | five assay-aware tracks; unpromoted |

No synthetic affinity labels were used. Presentation negatives remain explicit
source-matched generated EL backgrounds rather than affinity negatives. The
parity addition contains 80,674 MHC-I and 62,539 MHC-II comparator-archive BA
TRAIN rows; it is an experimental arm, not independent comparator evidence.

## Development results and decision

Equal-study estimates are primary; pooled exact-IC50 values are secondary.

| Task | Reference | Parity arm | Comparator | Result |
|---|---:|---:|---:|---|
| MHC-I exact-IC50 Spearman, equal-study | 0.5850 | 0.6211 | NetMHCpan 4.2c: 0.6834 | affinity gate fails |
| MHC-II exact-IC50 Spearman, equal-study | 0.5765 | 0.5400 | NetMHCIIpan 4.3e: 0.8070 | parity parent and affinity gates fail |
| presentation AUROC, equal-study | 0.9520 | 0.9523 | MHCflurry: 0.9484 | similar; superiority fails |
| operational-processing AUROC, equal-study | 0.9058 | 0.9068 | MHCflurry: 0.8057 | pass |

Parity improved MHC-I but changed pooled MHC-II Spearman by only `+0.00636`,
below the preregistered `+0.05` gate; its equal-study MHC-II delta versus
reference was `-0.03648`, 95% CI `[-0.08256,+0.00961]`. Reference is retained.
Parity minus MHCflurry processing AUROC was `+0.10114`, CI
`[+0.07829,+0.14520]`, while parity minus MHCflurry presentation AUROC was
`+0.00382`, CI `[-0.01193,+0.02070]`.

The report contains 64 registered metrics, 28 supplemental figure/table
artifacts, eight per-example prediction references, full data/checkpoint/
comparator hashes, and the independent 110-estimate replay receipt. The
standardized panels and their generating code are preserved rather than copied
from MHCflurry.

## Comparator and TEST status

Receipts pin NetMHCpan 4.2c, NetMHCIIpan 4.3e, MHCflurry package 2.3.16 with
weights 2.3.0, and PRIME 2.1 through mhctools with MixMHCpred 3.0. PRIME's
two-peptide smoke passed, but no accuracy evaluation was run. No independent
all-task TEST exists, and no TEST or locked confirmation row was accessed.

## Reproduction

Run `reproduce/launch.sh` from the source commit or a reviewed descendant. Set
`PRESTO_REPORT_OUTPUT_ROOT` to a new directory; the launcher rebuilds the
manifest from committed receipts and refuses to overwrite an existing report.
The canonical invocation and non-default `PYTHONPATH`/`MPLCONFIGDIR` are frozen
in `reproduce/launch.json`.
