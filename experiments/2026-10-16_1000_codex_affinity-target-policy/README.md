# Canonical affinity target-policy audit

Status: planned. Agent: Codex (GPT-6). This family checks the candidate Presto
0.1.16 census against the published 2,614 corrected-human IC50 observations.
See [the prospective plan](../agents/codex/plans/affinity-target-policy.md).

The canonical affinity policy is `log10(clip(nM, 0.001, 50000))` with unchanged
censor direction. Original values remain original measurements/bounds. This
family constructs label-only samples, retaining identity, class, allele sets and
source strings, without MHC sequences, a model, generated data or optimization.
It does not qualify full model inputs, output performance or a supported-output
manifest. Validation and test rows are audited but have no predictions because
no model is constructed. Donor/specimen/acquisition identities remain unresolved.

The input table preserves citations, assay types, cell context and conditions:
[public original/effective target audit](https://github.com/pirl-unc/presto/releases/tag/corrected-human-target-policy-audit-v1).
The [source preflight](https://github.com/pirl-unc/presto/releases/tag/corrected-human-preflight-v1)
retains the complete 9,053-row cohort and partition/leakage evidence. No existing
artifact is changed. This audit compares direct IC50 observations, not independent
experiments. Repeated objective/panel exposures must not be added together.

Run `reproduce/launch.sh` from the clean recorded recipe revision. It uses local
CPU, one Torch/BLAS thread, no GPU/workers, batches of 128 and a ten-minute alarm.
Inputs and outputs use datacache under `~/.hitlist/presto-affinity-target-policy`.
The script records runtime and versions; results and download pins follow after
execution. There are no fitted conditions, loss weights or baseline winners.
