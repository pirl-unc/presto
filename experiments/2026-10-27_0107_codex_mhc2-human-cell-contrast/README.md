# Human endogenous MHC-II condition-source qualification

Status: audit complete; source not admitted for paired condition fitting.
Codex / GPT-6,2026-10-27. [Prospective plan](../agents/codex/plans/2026-10-27_mhc2-human-cell-contrast.md).
[Hitlist858](https://github.com/pirl-unc/hitlist/issues/858) is filed and its body
verified by readback. Existing control/engineering work in70/576/716 is preserved.

## Evidence and source contract

The [primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9724678/) describes
human endogenous immunopeptidomics, broad class-II capture, five technical runs,
algorithm-aggregated core intensities and computational DR affinity estimates.
These evidence types cannot establish measured peptide-specific binding.
The [PXD024417 deposit](https://www.ebi.ac.uk/pride/archive/projects/PXD024417) has
40raw files,40MSF search outputs and one community SDRF. All40SDRF raw-file names
join the inventory; eight source/capture groups each have five technical runs.
The paper states two biological siRNA experiments; the deposit mapping has one
source per siRNA condition and two Mock sources per capture class. The biological
relationship remains unresolved. SDRF extended typing has no declared method and
is retained as unverified secondary annotation. No default MHC partner is added.

Current2.0.42 curation13b91424295e01286db46e8e9818fcc2fe816834 and a fresh default-
branch read agree across the165requested lines, disregarding only the connector's
omitted final newline. Full-file GitHub blob97b951a201686104f8acc1c1482e86494dfaa80b.
The corrected-v4 export is a separate artifact, SHA256
`a665f37c0b5917fefb6d12eade530d7e2001de1f0ee73479b21d2c393d173ab4`.
Its path/hash matches both the published1803audit and downloaded checksum manifest.
Do not confuse it with the older f902341e... native export frozen for2036 fitting.

## Independently replayed source census

| Observation scope | Rows |
|---|---:|
| All PMID36215666 |14903|
| Class I |3028|
| Class II / distinct peptides |11875|
| Class II, one literal condition |5980|
| Class II, shared condition memberships |5895|
| Class II, quantitative values |0|

All class-II rows retain generic class-only restriction and an empty allele set.
Single-condition counts: Mock3031,siCTRL990,siT6BP1959. Shared memberships remain
source statements; they are not independent per-condition measurements. Missing
detections are not measured negatives. Current curation retains controls and
perturbation identities, while original run-specific measurements, scales,
replicate links and a complete primary-attested molecule set remain unqualified.

The source inventory is useful for finding the next data requirement, but it
cannot yet support the desired paired-MHC condition fit. Hitlist858 requests the
primary measurement path and explicit source/replicate qualification. No model,
synthetic controls, loss, pretraining, validation/test split, predictive metrics
or prediction dumps apply to this audit. No predictive winner is selected.

## Reproduction and closure

`reproduce/launch.json`/`launch.sh`, `metadata.sh`, `sdrf.sh`, `audit-v2.sh`,
`qualify-v2-launch.json` and frozen sources pin code, environment, git dirt and
source identities. `results/` contains14903projected source rows with physical
row indices, complete distributions, verified qualification and the PRIDE
inventory. Raw primary XML/API/SDRF files remain under
`/private/tmp/presto-mhc2-20261026/human-cell-contrast/`, with hashes in receipts.

The successful audit took6.770s/386,514,944bytesRSS, localCPU/one thread. No Modal,
GPU or installs. Original wrong-export hash failure stopped before row selection;
it and the connector-terminal-newline verification failure are preserved with
explicit corrections. Thirteen of16network operations were used, including issue
creation/readback; discovery is now closed. Direct retrieved payloads total
465357bytes. Web/connector network byte usage is not measured. Broad scientific
goal remains active, including actual fitted pathway/condition qualification.
