# Canonical real-source coverage evidence — #48/#50

Agent/model: Codex / GPT-6. Prepared 2026-09-09 after the routing, lineage and
selector repairs (#61/#63/#64). No training-quality claim follows from this work.

## Scope and execution decisions

Use the existing `scripts.train_iedb.run(..., data_preflight_only=True)` path for
uncapped ingestion, filtering, sample identity, augmentation and peptide-group
splitting. Use its `audit_training_coverage` and effective loss target resolvers.
Do not implement an alternative trainer or reconstruct globally numbered sample
IDs from independently loaded chunks. Canonical collation already processes
bounded chunks and the declared-output census stores exact distinct evidence in
SQLite. Retain that database as a raw artifact instead of discarding it.

The earlier 11:06 inventory remains an immutable historical source inventory.
Register a new full-census family because the effective curation contract changed
in #61 and metadata propagation changed in #63/#64. Link the two families.

Execute these source/augmentation conditions separately:

| Source | Measured population | Canonical augmented population |
|---|---|---|
| merged TSV | all supported modalities, no generated records | explicit default generation ratios |
| exclusive Hitlist | binding/stability/kinetics/MS only | explicit default generation ratios |
| Hitlist plus bulk MS | same exclusive Hitlist plus explicit bulk supplement | default ratios plus wrong-enzyme generated labels |

For the measured bulk condition, set wrong-enzyme generation to zero while
retaining the observed bulk-derived proxy objectives. A proxy is not a direct
biological measurement. Do not union merged T-cell/TCR observations into Hitlist.
Keep original source contamination visible; this audit does not silently repair
Hitlist #444 or mutate the caches. Any source correction gets a separate contract.

All modality caps are zero (unlimited), including bulk. Use data seed 17 and
peptide-disjoint train/validation/test 80/10/10 with split seed 42. Require complete
class-correct MHC resolution and apply the canonical unresolved-MHC filter and
`source_mapping_policy=mask_unresolved`. Freeze the mhcseqs catalog and fallback
MHC index alongside the exact source artifact hashes and source build sidecars.
Record requested and actual MHC coverage and all mutually exclusive drop reasons.

Freeze every resolved trainer argument in JSON before launch, including absence
of optional UniProt inputs. Measured conditions set all negative-generation
ratios and MHC-only augmentation to zero. Augmented conditions explicitly set
pMHC/elution/cascade ratios to the resolved canonical defaults, class-I missing
beta to 0.25, processing to 0.5, and MHC-only requests to 60,000 with fraction 0.05.
Record generated families independently and diagnose parent-provenance limits.

## Hardware and reproducibility

The local machine has 32 GiB RAM and is already under compression pressure.
Run the full materialization on Modal CPU: request 4 physical cores (limit 8),
64 GiB RAM (hard limit 192 GiB), no GPU, and a four-hour per-condition timeout.
Run the first merged measured condition alone; inspect its real memory/runtime
and artifacts before scheduling the remaining conditions. Do not retry a timeout
or OOM unchanged, or substitute a capped census. Preserve the failed receipt and
write a revised plan before any resource adjustment.

Use an isolated Python 3.12 image with the inventory's exact release pins and a
CPU torch 2.7.0 build. Record the complete actual Linux package environment,
including platform differences. The existing Modal client is 1.1.4; freeze its
version and launcher API. No shared environment upgrade. Use a frozen archive
of a clean, merged-main-derived commit; record hashes for every production file
and launcher. Hash every source before and after each condition. Upload into a
new experiment-specific volume prefix; do not replace existing shared objects.
Preserve a durable detached app/call handle and commit outputs on success/failure.

Source-package review found tracked historical raw ZIP/FASTA files under `data/`.
Archive only Python package sources, packaging files, the required 1.5 KB B2M
resource and the current experiment's executable/configuration files. Raw
datasets enter exclusively through the explicit ten-file input manifest. Retain
the first local broad archive as an unlaunched preparation receipt; do not ship it.

Automatic approval review rejected the ten-file Modal upload because it requires
explicit user authorization for the local payload and external destination.
No data was transferred and no cloud computation launched. Complete code/review
work and request approval for the concrete manifest and `iskandr` workspace /
`presto-data` volume / `2026-09-09_1541_codex_canonical-coverage/` prefix. Do not
retry the upload or indirectly export the same payload pending that authorization.

## Instrumentation and verification

Experiment-local wrappers may add receipts, progress, SQLite retention and
deterministic training-candidate collection around the canonical audit. They
must delegate unmodified target resolution, collation and support gates, restore
patched references in `finally`, and fail loudly on input/hash drift. No source
filters, labels, split assignments or support thresholds are selected in hooks.

Verify fixture parity against the uninstrumented canonical audit, retained
database counts, failure cleanup and source/hash rejection before full execution.
Use a real-data census only after those meaningful checks pass on frozen code.
Write periodic phase/progress and resource information so stalled stages can be
distinguished from live computation. Preserve complete endpoint/column zeros,
roles/families, exact/censor/response balance, lineage, alleles and context facets.
Reconcile SQLite evidence, report matrices, split fingerprints and loader funnels.
Count T-cell pathway bags and all selected axes explicitly for #50.

## Update diagnostics and subsequent quality gate

Collect deterministic candidates exclusively from the real training split while
censusing. Before optimization, freeze their exact IDs/payloads, selection rule,
all tested configurations, step budget, optimizer and loss weights in a separate
phase receipt. Use fresh d128/l2/h4 models and the canonical train/update tracking
path. Compare expanded/collapsed and explicitly declared residual/KD grouping
configurations; preserve unsupported and untested columns separately. Specify
H100! if Modal GPU diagnostics are used and record observed hardware/memory.

Declare support requirements prospectively before #53 fitting, including minimum
distinct observations/peptides and binary balance per split/column. Report direct,
proxy, generated and auxiliary claims separately; aliases and repeated losses
must not increase independent endpoint counts. No threshold is fitted to a
predictive result. Census/update evidence is not validation/test prediction
quality; those metrics and per-example predictions belong to #53's fresh runs.

## Completion gate

Close every materially informative condition in its README and the canonical
experiment log, retaining exact invocations, frozen source, complete summary
tables/JSON, raw artifact locations and failure evidence. Link confirmed Hitlist
findings upstream after checking existing issues. Review the final evidence PR,
run the relevant tests/CI, merge, then continue #53. Do not close #48/#50 on
infrastructure or fixtures without the required real-source evidence.
