# Consolidated foundation verification

2026-11-18 UTC · Codex / GPT-6. Software consolidation and protected data
preparation; no new biological training or predictive-performance claim.

## Contract and reproduction

Base commit: `16df727f95780c72a8337c59610f247ae927cbe1`, with uncommitted integration
at execution. Promoted implementation: `aaec929206e834e3a3cd734a8bd2b1ecc3214356`.
The exact executed wheels, frozen Modal launcher sources, requests, input hashes,
call IDs and launch metadata are in `reproduce/`. Run `bash reproduce/launch.sh`
for the original preparation contract; its outputs already exist and are protected
against overwrite. For a new execution, choose a new remote namespace and freeze
that changed request. `reproduce/audit/launch.sh` records the independent pair audit.

The shared launcher uses an installed wheel; it imports no experiment-local model.
Preparation/audit requested 8 CPUs and 65,536 MiB memory, **no GPU**. `H100!` is the
launcher default only if a future request explicitly selects GPU execution.
Preparation app: `ap-IfujCVNhUImvZjEY5sz5pH`; audit app:
`ap-Jr1dBnHCbj0DgN3N97EULH`. Receipts and raw prepared observations remain in Modal
volume `presto-checkpoints` under `2026-11-18_0315_codex_foundation-integration/`.

## Exact data and curation

- Corrected Hitlist corpus v4 native manifest:
  `2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7`.
- Public Hitlist 2.1.12 loaders; original reported/derived/donor_set roles retained.
- Uncapped export: 2,842,058 class-I and 1,408,813 class-II source observations.
  These are pre-admission source rows, not training pairs.
- Native MS detections are positives. Known modified peptidoforms and unresolved
  MHC assignments follow centralized adapter exclusion rules; unspecified PTMs
  are explicitly sequence-only unknown. Unknown APC state remains unknown.
- Original measurement identity deduplication, explicit source exclusion for the
  untransfected class-II parental arm, reference-protein mapping, and identical
  APC/MHC context within generated pairs. Ambiguous flanks remain unknown.
- Generated backgrounds are unobserved same-length windows from a possible source
  protein. They are **not measured nonbinders**. No BA/EL prediction supplies labels
  for this broad preparation; the separate affinity-matched benchmark is unchanged.
- Complete declared retained-checkpoint ancestry, fixed validation and candidate
  background identities are individually pinned in `reproduce/identity-sources.json`.
  Exact/shared-8mer/Hamming-distance-one embargoes and source-family protection
  precede split assignment. Sealed TEST contributes peptide identities only.

Native admission produced 509,295 class-I and 305,528 class-II observations before
reference mapping, family protection and background qualification; conflicting
original identities: zero. Full exclusions and source hashes are in
`results/preparation.json`.

| Written population | Positive/background pairs | Studies |
| --- | ---: | ---: |
| TRAIN class I | 161,573 | 156 |
| TRAIN class II | 134,127 | 42 |
| New VALIDATION class I | 2 | 1 |
| New VALIDATION class II | 0 | 0 |

The independent `audit-presentation` run passes file hashes, complete-pair identity,
APC/MHC consistency, source windows, observed-peptide exclusion, and related
held-out/ancestry protection (`results/pair-audit.json`). Preparation took
1,646.66 seconds; its complete remote execution took 1,816.39 seconds. The
independent audit took 90.61 seconds.

The stronger accumulated ancestry embargo leaves essentially no new validation.
**Keep fixed development validation for class-balanced checkpoint selection.**
`results/training-sources.json` composes the corrected training pairs with immutable
measured-affinity replay and existing fixed validation. It supplies source contracts,
not a trained checkpoint or an implicit training configuration. More source rows
have not produced a claim of more usable pairs or independent evidence.

## Software and checkpoint checks

The installed-wheel smoke executes one artificial optimizer update, checkpoint
selection/reload, validation-only evaluation and label-free inference for both
classes. An explicitly declared nonexistent TEST source is never opened during
validation. Artificial IC50 targets use the normal endpoint/unit mapping and
manifest trainer; they establish software behavior only. Their manifests, selected
run summaries and validation prediction dumps are preserved in `results/software/`.

Eight label-free CPU requests (presentation and IC50, both classes, both checkpoints)
exactly reproduce the prior saved scores: maximum absolute output difference zero
(`results/checkpoint-parity.json`). The retained checkpoint remains
`1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68`; the provisional
junction candidate remains
`e4898af3a7a442a999177273687ea4e4ce5d4216e908f2e435b2d73226e56963`.
No architecture/default-checkpoint promotion or new biological loss weights apply.

Historical preservation verifies 3,096 original tracked files across three
families, including round-trip restoration and refusal to overwrite changed files.
The original bytes are in each family's checksummed history archive; readable
choices/results remain alongside it. Regression fixtures remain plain source.

## Validation limits and decision

TEST outcomes were never opened. No new measured-assay regression, thresholded
binding, presentation AUROC, or intervention-effect estimate is claimed: this
family performs source preparation and software verification, not a biological
fit/evaluation. The last retained checkpoint's published development metrics remain
attached to its original cohort. Isolated gene/drug effects and absolute abundance
remain unvalidated.

The isolated Modal image passes `pip check`. The shared laptop environment retains
pre-existing Varcode upper-bound conflicts in Varlens/Topiary/Vaxrank; editable
checkouts and local dependency edits are preserved. See `results/environment.json`.
Final regression, package, lint and documentation results are recorded in
`results/verification.json` at PR closure.
