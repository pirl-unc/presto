# MHC-II MS transfer-cohort qualification

Status: independently verified source/dependency inventory; predictive fit pending.
Codex / GPT-6, 2026-10-26, registered at Presto
`aa5eb5ed41e2916b8f94c9e4a30e9d069bf3a5fc` with recorded experiment/task dirt.

The 98,220 admitted MS detections contain useful training candidates, but their
allele and condition labels have different evidence. DP/K562 PMID 35251023 is
the best-supported initial training source reviewed here. DQ8 PMID 34433824 is
an external-study candidate with explicit source-pair/origin qualifications.
Atlas named allele associations need prediction-provenance joins before use as
independent measured-restriction validation. No model was fitted in this family.

## Data and policy

[Prospective plan](../agents/codex/plans/2026-10-26_mhc2-ms-transfer-cohort.md).
The verified1825 admission supplies observations and all contributor links under
`/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01/`.
Observation SHA256:
`532977c4a5e9b6530fca36d039b73a3b832c7b15197dc7a6def0bd66191aecef`.
The native human export was produced by Hitlist1.78.0, SHA256
`f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd`.
Current notes are separately pinned at Hitlist2.0.42,
commit `13b91424295e01286db46e8e9818fcc2fe816834`; they do not retroactively
change old rows.

All retained observations are reported positive class-II MS detections with
complete explicit source candidate pairs and source `has_ptm=False`. That does
not establish linear origin, measured restriction, biological negative controls
or isolated processing. No other assay or synthetic outcome is introduced here.
There are 56,853 peptides, 20 contributor studies and 213 joint policy groups.
All 196,304 contributor links were replayed; no retained row spans contributor
studies. Current explicit curation exists for eight of the 20 studies.

| PMID | Rows | Initial predictive decision |
|---|---:|---|
| 35251023, DP/K562 | 39,293 | Candidate training source; retain engineering qualification |
| 33858848, Atlas | 24,399 | Defer named pairs pending assignment-provenance join |
| 25502872, DQ | 17,105 | Full Methods not recovered within retrieval budget |
| 30573663, DO | 5,164 | Full Methods/quantitative qualification remain open |
| 33592498, CIITA | 4,820 | Retain background/ambiguous-arm qualifications; defer initial fit |
| 34433824, DQ8 | 4,659 | Candidate transfer set, with source-pair/origin caveats |

Other studies remain policy-unresolved here. Review selection used admitted counts
before model outcomes. See [primary evidence](results/primary-evidence.md).

## Dependency evidence

Directed overlap counts any shared contiguous nine-residue segment. Nested
peptides make directions differ; this does not establish parent, homology, donor,
specimen or search independence.

| Query → reference PMID | Query peptides | Exact overlap | Any nine-residue overlap |
|---|---:|---:|---:|
| 25502872 → 34433824 | 11,262 | 894 | 3,799 |
| 34433824 → 25502872 | 4,659 | 894 | 2,492 |
| 35251023 → 34433824 | 26,949 | 182 | 1,768 |
| 34433824 → 35251023 | 4,659 | 182 | 1,114 |

Against the IC50 initializer's train/validation/test input partitions, DP has
297/61/25 rows with nine-residue overlap and 9/0/0 exact overlaps. DQ8 has
119/70/29 and 6/3/0, respectively. These overlapping counts cannot be summed as
unique exclusions. The next recipe must exclude exposed evaluation components
and transfer overlap before fitting. Only peptide inputs were inspected here.

## Current-source findings

Atlas replay through current Hitlist leaves all 24,399 named-pair rows at
`restriction_evidence=unknown`. Primary Methods establish a computational
association policy; exact row joins are still required, without blanket relabeling.
[Hitlist855](https://github.com/pirl-unc/hitlist/issues/855) records the gap.
The primary donor-aware loader already exists (#702 completed).

DP's current curation has one generic unperturbed/untreated arm despite defined
paired K562 populations. [Hitlist856](https://github.com/pirl-unc/hitlist/issues/856)
requests source-backed engineering/arm identity; #593 does not cover this study.

All208 CIITA parental rows retain probable-background notes. Among4,612 old
transduced-arm rows,40 literal statements name both parental and treated
conditions (13/26/1 across three lines);4,572 name treated alone. Current curation
already addresses ambiguity. Do not report the old policy as a new missing-note
bug, silently backfill current labels, or infer causal processing from counts.

## Verification and reproduction

| Action | Seconds | Peak RSS bytes | Result |
|---|---:|---:|---|
| Inventory | 6.772 | 390,758,400 | 98,220 rows/213 policy groups |
| Independent replay | 8.717 | 452,820,992 | All joins/400 overlap entries verified |
| Current replay, attempt03 | 3.350 | 172,195,840 | All Atlas rows remain unknown |

Independent verification rereads original JSONL and individual contributor links,
uses direct sets and regex overlapping windows instead of the producer bitmask,
and verifies raw IC50 inputs/current notes and hashes. Local CPU/one thread,
no Modal/GPU/package changes. Limits:300s/2GiB RSS,10GiB RAM/disk floors,
250MiB compact outputs; current replay uses a stricter120s limit.

Current replay attempt01 hit the RAM floor; attempt02 finished scanning but
failed to identify an lru-cached function's source. Attempt03 unwraps it for
hashing; data/classification are unchanged. Both failures and exact sources remain.
Primary retrieval failures and the successful Atlas XML retry are preserved.

`reproduce/launch.sh`, `launch.json`, `closure.json` and `source/` pin invocation,
environment, source and artifacts. Compact source-linked metadata, policy groups,
overlap and exposure tables are under `results/inventory/`. Primary XML remains
at the exact local path/hash in its receipt. No training loss, held-out predictive
metrics or prediction dumps apply: this is an inventory with no predictive split.
They are required in the following MS comparison.

**Next step:** fit qualified detection enrichment against binding-only and sequence
controls with an explicit generated-background estimand, initializer exclusions
and source-transfer limits. The full binding/MS/pathway goal remains active.
