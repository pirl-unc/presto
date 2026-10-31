# MHC reference annotation admission

Registered before preparation, 2026-10-21 UTC; Codex / GPT-6. Data-only stage;
no pretraining, optimizer, predictions or model selection. **Closed: source replay
and mechanical readiness pass; the partition has substantial group concentration.**

The [prospective contract](../agents/codex/plans/mhc-reference-admission.md)
freezes reference bytes, four exact taxonomy mappings, source/sequence quality
checks, conflict/alias handling, deterministic connected-component splits and
readiness checks. Single-groove species labels train the existing shared readout;
class/pairing and assay targets are not inferred from missing partners.

All raw rows must have dispositions and all admitted domain labels must replay
against their source aliases. Source grouping is not homology qualification.
No validation/test predictive metrics apply because no model is evaluated;
the prepared partitions are retained for a separately registered future recipe.

## Source and execution contract

Executed clean commit `1b218e7f77337398889bffdefa37309597dfa6bc` on local CPU,
Python 3.12.6, one Torch/BLAS thread; no GPU or Modal requested or observed.
The pinned existing `/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv` has SHA256
`ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`.
Its producer version is not recorded; the runtime package versions in the receipt
must not be read as the versions that produced the source.

The exact frozen taxa are `Homo sapiens`, `Mus musculus (Mouse)`,
`Rattus norvegicus (Norway rat)` and `Macaca mulatta`, mapped respectively to
human, mouse, rat and rhesus macaque. Each used domain has 60–130 canonical
residues, agrees with its declared length and occurs in the original protein.
Eligible records have the required source identifiers and functional/quality
flags. Class I contributes both original, ordered, nonoverlapping domains;
class II contributes its actual alpha or beta groove. The opposite class-II slot
stays empty. No absent partner, pair compatibility or class target is synthesized.
Nonclassical genes can remain; source annotation does not prove peptide presentation.

One annotation is retained per unique slot/sequence/species, with all original
aliases. Conflicting species for an identical sequence would defer every affected
original chain; none occurred. Exact shared sequences and original-protein parent
links form connected components. Sorted component sequence keys determine their
hashes; SHA256 with salt `presto-mhc-reference-v1` assigns whole components at
70/15/15 boundaries. The salt and membership were not changed after inspection.

The exact invocation and environment are in
[executed-launch.json](reproduce/executed-launch.json), with the executed launcher
and relevant application source under [reproduce/source](reproduce/source/).
From the frozen checkout, `bash reproduce/launch.sh FRESH_OUTPUT FRESH_COHORT`
(using this family's full script path) reproduces preparation into fresh paths.
The recorded `PYTHONPATH=/private/tmp/presto-modular-imports` resolves the isolated
checkout; the Python executable is `/Users/iskander/code/shared-virtual-env/bin/python`.
All three thread overrides are 1. Runtime was **4.4094 seconds**, peak process
RSS **641,744,896 bytes**, excluding the separate independent replay.

## Admission and conservation

| Raw-row disposition | Rows |
| --- | ---: |
| Admitted source, with at least one new annotation | 12,612 |
| Duplicate alias, all domains already represented | 7,211 |
| Excluded | 36,617 |
| Total | 56,440 |

Excluded rows comprise 27,234 outside the frozen taxa, 679 failing functional
flags, 8,702 failing groove status/flags, one unsupported class/chain assignment
and one noncanonical/out-of-scope domain. There are **13,218 unique annotations**,
**33,696 source-domain aliases**, **3,687 connected components** and zero species
conflicts. These are different denominators: original chains may supply two
domains, and multiple chains may alias the same domain annotation.

The [independent verifier](analysis/verify.py) imports none of the preparation
functions. It replays all 33,696 aliases from the original CSV, including sequence,
slot, taxonomy, gene, source identifiers, quality flags and protein coordinate;
conserves every raw-row disposition and admitted original-chain domain; and
recomputes component hashes and split assignments. It also verifies the source,
manifest and normalized observation hashes. [Replay passed](results/independent-replay.json).
Canonical source loading/preflight passes for the species-only manifest. No model
forward or optimizer step was used as part of admission.

| Species | Train annotations / components | Validation annotations / components | Test annotations / components |
| --- | ---: | ---: | ---: |
| Human | 4,848 / 2,060 | 419 / 416 | 6,160 / 430 |
| Mouse | 142 / 80 | 34 / 19 | 42 / 19 |
| Rat | 131 / 70 | 35 / 17 | 37 / 18 |
| Rhesus macaque | 971 / 389 | 206 / 92 | 193 / 77 |
| Total | 6,092 / 2,599 | 694 / 544 | 6,432 / 544 |

The prospective mechanical minimum (20 annotations and three components for
every species/split) passes without adjustment. This is not a statistical or
biological qualification gate. The largest component contains 5,664 human B/C
domain annotations, **91.95% of human test and 88.06% of all test annotations**.
Another human A component contains 2,623 annotations, 54.10% of human training.
The size-concentration statistic `(sum n)^2 / sum(n^2)` is only 1.18 for human
test despite 430 nominal components; this describes group concentration, not an
estimated effective biological sample size. The [concentration table](results/split-concentration.csv)
and top-component genes are retained. Actual row fractions therefore differ
substantially from the nominal component-assignment fractions. No reseeding,
rebalancing or claim of homology independence follows.

## Artifacts and decision

- [Admission receipt](results/admission.json): source/disposition counts, all
  source/class/slot/species/split alias counts, runtime, hashes and readiness.
- [Every raw-row disposition](results/source-dispositions.jsonl.gz),
  [annotation aliases](results/annotations.jsonl.gz) and
  [conflict records](results/species-conflicts.jsonl.gz).
- [Canonical preflight](results/preflight.json),
  [admission log](launch_logs/admission.log) and
  [independent replay log](launch_logs/independent-replay.log).
- Generated cohort: `data/modular_cohorts/2026-10-21-mhc-reference/`;
  manifest SHA256 `4e4d4a5d2b3d47fc16ddf63ddeeaf6111836138661356ab6805945f13c7ef06c`,
  observation SHA256 `5ba2ef6f40b047ddc256c61fb06e802c82bc2a400830a178d44a0a82455ef510`.
  The manifest is tracked; the generated JSONL is local and reproducible from
  the frozen source/recipe. It is not duplicated in the experiment directory.

There is no trained condition, winner, checkpoint, loss weight or predictive
metric in this data-only family. All affinity/stability/processing/MS/TCR assays
and censoring are outside this annotation preparation; no synthetic sequences or
labels are generated. Validation/test partitions exist, but prediction dumps,
loss, discrimination and IC50/500 nM metrics are inapplicable because no model
was evaluated. Future fitting needs its own frozen baseline/selection/evaluation
recipe and must address the observed group concentration explicitly.

This supplies source-traceable single-domain species supervision. It supplies
neither observed class-II pairs nor evidence that species pretraining improves
affinity. Real paired annotations, calibration/OOD checks, controlled transfer and
the wider APC/MS/TCR model remain unfinished.
