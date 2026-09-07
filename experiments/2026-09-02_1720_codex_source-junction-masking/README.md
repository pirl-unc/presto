# Source-junction ambiguity masking

> **Validity erratum (2026-09-04):** Retain these files as the exact historical
> record, but do not use this family to choose a mapping policy. Nullable
> unmapped flanks were converted through `str(float("nan")).upper()` into the
> valid-looking sequence `NAN` in the legacy arm, while masking cleared them.
> The comparison therefore changed 6,212 unmapped binding rows for an
> accidental normalization reason in addition to the intended ambiguous
> junctions. Sampling also followed the model seed, sparse targets lacked
> held-out support, and elution was all-positive. The 2026-09-04 remediation
> family supersedes the decision after an exact mandatory preflight.

**Experiment id:** `2026-09-02_1720_codex_source-junction-masking`

**Run dates:** 2026-09-02--2026-09-03

**Agent/model:** Codex / GPT-5

**Git state:** `a1cdcc0d27d63a7440188f98f0dc503c596c8dfd`, branch `main`, dirty
**Status:** complete

## Question and decision

This paired experiment asks whether genuinely unresolved source-protein
junctions should retain the historical deterministic candidate or use explicit
unknown flank context (`?/?`), and whether the effect is large enough to
justify candidate-junction marginalization.

**Decision: use `mask_unresolved` in production and defer candidate-junction
marginalization.** Masking is the honest representation of unknown source
context and produced no material overall regression, but it did not meet the
predeclared marginalization gate. On the exact-only unresolved test union, the
mean masked-minus-legacy change was **-0.0138 Spearman** and **+0.0129
log10(nM) RMSE**; neither direction was consistent across the three seeds.

The negative result is informative but not a claim that flanks never matter:
the unresolved exact test strata were small (21--42 rows per seed). It says
that the present HLA-A*02:01 data do not justify the complexity and trunk cost
of candidate marginalization.

## Dataset and curation contract

- Source: frozen hitlist 1.55.8 curated indexes, loaded through the hitlist-only
  Presto path. The rebuilt `peptide_mappings.parquet` has 5,880,924 rows,
  `gene_biotype`, and artifact contract v2.
- Allele: `HLA-A*02:01`; peptide length 7--30 aa.
- Per seed after curation: 16,721 binding, 20,000 elution, 1,000 stability,
  and 2 kinetics records (37,723 total). Binding was uncapped; capped modalities
  used reservoir sampling with sampling seed `training_seed + 17`.
- Split: peptide-disjoint 80/10/10 train/validation/test. Every observation for
  one peptide stays in one partition.
- Binding curation: all qualifying numeric affinity rows routed from hitlist's
  controlled `response_measured` vocabulary. Inequalities are retained with
  their censor qualifier. Exact regression uses only `qualifier == 0`;
  `<=500 nM` metrics interpret inequalities in the direction that is known.
- Included assay families: affinity (IC50, direct KD/KD proxies, and EC50),
  half-life/Tm stability, kon/koff kinetics, and positive MS/elution evidence.
  No processing, T-cell, TCR, merged-TSV, bulk-proteomics, RNA-expression, or
  synthetic observations were included.
- MS/elution has no real negative class in this contract, so it is trained but
  is not treated as a biological held-out discrimination result.
- Mapping strata are `single`, `flanks_agree`, `within_gene_canonical`,
  `cross_gene_unresolved`, `within_gene_unresolved`, and `unmapped`. The two
  policies have identical targets, qualifiers, categories, and split rows;
  all six validation/test parity checks passed exactly.

Frozen artifact hashes:

| Artifact | SHA256 |
|---|---|
| `observations.parquet` | `f51440ab229fd187d2548b4dddcd1fc04580d97d45fb4d5b8e0222aa8080f928` |
| `binding.parquet` | `fbcae6f762f43edb4eb87b1a7c7f3849757859204d37f3ce82d1f83c23cece9f` |
| `peptide_mappings.parquet` | `45580c16649daf75b11b51497d9d96dfaa987cdcf1197d6449deaa9261f6ec5c` |
| `observations_meta.json` | `ac459e184fc6c54c73f7f1b4fc7dff424b2360b13f55f12bb68a3bbb743de118` |
| `peptide_mappings_meta.json` | `9a93de21753029ac08f1ba05e1a667ee6d7fd1b63a885bdc8dade1e626c7cbbb` |
| `data/mhc_index.csv` | `497938937f01394aeb18a3db15314f04ac1be162efe2844a1f018bcaff121063` |

## Conditions

Each condition used seeds 42, 43, and 44.

| Condition | Junction input policy |
|---|---|
| `legacy_global_canonical` | Global canonical preference followed by deterministic candidate selection, including across genes. |
| `mask_unresolved` | Preserve single/agreed flanks and a unique canonical source only within one gene; encode cross-gene and residual within-gene disagreement as `?/?` and clear position/terminus evidence. |

The policies deliberately do not change assay supervision or target weights.

## Training contract

- Unified Presto trainer; expanded topology, `d_model=128`, 2 layers, 4 heads.
- 10 epochs, batch size 256, ordinary seeded shuffle, AdamW,
  `lr=2.8e-4`, weight decay 0.01, AMP/bfloat16, no `torch.compile`.
- No pretraining checkpoint.
- Synthetic-data contract: pMHC, class-I-no-B2M, processing-negative, and MHC
  augmentation counts are zero. The default UniProt ratio remained 0.1, but
  the intentionally empty trainer data directory contained no
  `uniprot/proteins.tsv`, so zero UniProt or other artificial observations
  entered the dataset (confirmed by the 37,723-row audit).
- Supervised losses use task-mean aggregation with learned uncertainty
  weighting. Relevant observed-label mappings are: affinity -> censor-aware
  normalized `KD_nM` plus the assay-specific `KD_nM`, `IC50_nM`, or `EC50_nM`
  output; half-life/Tm -> censor-aware `t_half`/`Tm`; kinetics -> MSE
  `kon`/`koff`; MS detection -> BCE `elution_logit` and
  `presentation_logit`. Ordinary task base weight is 1.0; MHC class/species/fine
  type auxiliary losses use 0.1. The configured consistency weights were
  cascade 0.2, assay-affinity 0.1, assay-presentation 0.1, no-B2M 0.5,
  T-cell-context 0.05, and T-cell-upstream 0.2, identically in all arms.
- Requested Modal GPU: `H100!`. Every arm observed
  `NVIDIA H100 80GB HBM3, 81559 MiB` with driver 580.95.05. Peak allocated
  memory was 31.63--32.00 GiB and peak reserved memory was 53.05--64.28 GiB.
  Runtime was 4,870 s/arm on average for legacy and 4,828 s/arm for masked
  (29,095 s total GPU runtime).

## Held-out results

Values are mean +/- sample standard deviation over three paired seeds. Affinity
RMSE is in `log10(nM)`.

### Overall binding

| Split | Policy | Exact n/seed | Spearman | Pearson | RMSE | Bal. acc. <=500 nM | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | legacy | 1320 +/- 27 | 0.7523 +/- 0.0145 | 0.7401 +/- 0.0138 | 0.9969 +/- 0.0157 | 0.8439 +/- 0.0164 | 0.9157 +/- 0.0083 | 0.8594 +/- 0.0128 |
| Validation | masked | 1320 +/- 27 | 0.7554 +/- 0.0163 | 0.7455 +/- 0.0160 | 0.9824 +/- 0.0168 | 0.8481 +/- 0.0086 | 0.9170 +/- 0.0110 | 0.8590 +/- 0.0171 |
| Test | legacy | 1384 +/- 50 | 0.7643 +/- 0.0091 | 0.7542 +/- 0.0107 | 0.9873 +/- 0.0293 | 0.8584 +/- 0.0052 | 0.9186 +/- 0.0050 | 0.8607 +/- 0.0117 |
| Test | masked | 1384 +/- 50 | 0.7673 +/- 0.0131 | 0.7571 +/- 0.0157 | 0.9813 +/- 0.0236 | 0.8576 +/- 0.0072 | 0.9186 +/- 0.0078 | 0.8631 +/- 0.0151 |

The remaining test `<=500 nM` means (legacy -> masked) were accuracy
0.8593 -> 0.8583, precision 0.8168 -> 0.8135, recall 0.8524 -> 0.8564,
and F1 0.8341 -> 0.8341.

### Test exact regression by mapping stratum

| Stratum | Exact n/seed | Legacy rho / RMSE | Masked rho / RMSE |
|---|---:|---:|---:|
| `single` | 693.3 | 0.7990 / 0.8398 | 0.7992 / 0.8371 |
| `flanks_agree` | 205.3 | 0.5240 / 1.2475 | 0.5468 / 1.2305 |
| `within_gene_canonical` | 10.3 | 0.6010 / 0.8717 | 0.5569 / 0.8514 |
| `cross_gene_unresolved` | 23.3 | 0.7291 / 1.0198 | 0.7207 / 1.0232 |
| `within_gene_unresolved` | 5.7 | 0.2520 / 0.8036 | 0.2520 / 0.8489 |
| unresolved union | 29.0 | 0.7608 / 0.9853 | 0.7470 / 0.9982 |

Full validation and test tables include all-row and exact-only regression plus
accuracy, balanced accuracy, precision, recall, F1, AUROC, and AUPRC for every
stratum in
[`results/condition_metric_summary.csv`](results/condition_metric_summary.csv).
Per-seed metrics and paired deltas are in
[`results/condition_metrics.csv`](results/condition_metrics.csv) and
[`results/paired_differences.csv`](results/paired_differences.csv).

### Predeclared decision gate

The primary scope was test binding, exact-only, unresolved union. Candidate
marginalization required the same direction in all three seeds and either mean
Spearman improvement >=0.01 or mean RMSE reduction >=0.02, without an overall
regression.

| Metric | Seed 42 delta | Seed 43 delta | Seed 44 delta | Mean delta |
|---|---:|---:|---:|---:|
| Spearman, masked - legacy | -0.0532 | +0.0156 | -0.0037 | **-0.0138** |
| RMSE, masked - legacy | +0.0723 | +0.0159 | -0.0495 | **+0.0129** |

The direction was inconsistent and both means missed the improvement gate.
Overall exact test performance was neutral/slightly favorable to masking:
Spearman +0.0030 and RMSE -0.0059, safely inside the frozen material-regression
bounds (-0.01 Spearman or +0.02 RMSE).

Validation agreed in direction on the unresolved union (masked-minus-legacy
Spearman -0.1063; RMSE +0.0730), although that subset was also small (17--30
exact rows per seed).

## Artifacts and reproduction

- Exact launcher: [`reproduce/launch.sh`](reproduce/launch.sh); invocation and
  dirty git state: [`reproduce/launch.json`](reproduce/launch.json); frozen
  source: [`reproduce/source/launch.py`](reproduce/source/launch.py).
- Six run bundles under `results/runs/` preserve config, summaries, full metric
  tables, and per-example validation/test prediction dumps. Local checkpoints
  are gitignored to avoid adding about 400 MB to the repository; they remain
  recoverable from the Modal paths in `manifest.json`.
- Contract validation: [`results/contract_checks.json`](results/contract_checks.json);
  exact paired row parity: [`results/data_parity.json`](results/data_parity.json);
  ingest/split audit: [`results/data_audit.json`](results/data_audit.json).
- Machine-readable decision:
  [`results/paired_differences.json`](results/paired_differences.json).
- Raw run artifacts remain on the Modal volume at the six
  `/checkpoints/presto-source-junction-mask-20260902b-*` paths recorded in
  [`manifest.json`](manifest.json).

An earlier `20260902a` family was stopped before results were read after review
found that selected-`X` filtering happened after policy transformation and
could change row membership. The corrected `20260902b` family filters the
shared deterministic selection first; aggregation rejects any loss of paired
supervision-row parity.

`legacy_global_canonical` preserves the old semantic error--feeding one
unresolved candidate junction as fact--but is not a byte-for-byte replay of
the former bulk fallback, which depended on input frame order because
`protein_id` was not projected. The experiment uses a stable candidate order
so the comparison itself is reproducible.
