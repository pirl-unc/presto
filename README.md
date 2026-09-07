# Presto

Unified immunoinformatics for pMHC presentation and T-cell recognition.

## Quickstart

```bash
python -m presto --help
python -m presto data list
python -m presto train unified --data-dir ./data --epochs 5 --checkpoint presto.pt
python -m presto predict presentation --checkpoint presto.pt --peptide SIINFEKL --allele HLA-A*02:01
```

## Canonical Docs

- Single authoritative input/output design: [model I/O contract](docs/model_io_contract.md)
- Current architecture guide: [design](docs/design.md)
- Assay policy summary: [assay modeling](docs/assay_modeling_contract.md)
- Assay source inventory and supervision map: `docs/assay_learning_scheme.md`
- Training and batch construction spec: `docs/training_spec.md`
- CLI usage: `docs/cli.md`
- Repo inventory and retention guide: [repo inventory](docs/repo_inventory.md)
- Implementation status audit: `TODO.md`

## Repo At A Glance

What is intentionally committed:
- source code, tests, and canonical docs
- compact reusable reference data under `data/`
- experiment history under `experiments/`

What is intentionally local-only and ignored:
- `artifacts/`
- `modal_runs/`
- large regenerable derived datasets such as `data/merged_deduped.tsv`
- transient launch logs and caches

## Canonical Assay Rule

Presto uses sequence-only encoding with scoped downstream biological context.
APC state, interventions and cytokines are allowed biological inputs. Presenting
MHC and repertoire-selection MHC, and APC/MHC/TCR/repertoire/antigen species,
are distinct roles. Assay identity only selects fixed output tracks and losses.

The full role-specific schema is not implemented yet. The
[I/O contract](docs/model_io_contract.md) separates target design, actual outputs
and remaining gaps; [issue #46](https://github.com/pirl-unc/presto/issues/46)
tracks implementation. Receptor evidence is currently pMHC-only, not matching
against a supplied TCR.

## Mouse MHC Overlay (IMGT + UniProt, Provenance Tracked)

Build a mouse MHC sequence overlay into `data/ipd_mhc/`:

```bash
python -m presto data mhc-index mouse-overlay --datadir ./data
python -m presto data mhc-index refresh --datadir ./data
```

Outputs:
- `data/ipd_mhc/mouse_uniprot_overlay.csv`: per-protein provenance catalog
- `data/ipd_mhc/mouse_uniprot_overlay.fasta`: selected allele-sequence overlay

The catalog includes explicit source columns per emitted protein:
- `imgt_source_url`
- `uniprot_gene_query`
- `uniprot_accession`
- `uniprot_record_url`
- `allele_derivation_rule`

## Synthetic Negatives (Default: Enabled)

Canonical unified training defaults enable all synthetic-negative categories.

These are synthetic hypotheses, not measured negatives. In particular,
`no_mhc_beta` currently deletes the second groove segment (class-I alpha2),
not beta2m. Its assembly interpretation and missing-sequence priors are known
design drift tracked in #46, not scientifically validated defaults.

| Category | Modes | Default control (unified) | Primary target effect |
|---|---|---|---|
| pMHC negatives | `peptide_scramble`, `peptide_random`, `mhc_scramble`, `mhc_random`, `no_mhc_alpha`, `no_mhc_beta` | `--synthetic-pmhc-negative-ratio 1.0` and `--synthetic-class-i-no-mhc-beta-negative-ratio 0.25` | Drive weak/non-binder supervision (`binding`/`affinity`) and low downstream presentation |
| Elution negatives | `peptide_random_mhc_real`, `peptide_real_mhc_random`, `peptide_random_mhc_random`, plus data-conditional hard-pair negatives | Derived from `--synthetic-pmhc-negative-ratio` (`0.5x` scale) | Drive low elution/presentation for implausible peptide:MHC pairs |
| Processing negatives | `flank_shuffle`, `peptide_scramble` | `--synthetic-processing-negative-ratio 0.5` | Drive low processing probability for corrupted cleavage/context inputs |
| Cascade negatives | binding->elution, binding->tcell projection | Derived from `--synthetic-pmhc-negative-ratio` (`0.5x` each for elution/tcell) | Enforce biological cascade consistency across tasks |

Semantics:
- `random` = de novo generation/sampling.
- `scramble` = permutation of existing sequence content.

Reference: [training guide](docs/training_spec.md).

## Development

```bash
./develop.sh
./lint.sh          # ruff check + ruff format --check
ruff format .      # apply formatting (lint.sh only *checks* it)
./test.sh
```
