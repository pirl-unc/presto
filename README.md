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

- High-level goals, data, inputs/outputs: `docs/design.md`
- Canonical assay modeling contract: `docs/assay_modeling_contract.md`
- Training and batch construction spec: `docs/training_spec.md`
- CLI usage: `docs/cli.md`
- Implementation status audit: `TODO.md`

## Canonical Assay Rule

Canonical Presto is sequence-only on the input side for assay prediction. Assay identity may choose supervision targets or output heads, but must never be fed back in as a predictive input feature for affinity, T-cell assays, mass spec, or related outputs.

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

| Category | Modes | Default control (unified) | Primary target effect |
|---|---|---|---|
| pMHC negatives | `peptide_scramble`, `peptide_random`, `mhc_scramble`, `mhc_random`, `no_mhc_alpha`, `no_mhc_beta` | `--synthetic-pmhc-negative-ratio 1.0` and `--synthetic-class-i-no-mhc-beta-negative-ratio 0.25` | Drive weak/non-binder supervision (`binding`/`affinity`) and low downstream presentation |
| Elution negatives | `peptide_random_mhc_real`, `peptide_real_mhc_random`, `peptide_random_mhc_random`, plus data-conditional hard-pair negatives | Derived from `--synthetic-pmhc-negative-ratio` (`0.5x` scale) | Drive low elution/presentation for implausible peptide:MHC pairs |
| Processing negatives | `flank_shuffle`, `peptide_scramble` | `--synthetic-processing-negative-ratio 0.5` | Drive low processing probability for corrupted cleavage/context inputs |
| Cascade negatives | binding->elution, binding->tcell projection | Derived from `--synthetic-pmhc-negative-ratio` (`0.5x` each for elution/tcell) | Enforce biological cascade consistency across tasks |

Semantics:
- `random` = de novo generation/sampling.
- `scramble` = permutation of existing sequence content.

Reference: `docs/training_spec.md` section "Synthetic Negative Schedule".

## Development

```bash
./develop.sh
./lint.sh
./test.sh
```
