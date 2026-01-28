# PRESTO

Unified immunoinformatics for MHC presentation and TCR recognition.

## Quickstart

```bash
python -m presto --help
python -m presto data list
```

## Common commands

```bash
python -m presto data download --dataset iedb_mhc_ligand --agree-iedb-terms
python -m presto train synthetic --epochs 5 --checkpoint presto.pt
python -m presto predict presentation --checkpoint presto.pt --peptide SIINFEKL --allele HLA-A*02:01
```

## Development

```bash
./develop.sh
./lint.sh
./test.sh
```

If you use `uv`, you can also run:

```bash
uv run presto --help
```
