# CLI

The PRESTO CLI follows the OpenVax-style subcommand layout.

## Data

```bash
python -m presto data list
python -m presto data download --dataset iedb_mhc_ligand --agree-iedb-terms
python -m presto data process --datadir ./data --outdir ./data/processed
```

## Training (synthetic demos)

```bash
python -m presto train synthetic --epochs 5 --batch_size 16 --checkpoint presto.pt
python -m presto train curriculum --epochs 10 --batch_size 32
```

## Prediction

```bash
python -m presto predict presentation \
  --checkpoint presto.pt \
  --peptide SIINFEKL \
  --allele HLA-A*02:01

python -m presto predict recognition \
  --checkpoint presto.pt \
  --peptide SIINFEKL \
  --allele HLA-A*02:01 \
  --tcr-alpha CAVRDSSYKLIF \
  --tcr-beta CASSIRSSYEQYF
```

## Evaluation (synthetic)

```bash
python -m presto evaluate synthetic --checkpoint presto.pt
```
