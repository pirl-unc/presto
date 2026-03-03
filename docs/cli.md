# CLI

The Presto CLI uses subcommands for data, training, prediction, evaluation, and weight management.

## Data

```bash
python -m presto data list
python -m presto data download --dataset iedb_mhc_ligand --agree-iedb-terms
python -m presto data process --datadir ./data --outdir ./data/processed
python -m presto data merge --datadir ./data
python -m presto data mhc-index mouse-overlay --datadir ./data
python -m presto data mhc-index refresh --datadir ./data --download-missing
```

`data merge` defaults:
- writes `./data/merged_deduped.tsv`,
- writes per-assay simplified CSVs under `./data/merged_assays/`.
- prints per-file/per-source/per-assay load counts and dedup retention stats (with `tqdm` progress bars).

Useful flags:
- `--assay-outdir <dir>`: choose per-assay CSV output directory,
- `--no-assay-csv`: disable per-assay CSV exports,
- `--types binding tcell tcr`: restrict merged record types,
- `--quiet`: suppress verbose merge progress/stats,
- `--json`: emit machine-readable merge statistics.

`data mhc-index mouse-overlay`:
- builds `./data/ipd_mhc/mouse_uniprot_overlay.csv` and `.fasta`,
- uses IMGT mouse MHC nomenclature for gene targets and UniProt for protein sequences,
- records explicit provenance (`imgt_source_url`, `uniprot_gene_query`, `uniprot_accession`, `uniprot_record_url`, `allele_derivation_rule`),
- then run `data mhc-index refresh` to include overlay sequences in `mhc_index.csv`.

## Training

Canonical production path:

```bash
python -m presto train unified --data-dir ./data --epochs 5 --checkpoint presto.pt
```

Unified training defaults:
- reads merged input from `./data/merged_deduped.tsv` (required by default),
- uses balanced mini-batches (`--balanced-batches`) across assay/source/label/allele/synthetic strata,
- uses `tqdm` with running loss + samples/sec.

Optional synthetic path:

```bash
python -m presto train synthetic --epochs 5 --batch_size 16 --checkpoint presto.pt
```

## Prediction

```bash
python -m presto predict presentation \
  --checkpoint presto.pt \
  --peptide SIINFEKL \
  --allele HLA-A*02:01

python -m presto predict tile \
  --checkpoint presto.pt \
  --protein-sequence MPEPSLLQHLIGLQWERTY \
  --allele HLA-A*02:01 \
  --min-length 8 \
  --max-length 11
```

`predict recognition` is reserved for a future TCR pathway and is currently not
active in canonical inference.

## Evaluation

```bash
python -m presto evaluate synthetic --checkpoint presto.pt
```

## Weights

```bash
python -m presto weights list --registry ./weights.json
python -m presto weights download --name foundation-v1 --registry ./weights.json
```
