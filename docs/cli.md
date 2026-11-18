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
- does not write per-assay CSVs unless requested.
- prints per-file/per-source/per-assay load counts and dedup retention stats (with `tqdm` progress bars).

Useful flags:
- `--per-assay-csv`: enable per-assay CSV exports,
- `--assay-outdir <dir>`: choose per-assay CSV output directory (used with `--per-assay-csv`),
- `--types binding tcell tcr`: restrict merged record types,
- `--quiet`: suppress verbose merge progress/stats,
- `--json`: emit machine-readable merge statistics.

`data mhc-index mouse-overlay`:
- builds `./data/ipd_mhc/mouse_uniprot_overlay.csv` and `.fasta`,
- uses IMGT mouse MHC nomenclature for gene targets and UniProt for protein sequences,
- records explicit provenance (`imgt_source_url`, `uniprot_gene_query`, `uniprot_accession`, `uniprot_record_url`, `allele_derivation_rule`),
- then run `data mhc-index refresh` to include overlay sequences in `mhc_index.csv`.

## Training

All new training uses one manifest:

```bash
python -m presto train --manifest /absolute/path/manifest.json --preflight
python -m presto train --manifest /absolute/path/manifest.json \
  --run-dir experiments/YYYY-MM-DD_HHMM_codex_description --device cpu
```

The manifest supplies model/optimizer settings and hash-pinned observation files.
Use `--resume` with the same run directory and frozen manifest for continuation.
The optional `unified`, `synthetic`, `pretrain` and `finetune` names are aliases;
they do not select different code or generate inputs. Old training flags are
rejected. See the [manifest schema](modular_model.md) and
[migration guide](unified_training.md).

## Shared-model prediction and evaluation

```bash
presto predict observations --checkpoint selected.pt --inputs queries.jsonl --output predictions.json
presto evaluate observations --checkpoint selected.pt --manifest training.json --output-dir evaluation
```

Python callers use `presto.inference.ObservationPredictor(checkpoint).predict(observations)`.
The canonical model format is shared-v7; historical formats dispatch to their
versioned readers. Prediction requests must be label-free and declare `split: predict`.
Evaluation selects explicitly declared source partitions before loading payloads.
See [the current workflow](current_workflow.md) for support guards, APC projection,
checkpoint hashes, output interpretation and Modal execution.

## Pinned presentation data

```bash
presto data verify-corpus --corpus-dir /path/to/corrected-v4
presto data export-presentation --corpus-dir /path/to/corrected-v4 --output-dir exported
presto data freeze-identities --manifest identity-sources.json --output-dir protected
presto data prepare-presentation --manifest preparation.json --output-dir prepared
```

See [the data contract](presentation_data.md). Corpus verification does not assign
training membership. Neither the loader version nor export timestamp substitutes
for a corpus generation pin.

## Historical default-model prediction

These commands read historical default-model checkpoints, not current shared-model files.

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

`predict recognition` reports repertoire-level recognition and immunogenicity.
It does not accept a TCR sequence or perform receptor-specific matching.

`predict presentation` accepts `--flank-n-is-terminus` and
`--flank-c-is-terminus` for observed protein boundaries, including short flanks
that reach a protein end. Omit these for genuinely unknown context. Protein
tiling derives the flags from its supplied complete sequence and coordinates.
The full role-specific biological context API remains pending; see the
[model I/O contract](model_io_contract.md) for actual inputs, output units and
the difference between implemented and desired behavior.

## Historical default-model evaluation

```bash
python -m presto evaluate synthetic --checkpoint presto.pt
```

## Weights

```bash
python -m presto weights list --registry ./weights.json
python -m presto weights download --name foundation-v1 --registry ./weights.json
```
