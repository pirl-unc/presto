# Presto

Unified immunoinformatics for pMHC presentation and T-cell recognition.

## Installation

Install a wheel from this repository's [GitHub releases](https://github.com/pirl-unc/presto/releases).
The PyPI name `presto` belongs to the unrelated Immcantation repertoire toolkit;
use a separate environment for these projects. For the 0.1.1 release:

```bash
python -m pip install https://github.com/pirl-unc/presto/releases/download/v0.1.1/presto-0.1.1-py3-none-any.whl
```

Release artifacts are built after CI runs `./lint.sh` and `./test.sh`. CI installs
the wheel outside the checkout to verify packaged B2M resources and the bulk
loader's missing-label masks. Publish the wheel and source archive from the
successful merged-main run, together with their SHA-256 hashes and source commit.
The source archive builds the tested wheel; no training datasets or model weights
are included. A software release does not certify the training data or model quality.

CI cancels superseded updates within the same pull request or feature-branch push.
PR and push checks have separate groups, and both final-head validations gate merge.
Each main-push and immutable-tag run has its own protected group, so neither running
nor pending release validation can be displaced by a later update. Distribution
still requires successful checks on the exact merged-main commit.

## Development quickstart

The released 0.1.1 interface predates this development architecture. New training
from this checkout uses one model and one observation manifest:

```bash
python -m presto --help
python -m presto train --manifest /absolute/path/manifest.json --preflight
python -m presto train --manifest /absolute/path/manifest.json \
  --run-dir experiments/YYYY-MM-DD_HHMM_codex_description --device cpu
```

Model, optimization and hash-pinned observation sources are configured in the
manifest. Pretraining, binding, MS, paired TCR and joint fits all use this path.
Historical checkpoint readers and frozen experiment records remain available.

## Canonical docs

- [Shared architecture and observation contract](docs/modular_model.md)
- [Single training interface and migration](docs/unified_training.md)
- [CLI usage](docs/cli.md)
- [Study curation contract](docs/study_curation_contract.md)
- [Repo inventory and retention](docs/repo_inventory.md)
- Experiment registration: `experiments/EXPERIMENT_WORKFLOW.md`

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

The shared model accepts explicit MHC and paired receptor sequences with scoped
cellular/MS context. Output availability and shared parameters do not establish
scientific validity. The new full-flank architecture has no qualified checkpoint;
see the [capability limits](docs/unified_training.md).

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

## Synthetic data

New training does not generate implicit negatives. Synthetic observations must
be explicitly prepared with provenance and label policy in the same manifest
schema. Such labels remain hypotheses or software fixtures, not measurements.
Historical synthetic defaults belong to the frozen legacy implementation.

## Development

```bash
./develop.sh
./lint.sh          # ruff check + ruff format --check
ruff format .      # apply formatting (lint.sh only *checks* it)
./test.sh
```

Verified bulk proteomics: see [scoped inputs](docs/scoped_proteomics.md) for the
PXD013455 Python adapter, frozen source hashes, evidence classes and limitations.
