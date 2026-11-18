# Pinned presentation data

## Three distinct identities

1. **Loader:** Hitlist 2.1.12, including current annotation/filter behavior.
2. **Corpus:** corrected complete corpus v4, produced by Hitlist 1.78.0. Its native
   `observations_meta.json` SHA-256 is
   `2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7`.
3. **Consumer cohort:** hashed exported observations plus explicit admission,
   reference, ancestry, held-out identity and split contracts.

The packaged `data/hitlist_corpus.json` pins (1) and (2). `verify-corpus` validates
every manifest-listed file. `export-presentation` uses public loaders in an
isolated process, verifies the selected index path and leaves original index
bytes unchanged. It never changes the user's default Hitlist data directory.

The first corrected export contains 2,842,058 class-I rows across 354 studies and
1,408,813 class-II rows across 141 studies. These are observations before model
admission, not independent experiments or training-pair counts. Reported, derived
and donor-set serotype roles remain distinct. Predicted restriction evidence does
not become an exact presenting-allele label. No export silently inherits old splits.

## Preparation manifest

`presto data prepare-presentation --manifest preparation.json --output-dir prepared`
uses schema `presto.presentation-preparation.v1`. It requires:

| Field | Meaning |
| --- | --- |
| `files.class_I`, `files.class_II` | Public-loader Parquet exports |
| `files.samples` | Curated sample CSV |
| `files.catalog` | Full MHC sequence catalog CSV |
| `files.proteins` | Explicit GENCODE protein/transcript/gene FASTA, optionally gzip |
| `files.observed` | Parquet peptide identity universe for background exclusion |
| `identities` | Peptide-only files with `sealed`, `fixed_validation` or `ancestry` role |
| `ancestry_complete` | Explicit declaration that the listed model ancestry is complete |
| `parent_checkpoint_sha256` | Parent identity for warm-start preparation; requires ancestry identities |
| `identity_source_manifest` | Optional pinned recipe for deriving TRAIN/VALIDATION identity files |
| `source_exclusions` | Explicit source/class/condition exclusions with reasons |

Every file entry contains `path` and `sha256`. Relative paths resolve against the
manifest directory. Identity files contain one canonical peptide per line, never
outcome tables. Sealed identity protection must be present. The preparation checks
hashes before admission and writes the complete resolved contract in its receipt.
The ancestry-completeness declaration must be supported by the checkpoint lineage;
a boolean is not independent proof of complete provenance.

The optional identity-source schema is `presto.presentation-identities.v1`. Each
source declares `path`, `sha256`, `role`, and `split`. Only TRAIN ancestry and
VALIDATION protection are accepted. `kind: candidate_options` includes alternative
validation background identities; ordinary sources use `kind: observations`.
The command never accepts TEST outcome sources. This facility extracts identities
from already admitted data; it does not reclassify source observations.

## Admission and leakage controls

Both classes use the same MS adapter. Native positives remain source observations;
unspecified modifications are explicitly sequence-only unknown PTMs. Known modified
peptidoforms and invalid/ambiguous MHC assignments follow the adapter's disposition
policy. Unknown APC state does not remove otherwise usable presentation evidence.
Duplicate original identities have one vote; conflicting identities are excluded.

All exact reference occurrences contribute to protein-family grouping. Unanimous
flanks are retained; conflicting source boundaries stay unknown. Generated
backgrounds have the same peptide length and a possible source protein, exclude
known observed peptides and related protected sequences, and carry explicit
generation provenance. These are sampled backgrounds, not measured non-binders.

New validation excludes historical training families and exact/shared-8mer/
Hamming-distance-one ancestry exposures. TRAIN excludes fixed/new validation
families and related protected identities. Validation backgrounds receive the
same protection. Declared replay TRAIN pairs are excluded together; existing
validation stays unchanged. The receipt preserves exclusions and per-study/class/
condition counts. No row or study cap is imposed.

The output `manifest.json` lists sources with hashes and explicit splits. It is a
source manifest; add the chosen model and training controls before fitting.
`--admission-only` produces a source audit, not a training-qualified cohort.

## Modal execution

The shared launcher runs the installed wheel with a JSON request:

```bash
PRESTO_MODAL_WHEEL=dist/presto-VERSION-py3-none-any.whl \
  modal run --detach -m presto.scripts.modal_workflow --request request.json
```

A request declares `commands` (Presto argument lists), `receipt_dir` under `/volume`,
optional `assets` (`local`, `remote`, `sha256`), and `gpu` (default false).
`PRESTO_MODAL_VOLUME` selects an existing volume, default `presto-checkpoints`.
GPU requests explicitly use `PRESTO_MODAL_GPU`, default `H100!`; CPU preparation
allocates 8 CPUs/64 GiB without a GPU. No arbitrary shell command is constructed.
Uploads verify hashes, refuse overwrite, and remote execution rechecks every asset
and the wheel. A spawned persistent function owns the detached work. Use a new
output namespace for retries; retain failed receipts and stop surviving workers
before relaunching mutating work. Register requests under `experiments/` before launch.

## Corrected-corpus integration result

The complete retained-checkpoint ancestry embargo leaves **161,573 class-I** and
**134,127 class-II** training pairs (156 and 42 studies respectively). Only two
new class-I validation pairs survive; no new class-II validation pairs survive.
More exported observations therefore do not imply more independent validation.
Use the existing fixed development-validation cohorts for class-balanced selection;
do not use this tiny new partition alone or weaken ancestry protection to enlarge it.

The integration record includes `results/training-sources.json`, composing the
corrected training population with immutable measured-affinity replay and existing
fixed validation sources. It is a source composition, not a newly trained model.
Previous checkpoint performance cannot be attributed to this corrected cohort.

Recheck prepared files independently with:

```bash
presto data audit-presentation --prepared-dir prepared --output pair-audit.json
```

The audit checks file hashes, complete pairs, matching APC/MHC context, source
protein windows, observed-peptide exclusions, distinct identities and related
held-out/ancestry protection. TEST contributes peptide identities only.

Modal requests may declare a `workspace` inside `/volume`; by default it is the
receipt directory's parent. Training uses that workspace's experiment registry.
