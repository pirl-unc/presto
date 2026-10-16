# Frozen Hitlist inputs

Canonical Hitlist training requires a frozen bundle produced by
`hitlist.write_training_bundle(...)` with current verified indexes. Supply its
directory with `--data-source hitlist --hitlist-bundle /path/to/bundle` and its
published `manifest.json` SHA-256 with `--hitlist-bundle-sha256 DIGEST`.
Disable unfrozen auxiliary sources with `--mhc-augmentation-samples 0
--uniprot-negative-ratio 0`. The canary profile fabricates extra modalities and
is unavailable for frozen Hitlist runs.
The bundle must contain both the training projection needed by the requested
modalities and Hitlist's identities, contributors and reviewed-lineage sidecars.
Use an unprojected export unless you have explicitly checked the column contract
in `presto.data.hitlist_source.training_columns`.

Presto verifies every payload hash, identity relationship and contributor
coverage using Hitlist's verifier, then installs/reuses the complete set through
datacache under `<hitlist data directory>/training-bundles/<manifest sha256>`.
Identical bundles share storage across runs. A damaged cached copy fails with an
error. Frozen reads do not consult live indexes or rebuild input data; archival
bundles created with freshness checks disabled are rejected. Hash verification
establishes byte identity and consistency, not scientific qualification.

The data-funnel receipt records the bundle identity, artifact hashes, creation
options, coverage, applied filters, mapping decisions, caps and label policy.
The local path is for inspection; the manifest digest is the portable identity.
Raw data remain shared in datacache. Run directories hold references and reports.
A remote launcher uses `PRESTO_HITLIST_BUNDLE` and
`PRESTO_HITLIST_BUNDLE_SHA256` for a bundle already installed on the worker;
`PRESTO_HITLIST_BUILD` is no longer a training-time operation.

## Record caps and reproducibility

Canonical training honors `--cap-sampling head|reservoir`. `head` keeps the first
eligible records; `reservoir` uses Python Random Algorithm R. Each assay family
has an independent stream initialized with the recorded source sampling seed.
Positive caps bound retained record objects; full projected input dataframes are
still loaded, and every eligible row is inspected for complete admission counts.
Absent or nonpositive caps remain unlimited. The data funnel records the actual
algorithm, seed, mode and memory boundary.

The Python loader defaults to `cap_sampling="uniform_list"` to preserve existing
`random.sample` selections exactly. Before 0.1.12 the canonical Hitlist route also
used that algorithm, ignoring its configured cap-sampling mode (#103). New
canonical reservoir runs can therefore select different records with the same
seed. Preserve historical frozen membership and record this change when comparing
experiments; do not relabel an old run as Algorithm R.

## Human inspection and lineage

| Artifact or field | Meaning |
| --- | --- |
| `training.parquet` | Frozen model-source table, including assay and cell context |
| `identities.parquet` | One row per observation, with reviewed biological identity status |
| `contributors.parquet` | Source rows and citable references contributing to observations |
| `lineage.json` | Frozen reviewed registry; unresolved identities remain unresolved |
| `source_bundle_id` + `evidence_row_id` | Lookup from measured records, batches and predictions back to the bundle |
| `provenance_id` / `provenance_status` | Contributor-group lookup and evidence availability |
| `source_endpoint`, `source_units`, `source_inequality`, `source_value` | Original exported measurement before Presto target conversion |

Cell types, cell lines, cell conditions, cellular MHC typing, recorded experimental
restriction and candidate allele sets are separate axes. Retain the complete
source projection for inspection. Display labels are not reviewed specimen or
donor IDs. Source identifiers and the receipt are host metadata, never model
features. Protein mapping alternatives collapse to one observation under the
recorded mapping policy; they do not add independent assay measurements.

## Label admission policy, version 1

| Endpoint | Required input units | Record target |
| --- | --- | --- |
| IC50, EC50, KD and recognized KD approximations | `nM` | Same concentration; endpoint retained |
| Half life | `min` | Hours (divide by 60) |
| 50% dissociation temperature | `°C` | Same temperature |
| Off rate | `1/s` | Same rate |
| On rate | `1/Ms` | Same rate |

Quantitative values must be finite and positive, except temperatures may be
zero or negative. Supported censoring is `<`, `<=`, `≤`, `>`, `>=`, `≥`, `=` or
empty. Empty retains the established exact-label policy, with its original
missingness visible in metadata. Unsupported tokens or unit spellings are
excluded and counted; no conversion is guessed. Original frozen rows remain
available for inspection. MHC-MS elution is observed presentation evidence; it
does not imply a detectability or excision target. Live bulk-MS inputs cannot be
added to a frozen Hitlist run; verified search-scoped consumption is tracked in
[Presto #70](https://github.com/pirl-unc/presto/issues/70).

## Frozen membership and generated controls

After source/target and MHC-resolution filtering, Presto assigns measured
observations before generating controls. Connected components keep repeated
peptides, IEDB/CEDAR assay mirrors, shared contributor rows and known reviewed
experiment/acquisition identities together. The seed orders equal-sized
components deterministically; components are never broken to hit a requested
fraction. Actual counts and the largest component are reported. Too few
components for the requested nonempty partitions is an error.

`--frozen-partition-policy` defaults to `peptide_disjoint`. A stronger policy
(`independent_experiments`, `specimen_disjoint`, `donor_disjoint`, `whole_study`)
requires the corresponding reviewed identity coverage. Required unknowns fail
with an inspectable audit, rather than silently weakening the policy. Every
frozen policy additionally keeps actual model peptide sequences disjoint.
A peptide holdout alone does not establish independent donors or experiments.
`--split-mode random_rows` is unavailable for frozen inputs.

To reuse published membership, supply `--frozen-assignment-file PATH` and
`--frozen-assignment-sha256 DIGEST`. The parquet requires `evidence_row_id`,
`source_bundle_id` and `source_partition` (`train`, `val`, `test`). Every selected
observation must be assigned once; extra filtered observations are counted.
Conflicting identities, split components, missing membership and failed policy
audits are errors. Fractions declare the required partitions; a supplied
assignment determines their actual sizes.

Controls draw all peptide/allele/flank donors inside their parent partition.
`generated_id`, `generation_kind`, seed/draw index, `parent_ids` and transitive
`root_observation_ids` survive dataset conversion, collation and prediction
exports. These are host metadata. Generated labels have no measured assay ID.
Known selected positive elution pairs are excluded from generated negatives;
real-MHC binding controls cannot duplicate selected measured pairs. Rejected
attempts are counted. Other generated labels remain explicit assumptions.

Datacache stores reusable artifacts under the Hitlist data directory:

| Directory | Inspection artifacts |
| --- | --- |
| `presto-partitions/<contract hash>` | `assignments.tsv/parquet`, `policy.json`, `coverage.json`, `lineage_audit.json` |
| `presto-membership/<contract hash>` | `membership.tsv/parquet`, `parent_graph.parquet`, `root_lineage_audit.json`, `model_peptide_audit.json`, `coverage.json`, `records_without_model_sample.json` |

The first artifact freezes selected measured observations; the second records
actual model samples after generation and dataset filtering. Parent graphs retain
records omitted from final model samples, so cascaded controls remain traceable.
TSV exports retain readable original endpoint/units/censor/value fields and links
back to the source bundle's full cell/assay context. Target provenance separates
measured, proxy and generated supervision. The data funnel records shared paths,
policy and all artifact hashes; the prospective output-support contract pins
membership and parent-graph identities. Predictions carry partition and parent
fields. A failed final overlap audit is saved before training is refused.

## Qualification still required

These contracts implement the frozen-input and descendant requirements of
[#71](https://github.com/pirl-unc/presto/issues/71).
A bundle identity or a passing software test is not a training-readiness claim.
Qualify the actual selected corpus and partitions, report unknown lineage, and
publish held-out evaluation evidence before making that claim.

The three corpus-specific regression checks require explicit qualification data:

```bash
PRESTO_TEST_HITLIST_BUNDLE=/path/to/frozen/bundle \
PRESTO_TEST_HITLIST_BUNDLE_SHA256=MANIFEST_SHA256 \
./test.sh tests/test_excision_window.py tests/test_stimulus_vocabulary.py
```

These check both MHC classes' flank coverage and every observed condition category.
Presto verifies and shares the bundle through datacache before reading it. With
neither variable, the three checks are reported as skipped, meaning corpus
qualification is unavailable. An incomplete pin, missing data, corrupt payload,
incompatible schema or failed coverage assertion fails. Ordinary software CI
still builds and consumes a small real-schema frozen Hitlist fixture; it does
not certify the full corpus. See [#77](https://github.com/pirl-unc/presto/issues/77).

The programmatic `load_records_from_hitlist()` call without `bundle_dir` retains
legacy live-index compatibility and marks `input_mode=legacy_live_indexes`.
Canonical training does not use that path.
The legacy adapter requires Hitlist 1.53 or newer; frozen ingestion enforces 1.74
or newer for its coherent-generation contract, matching the installation extra
and remote launcher.
