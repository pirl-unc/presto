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

## Qualification still required

This adapter establishes frozen acquisition and measured-record traceability.
Generated-control parents, actual partition artifacts and biological leakage
audits remain tracked in [#71](https://github.com/pirl-unc/presto/issues/71).
A bundle identity or a passing software test is not a training-readiness claim.
Qualify the actual selected corpus and partitions, report unknown lineage, and
publish held-out evaluation evidence before making that claim.

The programmatic `load_records_from_hitlist()` call without `bundle_dir` retains
legacy live-index compatibility and marks `input_mode=legacy_live_indexes`.
Canonical training does not use that path.
