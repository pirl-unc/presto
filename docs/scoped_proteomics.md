# Verified scoped proteomics inputs

New datasets must preserve comparable study, sample, assay and measurement
metadata under the shared [study curation contract](study_curation_contract.md).

The Python adapter accepts Hitlist's published PXD013455 HeLa Trypsin/P search
scope. Install `presto[hitlist]` with Hitlist >=1.75.0, then:

```python
from presto.data.scoped_bulk import load_scoped_bulk_records
from presto.data.loaders import PrestoDataset

records, receipt = load_scoped_bulk_records(max_records=10_000, seed=42)
dataset = PrestoDataset(bulk_ms_records=records)
```

Hitlist downloads its pinned [public evidence archive](https://github.com/pirl-unc/hitlist/releases/tag/pxd013455-hela-tryp46-v1).
Datacache verifies and retains the complete input inventory and the reusable
collapsed projection. For an existing portable directory, supply `bundle_dir`
and `bundle_sha256` (the SHA-256 of its `manifest.json`). No mutable bulk index
or arbitrary replacement reference participates in this path.

This adapter supplies **binary observation within one documented search**, as
proxy supervision for the detectability output. Direct MS/MS, matching and
unobserved candidates remain separate evidence categories. Unobserved means
absent in this search, conditional on same-scope parent support. It does not mean
intrinsically undetectable. Depth and biological replicate labels remain missing;
46 fractions are not 46 biological replicates. The adapter supplies no excision
labels or generated wrong-enzyme controls. The older raw bulk path retains its
separate excision proxies and controls, with detectability labels missing.

Each peptide contributes one row, even when the search reference contains several
candidate occurrences. Collapse happens before seeded reservoir sampling. A
unique occurrence retains its parent, flanks and known terminal boundaries.
Multiple occurrences mask those model inputs. Every original occurrence remains
in `candidates.parquet`, looked up by the frozen bundle and `candidate_lookup_key`.
Positive source-observation IDs, scoped peptide identities and candidate occurrence
IDs have distinct roles. Unobserved candidates never acquire measured assay IDs.

The returned receipt records full and admitted evidence counts, cap policy,
source/projection locations and hashes. A tab-separated `inspection.tsv` preview
shows up to 20 peptides per evidence category; full coverage and policy JSON files
are formatted for reading. `source_registry.json` preserves original
and reanalysis citations, cell type, conditions, MHC-not-applicable status,
acquisition protocol, search reference and unresolved donor/specimen identities.
These metadata are available to human inspection and prediction exports; they
are not model inputs. Parent support through leading-razor assignments does not
independently establish every possible peptide parent.

The canonical trainer accepts `--data-source scoped_bulk` and
`--max-scoped-bulk`, with optional pinned `--scoped-bundle` and
`--scoped-bundle-sha256`. It freezes within-search membership and source lineage;
existing assignments can be supplied with `--frozen-assignment-file` and
`--frozen-assignment-sha256`. Run `--data-preflight-only` to retain coverage and
leakage reports before fitting. Disable live auxiliary inputs with
`--mhc-augmentation-samples 0 --uniprot-negative-ratio 0` and leave `--bulk-ms`
unset. The Python adapter alone assigns no partitions.
The [published bounded qualification](https://github.com/pirl-unc/presto/releases/tag/scoped-sampling-qualification-v1)
contains frozen inputs, assignments, predictions and baseline comparisons.
A single cultured-cell experiment
cannot establish experiment-, donor- or specimen-independent held-out performance.
Neither software tests nor projection counts establish training readiness.
