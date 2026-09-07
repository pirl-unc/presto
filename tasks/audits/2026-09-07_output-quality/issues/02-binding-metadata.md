The default merged TSV adapter drops observed binding assay metadata before collation. Known preparation/geometry/readout observations therefore supervise the unknown panel columns, leaving named columns without their intended source supervision.

Audited at `1535b5a208b2ab59b70b83d850b1f0a7e1959537`; follow-up to #46/#47. This predates the PR #47 fixes.

### Evidence

[The adapter reads assay_type and assay_method](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/scripts/train_iedb.py#L3836), passes them to an intermediate UnifiedRecord, then [constructs BindingRecord without them](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/scripts/train_iedb.py#L3894). BindingRecord already has both fields. [Dataset construction forwards them](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/data/loaders.py#L2070) and [collation factorizes the method](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/data/collate.py#L1010), so the break is in the TSV-to-record adapter.

Deterministic reproduction using the actual loader:

| Source input | Result |
|---|---|
| Numeric IC50 100 nM; assay_type IC50 | BindingRecord.assay_type is None; family fallback still recognizes IC50 |
| assay_method `purified MHC/direct/fluorescence` | BindingRecord.assay_method is None |
| Expected PURIFIED / DIRECT / FLUORESCENCE | Batch prep / geometry / readout indices all **0 (unknown)**; method index also 0 |

A concrete real input exists at line 6 of the local `data/merged_deduped.tsv`: **EVMPVSMAK / HLA-A*03:01 / 473 nM / exact / dissociation constant KD (~EC50)** with `purified MHC/direct/fluorescence`. This is an observed source example, not an estimate of affected corpus prevalence.

### Minimal reproduction

```python
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from presto.scripts.train_iedb import load_records_from_merged_tsv

row = dict(
    peptide="SIINFEKL",
    mhc_allele="HLA-A*02:01",
    mhc_class="I",
    source="audit_fixture",
    record_type="binding",
    value="100",
    value_type="IC50",
    qualifier="0",
    assay_type="IC50",
    assay_method="purified MHC/direct/fluorescence",
)
with TemporaryDirectory() as directory:
    path = Path(directory) / "binding.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    records = load_records_from_merged_tsv(
        path,
        max_binding=10,
        max_kinetics=10,
        max_stability=10,
        max_processing=10,
        max_elution=10,
        max_tcell=10,
        max_vdjdb=10,
    )[0]
    print(records[0].assay_method)  # None
```

### Acceptance criteria

- [ ] Copy observed assay type/method and applicable existing culture fields into BindingRecord without changing units, qualifiers or treating descriptors as predictive inputs.
- [ ] Trace a real source observation through TSV → BindingRecord → PrestoSample → binding_context → selected panel column.
- [ ] Verify a known preparation/readout embedding row receives gradient from that observation; unknown is selected only when the source metadata is unknown.
- [ ] Preserve exact before/after source counts, selected-column support and descriptor missingness; report affected observations, not only a synthetic fixture.
- [ ] Audit analogous kinetic/stability adapter fields and explain whether their panels are supported. Their vocabulary entries alone must not imply quantitative supervision.
- [ ] Add an adapter regression test exercising conflicting/non-default descriptors; retain fixed-output invariance to supplied assay metadata.

This is an ingestion repair. Family-specific quantitative restructuring and the broader biological-context schema remain #46 work.
