"""Independent original-row replay of the condition admission artifacts."""
from collections import Counter
import bisect
import gzip
import hashlib
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1] / "results"
    summary = json.loads((root / "summary.json").read_text())
    columns = json.loads((root / "schema.json").read_text())["columns_read"]
    expected, classes, factors = {}, Counter(), Counter()
    with gzip.open(summary["observations"]["path"], "rt") as stream:
        for line in stream:
            row = json.loads(line)
            provenance = row["provenance"]
            ordinal = provenance["source_ordinal"]
            raw = provenance["raw_record"]
            assert ordinal not in expected
            assert row["observation_id"] == raw["evidence_row_id"]
            assert row["peptide"] == raw["peptide"]
            assert row["split"] == "predict" and row["group_id"] == "unpartitioned"
            assert row["endpoint"] == "elution_observation" and row["value"] == 1
            assert provenance["audit_only_not_train_eligible"] is True
            assert raw["has_ptm"] is False and not raw["peptide_modifications"]
            assert raw["is_non_peptide_ligand"] is False
            assert raw["condition_status"] == "annotated"
            expected[ordinal] = digest(raw)
            cls = raw["mhc_class"]
            classes[cls] += 1
            for family, prefix in (("interventions", "intervention"), ("cytokines", "cytokine")):
                for name, claim in row["context"][family].items():
                    evidence = claim["evidence"]
                    value = raw[evidence["source_field"]]
                    assert evidence["source_value"] == value
                    state = claim["state"]
                    token = name.split(":", 1)[1] if family == "interventions" else name
                    tokens = {x.strip() for x in value.split(";")}
                    assert (state == "absent" and tokens == {"none"}) or (state == "present" and token in tokens)
                    factors[(raw["pmid"], cls, f"{prefix}:{name}", state)] += 1
    ordinals = sorted(expected)
    offset = 0
    verified = 0
    pf = pq.ParquetFile(summary["inputs"]["source"]["path"])
    for batch in pf.iter_batches(columns=columns, batch_size=65536, use_threads=False):
        lo, hi = bisect.bisect_left(ordinals, offset), bisect.bisect_left(ordinals, offset + len(batch))
        selected = ordinals[lo:hi]
        if selected:
            rows = pa.Table.from_batches([batch]).take(pa.array([x - offset for x in selected])).to_pylist()
            for ordinal, row in zip(selected, rows, strict=True):
                assert digest(row) == expected[ordinal], ordinal
                verified += 1
        offset += len(batch)
    assert verified == len(expected)
    assert dict(classes) == {r["mhc_class"]: r["n_rows"] for r in summary["dispositions"] if r["disposition"] == "admitted"}
    for contrast in json.loads((root / "candidate-contrasts.json").read_text()):
        for state, values in contrast["states"].items():
            assert values["n_observation_rows"] == factors[(contrast["pmid"], contrast["mhc_class"], contrast["factor"], state)]
    receipt = {"status": "passed", "n_original_rows_replayed": verified,
               "n_rows_by_class": dict(classes), "n_source_columns_verified_per_row": len(columns),
               "source_values_and_explicit_condition_claims_match": True,
               "candidate_contrast_counts_match": True, "all_rows_remain_unpartitioned": True,
               "elapsed_seconds": time.perf_counter() - started}
    (root / "independent-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
