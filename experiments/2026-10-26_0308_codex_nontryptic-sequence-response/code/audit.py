"""Independently qualify admission before any original-data model fit."""

from collections import Counter
import math
from pathlib import Path

import pyarrow.parquet as pq

from common import ROOT, read, request, write


def main():
    req = request()
    base = Path(req["admission_directory"])
    cfg = read(ROOT / "reproduce/launch.json")
    typed = Path(cfg["package"]) / "nontryptic_calibration/typed"
    source = {(r["sheet"], r["source_cell"]): r for r in read(req["qualified_s3_cells"])}
    standards = {
        r["standard_id"]: r for r in pq.read_table(typed / "standards.parquet").to_pylist()
    }
    amounts = {
        r["nominal_amount_id"]: r
        for r in pq.read_table(typed / "nominal_amounts.parquet").to_pylist()
    }
    decisions = read(base / "decisions.json")
    assert len(decisions) == 348
    seen = set()
    for r in decisions:
        sheet, cell = r["source_locator"].split("#")[1].split("!")
        assert (sheet, cell) not in seen
        seen.add((sheet, cell))
        ref = source[(sheet, cell)]
        assert r["quantitative_value"] == ref["raw_signal"]
        assert r["value_status"] == ref["signal_status"]
        assert r["standard.peptide_sequence"] == ref["sequence"]
        assert r["standard.gene_symbol_raw"] == standards[r["standard_id"]]["gene_symbol_raw"]
        assert r["amount.nominal_amount_fmol"] == ref["added_fmol"]
        assert (
            r["amount.nominal_amount_fmol"]
            == amounts[r["nominal_amount_id"]]["nominal_amount_fmol"]
        )
        assert r["slot.source_slot_ordinal"] == ref["replicate_column_ordinal"]
        assert r["slot.acquisition_id"] is None
        if sheet == "DatasetS3A":
            assert r["decision"] == "accepted"
        else:
            assert sheet == "DatasetS3B" and r["decision"] == "excluded"
            assert "context_precision_unmet" in r["reason_codes"]
    assert seen == set(source)
    rows = read(base / "admitted.json")
    assert len(rows) == 213
    expected = {r["measurement_id"] for r in decisions if r["decision"] == "accepted"}
    assert {r["measurement_id"] for r in rows} == expected
    by_id = {r["measurement_id"]: r for r in decisions}
    for r in rows:
        assert all(r[k] == v for k, v in by_id[r["measurement_id"]].items())
        assert r["y"] == math.log10(r["quantitative_value"])
        assert r["x"] == math.log10(r["amount.nominal_amount_fmol"])
        assert r["sequence"] == r["standard.peptide_sequence"]
        assert r["gene"] == r["standard.gene_symbol_raw"]
    folds = read(base / "folds.json")
    assert len(folds) == len({r["gene"] for r in rows}) == 13
    for f in folds:
        assert set(f["held_sequences"]) == {
            r["sequence"] for r in rows if r["gene"] == f["held_gene"]
        }
        assert set(f["training_sequences"]) == {
            r["sequence"] for r in rows if r["gene"] != f["held_gene"]
        }
        assert not set(f["held_sequences"]) & set(f["training_sequences"])
    result = {
        "status": "independent_admission_passed",
        "n_original_slots": 348,
        "n_admitted": 213,
        "n_precursor_genes": 13,
        "decisions": dict(Counter(r["decision"] for r in decisions)),
        "n_model_fits": 0,
        "source_scope": "free1H_postpurification_nominal_MS2_response",
    }
    write(ROOT / "results" / req["result_file"], result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
