"""Check complete audit closure and that real-value perturbations are detected."""

import gzip
import io
import json
import zipfile
from pathlib import Path

import replay

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT / "results/conditions.json").read_text())
    summary = json.loads((ROOT / "results/summary.json").read_text())
    assert len(result) == len({r["condition"] for r in result}) == 17
    assert all(r["passed"] and not r["errors"] for r in result)
    assert summary["raw_rows"] == sum(r["raw_rows"] for r in result) == 545664
    assert summary["final_slots"] == sum(r["final_slots"] for r in result) == 280664
    assert sum(r["source_missing_slots"] for r in result) == 4782
    assert sum(r["source_zero_slots"] for r in result) == 62860
    with gzip.open(ROOT / "results/discrepancies.jsonl.gz", "rt") as stream:
        assert stream.read() == ""
    previous = json.loads(
        (
            ROOT.parent
            / "2026-10-23_0800_codex_digestion-source-audit/results/h4-arithmetic-replay.json"
        ).read_text()
    )
    h4 = next(r for r in result if r["condition"] == "aSPIRE_H4")
    for old, new in {
        "raw_quantity_rows_replayed": "raw_rows",
        "raw_unique_precursor_rows": "distinct_precursor_rows",
        "global_background_median": "baseline_median",
        "values_censored_by_background": "below_background_contributors",
        "final_vector_values_replayed": "final_slots",
        "final_missing_values_replayed": "source_missing_slots",
    }.items():
        assert previous[old] == h4[new]

    read = replay.read_rows
    mutated = {}

    def changed_rows(table, condition):
        rows = read(table, condition)
        if table == "raw_quantities":
            for row, rid in rows:
                if float(row["intensity"]) > 0:
                    row["intensity"] = str(float(row["intensity"]) * 1.01)
                    mutated[table] = rid
                    break
        if table == "final_vectors":
            for row, rid in rows:
                values = row["intensities"].split(";")
                for index, value in enumerate(values):
                    if replay.number(value) is not None and float(value) > 0:
                        values[index] = str(float(value) * 2)
                        row["intensities"] = ";".join(values)
                        mutated[table] = rid
                        break
                if table in mutated:
                    break
        return rows

    replay.read_rows = changed_rows
    discrepancies = io.StringIO()
    with zipfile.ZipFile(replay.SOURCE / "assets/aSPIRE_outputs_zip.zip") as outer:
        name = next(n for n in outer.namelist() if Path(n).stem == "aSPIRE_H4")
        with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
            mutant = replay.condition_replay(inner, "aSPIRE_H4", discrepancies)
    assert not mutant["passed"]
    assert mutant["errors"]["raw"] == mutant["errors"]["final"] == 1
    assert len(mutated) == 2
    record = {
        "passed": True,
        "n_conditions": 17,
        "raw_rows": 545664,
        "final_slots": 280664,
        "prior_independent_H4_invariants_match": True,
        "in_memory_real_value_mutations": mutated,
        "detected_errors": mutant["errors"],
        "original_source_unchanged": True,
    }
    (ROOT / "results/audit-verification.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    (ROOT / "results/mutation-discrepancies.jsonl").write_text(discrepancies.getvalue())
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
