"""Diagnose original panel length claims without changing any source data."""
import argparse
import json
import sys
from pathlib import Path
import openpyxl

p = argparse.ArgumentParser()
p.add_argument("--requests", type=Path, required=True)
r = json.loads(p.parse_args().requests.read_text())
sys.path.insert(0, r["hitlist_checkout"])
from hitlist.nontryptic_workbooks import read_source_workbook
path = Path(r["panel"])
sheet = read_source_workbook(path)[0]
independent = openpyxl.load_workbook(path, data_only=True)[sheet.name]
rows = []
for row in range(4, 22):
    seq = sheet.cells[f"B{row}"].value_literal
    length = sheet.cells[f"E{row}"]
    assert independent[f"B{row}"].value == seq
    raw = length.value_literal
    expected = float(raw) if length.excel_type == "n" and raw is not None else raw
    assert independent[f"E{row}"].value == expected
    rows.append(dict(sequence=seq, calculated_length=len(seq), length_cell=f"E{row}",
                     raw_length=raw, excel_type=length.excel_type, has_formula=length.has_formula,
                     formula=length.formula, independent_value=expected,
                     agrees=expected == len(seq)))
result = dict(status="original_cells_replayed", rows=rows,
              disagreements=[x for x in rows if not x["agrees"]])
root = Path(__file__).resolve().parents[1]
with (root / "results/panel-length-diagnosis.json").open("x") as out:
    json.dump(result,out,indent=2)
print(json.dumps(result,indent=2))
