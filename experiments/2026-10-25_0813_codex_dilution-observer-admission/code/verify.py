"""Independent dataframe/source-text audit; no import of the production auditor."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import xml.dom.minidom as DOM

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
C = json.loads((P / "reproduce/audit-launch.json").read_text())
D = Path(C["source_directory"])
A = json.loads((P / "results/source-audit.json").read_text())
F = pd.read_csv(D / "Dilute_peptide_quantities.csv", sep="\t", dtype=str, keep_default_na=False)
assert F.columns.tolist() == A["columns"]
assert len(F) == A["n_rows"]
for name in F.columns:
    digest = hashlib.sha256()
    for value in F[name].tolist():
        digest.update((json.dumps(value, ensure_ascii=True) + "\n").encode())
    assert digest.hexdigest() == A["column_string_sha256"][name], name
for name, expected in A["numeric_states"].items():
    values = F[name]
    missing = values.isin(["", "NA", "NaN", "nan", "null"])
    counts = {"missing:" + k: int(v) for k, v in values[missing].value_counts().items()}
    numbers = pd.to_numeric(values[~missing], errors="raise").to_numpy()
    finite = np.isfinite(numbers)
    states = {"nonfinite": int((~finite).sum()), "positive": int((numbers[finite] > 0).sum()),
              "negative": int((numbers[finite] < 0).sum()), "zero": int((numbers[finite] == 0).sum())}
    counts.update({k: v for k, v in states.items() if v})
    assert counts == expected, name
for key, expected in A["identity_counts"].items():
    assert F[key].nunique(dropna=False) == expected, key
slots = F.groupby(["precursor", "run"], dropna=False).size()
assert len(slots) == A["n_precursor_run_slots"]
assert int((slots > 1).sum()) == A["duplicate_slots"]
assert {str(k): int(v) for k, v in slots.value_counts().items()} == A["slot_multiplicity"]
project = json.loads((D / "project.json").read_text())
panel_text = project["sampleProcessingProtocol"].split("distinction between native and spike-in peptides (", 1)[1].split("). The synthetic peptide mix", 1)[0]
panel = panel_text.split(", ")
assert len(panel) == 10
standards = {s.replace("[L(13C6; 15N)]", "L"): s for s in panel}
assert standards == {x["canonical_candidate"]: x["reported_form"] for x in A["nominal_standards"]}


def unmodified(value):
    output, depth = [], 0
    for char in value:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            assert depth >= 0
        elif not depth and char != "_":
            output.append(char)
    assert depth == 0
    return "".join(output)


bare = F["sequence"].map(unmodified)
selected = F.loc[bare.isin(standards)]
assert len(selected) == A["standard_query_rows"]
reported_rows = json.loads((P / "results/standard-source-rows.json").read_text())
assert [int(i) + 2 for i in selected.index] == [r["source_line"] for r in reported_rows]
for source, reported in zip(selected.to_dict("records"), reported_rows):
    assert source == {k: reported[k] for k in F.columns}
for standard in A["nominal_standards"]:
    assert F.loc[bare == standard["canonical_candidate"], "sequence"].value_counts().to_dict() == standard["observed_sequence_forms"]
document = DOM.parse(str(D / "article.xml"))
table = next(x for x in document.getElementsByTagName("table-wrap") if x.getAttribute("id") == "T1")


def text(node):
    return "".join(text(x) if x.nodeType == x.ELEMENT_NODE else x.data if x.nodeType in (x.TEXT_NODE, x.CDATA_SECTION_NODE) else "" for x in node.childNodes)


rows = [row.getElementsByTagName("td") for row in table.getElementsByTagName("tr")]
rows = [cells for cells in rows if cells]
assert len(rows) == 5
for index, (cells, expected) in enumerate(zip(rows, A["article_table1_doses"])):
    factor, sample, concentration = [text(c) for c in cells]
    amount, unit = concentration.split(" ")
    assert Fraction(amount) == Fraction(1, 5 * (2 ** index))
    assert expected == {"sample": sample, "factor": factor, "concentration": amount, "units": unit}
files = json.loads((D / "files-project-page0.json").read_text())
receipt = json.loads((P / "results/files-retrieval.json").read_text())
assert len(files) == int(receipt["assets"][0]["headers"]["total_records"]) == A["inventory_n_files"]
assert len({x["fileName"] for x in files}) == len(files)
for link in A["run_links"]:
    run = link["run"]
    assert link["rows"] == int((F["run"] == run).sum())
    assert link["raw_file_matches"] == [x["fileName"] for x in files if x["fileName"] == run.replace("#", "_") + ".raw"]
    pieces = run.split("_")
    assert int(pieces[5][2:]) == link["sample_index_from_name"]
    assert pieces[6].removeprefix("Dilution") == link["factor_token"]
    assert pieces[7].removesuffix("fmol-ul") == link["concentration_token"]
    assert int(pieces[8].split("#")[1]) == link["acquisition_index_from_name"]
result = {"passed": True, "source_string_cells_verified": len(F) * len(F.columns),
          "rows": len(F), "standards": len(panel), "standard_rows": len(selected),
          "inventory_files": len(files), "table1_values": len(rows),
          "complete_precursor_by_run_grid": len(slots) == F["precursor"].nunique() * F["run"].nunique(),
          "reader": "pandas string-preserving parser and independent DOM/literal-protocol parser",
          "no_production_auditor_import": True, "model_fitted": False}
(P / "results/verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
