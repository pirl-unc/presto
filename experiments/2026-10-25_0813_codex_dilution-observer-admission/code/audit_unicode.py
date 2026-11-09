"""Preserve processed MS slots/forms and qualify the observation contract only."""

from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parents[1]
C = json.loads((P / "reproduce/audit-unicode-launch.json").read_text())
D = Path(C["source_directory"])
project = json.loads((D / "project.json").read_text())
files = json.loads((D / "files-project-page0.json").read_text())
article = ET.parse(D / "article.xml").getroot()
table = article.find(".//table-wrap[@id='T1']")
table_rows = [
    ["".join(cell.itertext()) for cell in row.findall("td")]
    for row in table.findall(".//tbody/tr")
]
assert len(table_rows) == 5
doses = []
for i, row in enumerate(table_rows, 1):
    factor, sample, value = row
    number, units = value.split()
    assert sample == f"No. {i}" and units == "fmol/µL"
    assert Decimal(number) == Decimal("0.2") / Decimal(2 ** (i - 1))
    doses.append({"sample": sample, "factor": factor, "concentration": number, "units": units})
panel = re.findall(r"[A-Z]+\[L\(13C6; 15N\)\][A-Z]+", project["sampleProcessingProtocol"])
assert len(panel) == 10 and len(set(panel)) == 10
standard_map = {s.replace("[L(13C6; 15N)]", "L"): s for s in panel}
standard_rows = []
standard_forms = defaultdict(Counter)
states = defaultdict(Counter)
identities = defaultdict(set)
slots = Counter()
run_counts = Counter()
column_hashes = defaultdict(hashlib.sha256)
numeric = ["RT", "intensity", "leftWidth", "rightWidth", "peak_group_rank", "m_score", "alignment_rank", "charge"]
columns = ["peptide_id", "precursor", "run", "RT", "intensity", "leftWidth", "rightWidth", "peak_group_rank", "m_score", "alignment_rank", "feature_id", "sequence", "charge", "group_label"]
with (D / "Dilute_peptide_quantities.csv").open(newline="") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    assert reader.fieldnames == columns
    n = 0
    for line, row in enumerate(reader, 2):
        assert set(row) == set(columns) and all(v is not None for v in row.values())
        n += 1
        for k, v in row.items():
            column_hashes[k].update((json.dumps(v, ensure_ascii=True) + "\n").encode())
        for k in numeric:
            value = row[k]
            if value in {"", "NA", "NaN", "nan", "null"}:
                state = "missing:" + value
            else:
                f = float(value)
                state = "nonfinite" if not math.isfinite(f) else ("positive" if f > 0 else "negative" if f < 0 else "zero")
            states[k][state] += 1
        for k in ["peptide_id", "precursor", "run", "sequence", "charge", "group_label"]:
            identities[k].add(row[k])
        slots[(row["precursor"], row["run"])] += 1
        run_counts[row["run"]] += 1
        bare = re.sub(r"\([^)]*\)|\[[^]]*\]", "", row["sequence"]).strip("_")
        if bare in standard_map:
            standard_forms[bare][row["sequence"]] += 1
            standard_rows.append({"source_line": line, "canonical_candidate": bare, **row})
run_links = []
for run in sorted(identities["run"]):
    match = re.search(r"_No([1-5])_Dilution([^_]+)_([^_]+)fmol-ul_DIA#([12])_", run)
    assert match, run
    sample, factor, amount_token, acquisition = match.groups()
    matching = [f for f in files if f["fileName"] == run.replace("#", "_") + ".raw"]
    run_links.append({"run": run, "sample_index_from_name": int(sample), "factor_token": factor,
                      "concentration_token": amount_token, "acquisition_index_from_name": int(acquisition),
                      "rows": run_counts[run], "raw_file_matches": [f["fileName"] for f in matching],
                      "join_evidence": "Name-derived, exact after replacing # with _; not a physical specimen/pairing assertion"})
paragraph_conflict = [" ".join("".join(el.itertext()).split()) for el in article.findall(".//body//p") if "1:10" in "".join(el.itertext())]
assert paragraph_conflict
result = {
    "passed": True, "columns": columns, "n_rows": n,
    "column_string_sha256": {k: v.hexdigest() for k, v in column_hashes.items()},
    "numeric_states": {k: dict(v) for k, v in states.items()},
    "identity_counts": {k: len(v) for k, v in identities.items()},
    "n_precursor_run_slots": len(slots), "duplicate_slots": sum(v > 1 for v in slots.values()),
    "slot_multiplicity": dict(Counter(slots.values())),
    "run_links": run_links, "article_table1_doses": doses,
    "nominal_standards": [{"canonical_candidate": k, "reported_form": v,
                          "observed_sequence_forms": dict(standard_forms[k])} for k, v in standard_map.items()],
    "standard_query_rows": len(standard_rows),
    "inventory_n_files": len(files),
    "inventory_extensions": dict(Counter(Path(f["fileName"]).suffix for f in files)),
    "all_deposit_checksums_missing": all(not f["checksum"] for f in files),
    "dilution_conflict": "Table1/project protocol twofold vs article Results 1:10; retain both source assertions",
    "source_conflict_paragraph_count": len(paragraph_conflict),
    "interpretation": "Unfiltered observation slots, no fitted calibration or native concentration claim",
}
(P / "results/source-audit.json").write_text(json.dumps(result, indent=2) + "\n")
(P / "results/standard-source-rows.json").write_text(json.dumps(standard_rows, indent=2) + "\n")
print(json.dumps({"n_rows": n, "identity_counts": result["identity_counts"], "standard_query_rows": len(standard_rows), "numeric_states": result["numeric_states"]}))
