"""Source arithmetic/label audit only; no normalization or training selection."""

from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/quantity-audit.json").read_text())
START = time.perf_counter()
results = []


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


for project in CONFIG["projects"]:
    base = Path(CONFIG["asset_root"]) / project / "extracted/txt"
    peptides, evidence, groups = [rows(base / name) for name in ("peptides.txt", "evidence.txt", "proteinGroups.txt")]
    ev = {row["id"]: row for row in evidence if row["id"]}
    assert len(ev) == len(evidence), "Evidence IDs must be present/unique"
    assert len({row["id"] for row in peptides}) == len(peptides)
    exp_raw = defaultdict(set)
    for row in evidence:
        exp_raw[row["Experiment"]].add(row["Raw file"])
    assert all(len(values) == 1 for values in exp_raw.values())
    summary = {row["Raw file"]: row for row in rows(base / "summary.txt") if row["Raw file"] in {v for values in exp_raw.values() for v in values}}
    crosswalk = [{"evidence_experiment": name, "raw_file": next(iter(values)),
                  "summary_experiment": summary[next(iter(values))]["Experiment"]}
                 for name, values in sorted(exp_raw.items())]
    columns = [key for key in peptides[0] if key.startswith("Intensity ")]
    states = Counter()
    max_relative_error = 0.0
    length_counts = Counter()
    qualifying_flags = Counter()
    with gzip.open(ROOT / f"results/{project}-quantity-comparisons.jsonl.gz", "wt") as dump:
        for index, row in enumerate(peptides):
            length_counts[int(row["Length"])] += 1
            qualifying_flags[(row["Reverse"], row["Potential contaminant"], row["Unique (Groups)"])] += 1
            linked = [ev[key] for key in row["Evidence IDs"].split(";") if key]
            assert all(e["Sequence"] == row["Sequence"] and e["Peptide ID"] == row["id"] for e in linked)
            sums, contributors = defaultdict(float), defaultdict(list)
            for e in linked:
                sums[e["Experiment"]] += float(e["Intensity"] or 0)
                contributors[e["Experiment"]].append(e["id"])
            for column in columns:
                experiment = column.removeprefix("Intensity ")
                assert experiment in exp_raw
                literal, total = row[column], sums[experiment]
                if literal == "":
                    state, reported = "unreported", None
                else:
                    reported = float(literal)
                    if reported == 0 and total == 0:
                        state = "both_zero"
                    elif reported == 0 and total > 0:
                        state = "reported_zero_positive_evidence"
                    elif reported > 0 and total == 0:
                        state = "reported_positive_zero_evidence"
                    elif math.isclose(reported, total, rel_tol=CONFIG["relative_tolerance"], abs_tol=CONFIG["absolute_tolerance"]):
                        state = "positive_agree_within_rounding"
                    else:
                        state = "positive_disagreement"
                    if reported > 0 and total > 0:
                        max_relative_error = max(max_relative_error, abs(reported - total) / max(reported, total))
                states[state] += 1
                dump.write(json.dumps({"peptide_id": row["id"], "sequence": row["Sequence"],
                                       "source_line": index + 2, "source_column": column,
                                       "reported_literal": literal, "evidence_sum": total,
                                       "evidence_ids": contributors[experiment], "state": state}) + "\n")
    result = {
        "project": project, "peptide_rows": len(peptides), "evidence_rows": len(evidence),
        "protein_group_rows": len(groups), "blank_group_ids": sum(not x["id"] for x in groups),
        "intensity_columns": len(columns), "comparisons": sum(states.values()),
        "states": dict(states), "max_positive_relative_discrepancy": max_relative_error,
        "length_counts": dict(sorted(length_counts.items())),
        "source_flag_counts": [{"reverse": k[0], "contaminant": k[1], "unique_groups": k[2], "rows": v} for k, v in sorted(qualifying_flags.items())],
        "evidence_types": dict(Counter(e["Type"] for e in evidence)),
        "unreported_evidence_intensity_rows": sum(e["Intensity"] == "" for e in evidence),
        "run_crosswalk": crosswalk,
    }
    results.append(result)
    print(json.dumps(result), flush=True)
(ROOT / "results/quantity-audit.json").write_text(json.dumps({"projects": results, "wall_seconds": time.perf_counter() - START}, indent=2) + "\n")
