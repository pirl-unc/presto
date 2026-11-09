"""Exact form membership across explicitly distinct precursor ID namespaces."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1]
C = json.loads((P / "reproduce/membership-launch.json").read_text())
library = json.loads((P / "results/library-memberships.json").read_text())
lookup = defaultdict(list)
for row in library:
    lookup[(row["ModifiedPeptideSequence"], row["PrecursorCharge"], row["Decoy"])].append(row)
counts = defaultdict(Counter)
exceptions, mapping_counts = [], Counter()
for asset in C["tabular_assets"]:
    role = asset["role"]
    if role == "master_query_library":
        continue
    with Path(asset["path"]).open(newline="") as source:
        for line, row in enumerate(csv.DictReader(source, delimiter="\t"), 2):
            if role == "aligned_quantities":
                key = (row["sequence"], row["charge"], "0")
                group = row["precursor"]
                extra_field, extra_value = "PeptideGroupLabel", row["group_label"]
            else:
                key = (row["FullPeptideName"], row["Charge"], row["decoy"])
                group = row["transition_group_id"]
                extra_field, extra_value = "PeptideSequence", row["Sequence"]
            matches = lookup[key]
            status = ("absent_form" if not matches else "ambiguous_form" if len(matches) > 1 else
                      "matched" if matches[0][extra_field] == extra_value else "disagrees:" + extra_field)
            counts[role][status] += 1
            if status != "matched":
                exceptions.append({"asset": Path(asset["path"]).name, "source_line": line,
                                   "state": status, "form_key": key})
            if len(matches) == 1:
                mapping_counts[(role, group, matches[0]["TransitionGroupId"], *key)] += 1
mapping = [{"role": key[0], "scored_or_aligned_group_id": key[1],
            "library_group_id": key[2], "form": key[3], "charge": key[4],
            "decoy": key[5], "n_source_rows": count}
           for key, count in sorted(mapping_counts.items())]
result = {"counts": counts, "n_mapping_entries": len(mapping),
          "n_ambiguous_library_form_keys": sum(len(v) > 1 for v in lookup.values()),
          "semantics": "Exact full-form/charge/target-state membership; original ID namespaces remain distinct"}
for name, value in [("membership-summary.json", result), ("membership-exceptions.json", exceptions),
                    ("namespace-mapping.json", mapping)]:
    (P / "results" / name).write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(result), flush=True)
