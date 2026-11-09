"""Literal source inventory; no fitted values, filters or inferred detection."""

from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/inventory-launch.json").read_text())
MISSING = {"", "NA", "NaN", "nan", "null"}
panel = json.loads(Path(CONFIG["prior_audit"]).read_text())["nominal_standards"]
standards = {row["canonical_candidate"] for row in panel}


def write(name, value):
    (ROOT / "results" / name).write_text(json.dumps(value, indent=2) + "\n")


def numeric_state(value):
    if value in MISSING:
        return "missing:" + value
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return "invalid"
    if not parsed.is_finite():
        return "nonfinite"
    return "positive" if parsed > 0 else "negative" if parsed < 0 else "zero"


inventories, standard_rows = [], []
memberships = defaultdict(lambda: {"n_transitions": 0, "first_source_line": None})
for asset in CONFIG["tabular_assets"]:
    path = Path(asset["path"])
    role = asset["role"]
    identity = defaultdict(set)
    missing = defaultdict(Counter)
    states = defaultdict(Counter)
    duplicates = Counter()
    flags = defaultdict(Counter)
    digest = hashlib.sha256()
    mismatched_names = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        assert len(header) == len(set(header))
        digest.update((json.dumps(header, ensure_ascii=True) + "\n").encode())
        n_rows = 0
        for line, cells in enumerate(reader, 2):
            assert len(cells) == len(header), (path.name, line)
            n_rows += 1
            digest.update((json.dumps(cells, ensure_ascii=True) + "\n").encode())
            row = dict(zip(header, cells))
            for key, value in row.items():
                if value in MISSING:
                    missing[key][value] += 1
            for key in asset["identities"]:
                identity[key].add(row[key])
            for key in asset["numeric"]:
                states[key][numeric_state(row[key])] += 1
            for key in asset["flags"]:
                flags[key][row[key]] += 1
            if role == "master_query_library":
                duplicates[row["TransitionId"]] += 1
                key = tuple(row[k] for k in CONFIG["membership_fields"])
                memberships[key]["n_transitions"] += 1
                if memberships[key]["first_source_line"] is None:
                    memberships[key]["first_source_line"] = line
                bare = row["PeptideSequence"]
                if bare in standards:
                    standard_rows.append({"asset": path.name, "role": role,
                                          "source_line": line, "cells": row})
            elif role == "per_run_scored_candidates":
                duplicates[(row["filename"], row["id"])] += 1
                if row["filename"].replace("#", "_") + ".tsv" != path.name:
                    mismatched_names.append(line)
                bare = row["Sequence"]
                if bare in standards:
                    standard_rows.append({"asset": path.name, "role": role,
                                          "source_line": line, "cells": row})
            else:
                duplicates[(row["run"], row["precursor"])] += 1
                bare = re.sub(r"\([^)]*\)|\[[^]]*\]", "", row["sequence"]).strip("_")
                if bare in standards:
                    standard_rows.append({"asset": path.name, "role": role,
                                          "source_line": line, "cells": row})
    summary = {"asset": path.name, "role": role, "headers": header,
               "n_rows": n_rows, "n_cells": n_rows * len(header),
               "literal_rows_sha256": digest.hexdigest(),
               "missing_tokens": {k: dict(missing[k]) for k in header},
               "numeric_states": dict(states), "flag_counts": dict(flags),
               "identity_counts": {k: len(v) for k, v in identity.items()},
               "identity_values": {k: sorted(identity[k]) for k in asset["retain_identity_values"]},
               "n_duplicate_keys": sum(v > 1 for v in duplicates.values()),
               "key_multiplicity": dict(Counter(duplicates.values())),
               "filename_mismatch_source_lines": mismatched_names}
    inventories.append(summary)
    print(json.dumps({"asset": path.name, "n_rows": n_rows,
                      "n_duplicate_keys": summary["n_duplicate_keys"]}), flush=True)
write("table-inventory.json", inventories)
write("library-memberships.json", [{**dict(zip(CONFIG["membership_fields"], key)), **value}
                                    for key, value in sorted(memberships.items())])
write("standard-candidate-rows.json", standard_rows)
write("inventory-completion.json", {"passed": True, "n_tables": len(inventories),
                                   "n_rows": sum(x["n_rows"] for x in inventories),
                                   "n_cells": sum(x["n_cells"] for x in inventories),
                                   "n_library_memberships": len(memberships),
                                   "n_standard_candidate_rows": len(standard_rows)})
