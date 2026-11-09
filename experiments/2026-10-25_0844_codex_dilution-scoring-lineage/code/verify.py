"""Independent pandas literal replay and rational-number verification.

Does not import any of the inventory, lineage or membership implementation.
"""

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parents[1]
C = json.loads((P / "reproduce/verify-launch.json").read_text())
MISSING = ("", "NA", "NaN", "nan", "null")


def read(name):
    return json.loads((P / "results" / name).read_text())


def equivalent(a, b):
    assert json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def number_state(value):
    if value in MISSING:
        return "missing:" + value
    try:
        if not math.isfinite(float(value)):
            return "nonfinite"
        number = Fraction(value)
        return "positive" if number > 0 else "negative" if number < 0 else "zero"
    except (ValueError, ZeroDivisionError):
        return "invalid"


def bare_sequence(text):
    depth = 0
    sequence = []
    for letter in text.strip("_"):
        if letter in "([":
            depth += 1
        elif letter in ")]":
            depth -= 1
        elif depth == 0:
            sequence.append(letter)
    assert depth == 0
    return "".join(sequence)


panel = json.loads(Path(C["prior_audit"]).read_text())["nominal_standards"]
names = {r["canonical_candidate"] for r in panel}
library_counts, library_first = Counter(), {}
standard_rows, inventories, score_frames = [], [], []
aligned = None
score_columns = list(dict.fromkeys(["id", "filename", "FullPeptideName", "Sequence", "Charge",
                                    "transition_group_id", "decoy", *C["compare_fields"].values()]))
for asset in C["tabular_assets"]:
    role, path = asset["role"], Path(asset["path"])
    counts, missing, states, flags = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    unique = {k: set() for k in asset["identities"]}
    row_hash = hashlib.sha256()
    header = None
    n_rows = 0
    bad_filenames = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                             chunksize=8192, encoding="utf-8"):
        if header is None:
            header = list(chunk.columns)
            row_hash.update((json.dumps(header, ensure_ascii=True) + "\n").encode())
        assert list(chunk.columns) == header
        for k in unique:
            unique[k].update(chunk[k])
        if role == "per_run_scored_candidates":
            selected = chunk[score_columns].copy()
            selected["source_line"] = range(n_rows + 2, n_rows + len(chunk) + 2)
            selected["asset"] = path.name
            score_frames.append(selected)
        elif role == "aligned_quantities":
            aligned = chunk.copy() if aligned is None else pd.concat([aligned, chunk], ignore_index=True)
        for cells in chunk.itertuples(index=False, name=None):
            row_hash.update((json.dumps(list(cells), ensure_ascii=True) + "\n").encode())
            n_rows += 1
            line = n_rows + 1
            row = dict(zip(header, cells))
            for key in header:
                if row[key] in MISSING:
                    missing[key][row[key]] += 1
            for key in asset["numeric"]:
                states[key][number_state(row[key])] += 1
            for key in asset["flags"]:
                flags[key][row[key]] += 1
            if role == "master_query_library":
                counts[row["TransitionId"]] += 1
                key = tuple(row[field] for field in C["membership_fields"])
                library_counts[key] += 1
                library_first.setdefault(key, line)
                bare = row["PeptideSequence"]
            elif role == "per_run_scored_candidates":
                counts[(row["filename"], row["id"])] += 1
                bare = row["Sequence"]
                if row["filename"].replace("#", "_") != path.name[:-4]:
                    bad_filenames.append(line)
            else:
                counts[(row["run"], row["precursor"])] += 1
                bare = bare_sequence(row["sequence"])
            if bare in names:
                standard_rows.append({"asset": path.name, "role": role, "source_line": line, "cells": row})
    inventories.append({"asset": path.name, "role": role, "headers": header, "n_rows": n_rows,
                        "n_cells": n_rows * len(header), "literal_rows_sha256": row_hash.hexdigest(),
                        "missing_tokens": {k: dict(missing[k]) for k in header},
                        "numeric_states": dict(states), "flag_counts": dict(flags),
                        "identity_counts": {k: len(v) for k, v in unique.items()},
                        "identity_values": {k: sorted(unique[k]) for k in asset["retain_identity_values"]},
                        "n_duplicate_keys": sum(v != 1 for v in counts.values()),
                        "key_multiplicity": dict(Counter(counts.values())),
                        "filename_mismatch_source_lines": bad_filenames})
    print(json.dumps({"independently_parsed": path.name, "n_rows": n_rows}), flush=True)
equivalent(inventories, read("table-inventory.json"))
equivalent(standard_rows, read("standard-candidate-rows.json"))
library = [{**dict(zip(C["membership_fields"], key)), "n_transitions": library_counts[key],
            "first_source_line": library_first[key]} for key in sorted(library_counts)]
equivalent(library, read("library-memberships.json"))
equivalent({"passed": True, "n_tables": len(inventories), "n_rows": sum(r["n_rows"] for r in inventories),
            "n_cells": sum(r["n_cells"] for r in inventories), "n_library_memberships": len(library),
            "n_standard_candidate_rows": len(standard_rows)}, read("inventory-completion.json"))
scores = pd.concat(score_frames, ignore_index=True)
assert not scores.duplicated(["filename", "id"]).any()
assert not aligned.duplicated(["run", "precursor"]).any()
by_id, by_form = defaultdict(list), defaultdict(list)
for row in library:
    by_id[row["TransitionGroupId"]].append(row)
    by_form[(row["ModifiedPeptideSequence"], row["PrecursorCharge"], row["Decoy"])].append(row)


def failed_namespace_check(group, form, charge, decoy, label=None):
    matched = by_id.get(group, [])
    if len(matched) != 1:
        return "absent_group" if not matched else "ambiguous_group"
    expected = dict(ModifiedPeptideSequence=form, PrecursorCharge=charge, Decoy=decoy)
    if label is not None:
        expected["PeptideGroupLabel"] = label
    different = [field for field in expected if matched[0][field] != expected[field]]
    return "disagrees:" + ",".join(different) if different else "matched"


score_states, score_exceptions = Counter(), []
member_states, member_exceptions, mappings = defaultdict(Counter), [], Counter()


def verify_membership(role, group, form, charge, decoy, field, literal, asset, line):
    candidates = by_form.get((form, charge, decoy), [])
    if not candidates:
        state = "absent_form"
    elif len(candidates) > 1:
        state = "ambiguous_form"
    else:
        state = "matched" if candidates[0][field] == literal else "disagrees:" + field
    member_states[role][state] += 1
    if state != "matched":
        member_exceptions.append({"asset": asset, "source_line": line, "state": state,
                                  "form_key": [form, charge, decoy]})
    if len(candidates) == 1:
        mappings[(role, group, candidates[0]["TransitionGroupId"], form, charge, decoy)] += 1


for cells in scores.itertuples(index=False, name=None):
    row = dict(zip(scores.columns, cells))
    state = failed_namespace_check(row["transition_group_id"], row["FullPeptideName"], row["Charge"], row["decoy"])
    score_states[state] += 1
    if state != "matched":
        score_exceptions.append({"asset": row["asset"], "source_line": row["source_line"], "id": row["id"], "state": state})
    verify_membership("per_run_scored_candidates", row["transition_group_id"], row["FullPeptideName"],
                      row["Charge"], row["decoy"], "PeptideSequence", row["Sequence"], row["asset"], row["source_line"])
equivalent(score_exceptions, read("candidate-library-exceptions.json"))

scores["run"] = scores["filename"].str.slice(stop=-5)
right = scores.rename(columns={k: "sc_" + k for k in scores.columns if k != "run"})
joined = aligned.merge(right, left_on=["run", "feature_id"], right_on=["run", "sc_id"],
                       how="left", sort=False, validate="many_to_one", indicator=True)
assert len(joined) == len(aligned)
saved = pd.read_csv(P / "results/aligned-feature-links.tsv.gz", sep="\t", dtype=str,
                    keep_default_na=False, na_filter=False)
assert len(saved) == len(joined)
links, lib_states, cross, compare_counts = Counter(), Counter(), Counter(), defaultdict(Counter)
aligned_exceptions, standard_links = [], []
forms, labels = {}, {}
for row in panel:
    prefix, suffix = row["reported_form"].split("[L(13C6; 15N)]")
    forms[row["canonical_candidate"]] = prefix + "L(UniMod:695)" + suffix
    labels[row["canonical_candidate"]] = prefix + "L(Label:13C(6)15N(1))" + suffix
inverse = {value: key for key, value in forms.items()}
numeric_checks = 0
aligned_asset = next(Path(a["path"]).name for a in C["tabular_assets"] if a["role"] == "aligned_quantities")
for line, (source_values, saved_values) in enumerate(zip(joined.itertuples(index=False, name=None),
                                                       saved.itertuples(index=False, name=None)), 2):
    row = dict(zip(joined.columns, source_values))
    actual = dict(zip(saved.columns, saved_values))
    out = {k: row[k] for k in ["run", "precursor", "feature_id", "sequence", "charge", "group_label"]}
    matched = row["_merge"] == "both" and row["feature_id"] not in MISSING
    out.update(aligned_source_line=line, n_candidates=int(matched))
    if row["feature_id"] in MISSING:
        state = "missing_feature_id"
    elif not matched:
        state = "unmatched_feature_id"
    else:
        booleans = {"form_agrees": row["sequence"] == row["sc_FullPeptideName"],
                    "charge_agrees": row["charge"] == row["sc_Charge"],
                    "precursor_agrees": row["precursor"] == row["sc_transition_group_id"],
                    "target_agrees": row["sc_decoy"] == "0"}
        out.update({k: str(value).lower() for k, value in booleans.items()})
        out.update(scored_asset=row["sc_asset"], scored_source_line=int(row["sc_source_line"]))
        state = "matched_identity" if all(booleans.values()) else "id_crosscheck_disagreement"
    out["link_state"] = state
    links[state] += 1
    libstate = failed_namespace_check(row["precursor"], row["sequence"], row["charge"], "0", row["group_label"])
    out["library_state"] = libstate
    lib_states[libstate] += 1
    if libstate != "matched":
        aligned_exceptions.append({"source_line": line, "state": libstate})
    cross["|".join([state, number_state(row["intensity"]), number_state(row["m_score"])])] += 1
    for field, right_field in C["compare_fields"].items():
        a = row[field]
        out[field + "_aligned"] = a
        if matched:
            b = row["sc_" + right_field]
            out[field + "_scored"] = b
            if a in MISSING or b in MISSING:
                relation = "both_missing" if a in MISSING and b in MISSING else "aligned_missing" if a in MISSING else "scored_missing"
                out[field + "_delta"] = ""
            else:
                x, y = Fraction(a), Fraction(b)
                relation = ("same_literal" if a == b else "decimal_equal" if x == y else
                            "within_numeric_tolerance" if abs(x-y) <= Fraction(C["atol"]) + Fraction(C["rtol"]) * max(abs(x), abs(y))
                            else "different")
                assert Fraction(actual[field + "_delta"]) == x-y, (line, field)
                out[field + "_delta"] = actual[field + "_delta"]
                numeric_checks += 1
            out[field + "_relation"] = relation
            compare_counts[field][relation] += 1
    assert {k: str(out.get(k, "")) for k in saved.columns} == actual, line
    if row["sequence"] in inverse:
        standard_links.append({"standard": inverse[row["sequence"]], **out})
    verify_membership("aligned_quantities", row["precursor"], row["sequence"], row["charge"], "0",
                      "PeptideGroupLabel", row["group_label"], aligned_asset, line)
equivalent(aligned_exceptions, read("aligned-library-exceptions.json"))
equivalent(standard_links, read("standard-links.json"))
summary = dict(link_states=links, aligned_library_states=lib_states, cross_states=cross,
               comparison_states=compare_counts, n_aligned_rows=len(aligned), candidate_library_states=score_states,
               n_scored_rows=len(scores), numeric_tolerance=dict(atol=C["atol"], rtol=C["rtol"]))
equivalent(summary, read("lineage-summary.json"))
standard_memberships = []
for name, form in forms.items():
    members = [row for row in library if row["ModifiedPeptideSequence"] == form]
    standard_memberships.append(dict(standard=name, expected_form=form, expected_group_form=labels[name],
                                     library_memberships=members,
                                     bare_sequence_memberships=[row for row in library if row["PeptideSequence"] == name],
                                     n_target_transitions=sum(r["n_transitions"] for r in members if r["Decoy"] == "0"),
                                     n_decoy_transitions=sum(r["n_transitions"] for r in members if r["Decoy"] == "1"),
                                     n_scored_candidates=int((scores["FullPeptideName"] == form).sum()),
                                     n_aligned_slots=int((aligned["sequence"] == form).sum())))
equivalent(standard_memberships, read("standard-memberships.json"))
mapping_rows = [{"role": key[0], "scored_or_aligned_group_id": key[1], "library_group_id": key[2],
                 "form": key[3], "charge": key[4], "decoy": key[5], "n_source_rows": count}
                for key, count in sorted(mappings.items())]
equivalent(mapping_rows, read("namespace-mapping.json"))
equivalent(member_exceptions, read("membership-exceptions.json"))
membership_summary = dict(counts=member_states, n_mapping_entries=len(mapping_rows),
                          n_ambiguous_library_form_keys=sum(len(v)>1 for v in by_form.values()),
                          semantics="Exact full-form/charge/target-state membership; original ID namespaces remain distinct")
equivalent(membership_summary, read("membership-summary.json"))
result = dict(passed=True, parser="pandas C parser with literal strings; no audit imports",
              arithmetic="fractions.Fraction versus audit decimal.Decimal",
              n_tables=len(inventories), n_source_rows=sum(r["n_rows"] for r in inventories),
              n_literal_cells=sum(r["n_cells"] for r in inventories), n_library_memberships=len(library),
              n_scored_candidates=len(scores), n_aligned_links=len(aligned),
              n_numeric_comparisons=numeric_checks, n_standard_slots=len(standard_links),
              original_namespace_hypothesis_replayed=True, amended_membership_replayed=True,
              original_native_outer_partitions_accessed=False)
(P / "results/verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
