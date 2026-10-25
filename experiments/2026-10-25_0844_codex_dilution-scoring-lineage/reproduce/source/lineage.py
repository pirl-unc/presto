"""Exact source-ID lineage, with all aligned slots and unresolved states retained."""

from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / "reproduce/lineage-launch.json").read_text())
MISSING = {"", "NA", "NaN", "nan", "null"}
ATOL, RTOL = Decimal(C["atol"]), Decimal(C["rtol"])
FIELDS = C["compare_fields"]


def write(name, value):
    (ROOT / "results" / name).write_text(json.dumps(value, indent=2) + "\n")


def compare(a, b):
    if a in MISSING or b in MISSING:
        return ("both_missing" if a in MISSING and b in MISSING else
                "aligned_missing" if a in MISSING else "scored_missing"), ""
    x, y = Decimal(a), Decimal(b)
    if not x.is_finite() or not y.is_finite():
        return "nonfinite", ""
    delta = x - y
    kind = ("same_literal" if a == b else "decimal_equal" if x == y else
            "within_numeric_tolerance" if abs(delta) <= ATOL + RTOL * max(abs(x), abs(y))
            else "different")
    return kind, str(delta)


def state(value):
    if value in MISSING:
        return "missing:" + value
    return "positive" if Decimal(value) > 0 else "negative" if Decimal(value) < 0 else "zero"


library = json.loads((ROOT / "results/library-memberships.json").read_text())
by_group = defaultdict(list)
for row in library:
    by_group[row["TransitionGroupId"]].append(row)


def library_check(group, form, charge, decoy, label=None):
    matches = by_group[group]
    if not matches:
        return "absent_group"
    if len(matches) != 1:
        return "ambiguous_group"
    lib = matches[0]
    errors = []
    for field, expected in [("ModifiedPeptideSequence", form), ("PrecursorCharge", charge),
                            ("Decoy", decoy)]:
        if lib[field] != expected:
            errors.append(field)
    if label is not None and lib["PeptideGroupLabel"] != label:
        errors.append("PeptideGroupLabel")
    return "matched" if not errors else "disagrees:" + ",".join(errors)


score_index = defaultdict(list)
candidate_library_states = Counter()
candidate_exceptions = []
score_fields = {"id", "filename", "FullPeptideName", "Charge", "transition_group_id", "decoy"}
score_fields.update(FIELDS.values())
for asset in C["tabular_assets"]:
    if asset["role"] != "per_run_scored_candidates":
        continue
    with Path(asset["path"]).open(newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream, delimiter="\t"), 2):
            assert row["filename"].endswith(".mzML")
            run = row["filename"][:-5]
            check = library_check(row["transition_group_id"], row["FullPeptideName"],
                                  row["Charge"], row["decoy"])
            candidate_library_states[check] += 1
            if check != "matched":
                candidate_exceptions.append({"asset": Path(asset["path"]).name,
                                             "source_line": line, "id": row["id"], "state": check})
            kept = {k: row[k] for k in score_fields}
            kept.update(source_line=line, asset=Path(asset["path"]).name)
            score_index[(run, row["id"])].append(kept)

panel = json.loads(Path(C["prior_audit"]).read_text())["nominal_standards"]
standard_forms = {r["canonical_candidate"]: r["reported_form"].replace("[L(13C6; 15N)]", "L(UniMod:695)")
                  for r in panel}
standard_labels = {r["canonical_candidate"]: r["reported_form"].replace("[L(13C6; 15N)]", "L(Label:13C(6)15N(1))")
                   for r in panel}
inverse_forms = {value: key for key, value in standard_forms.items()}
summary = {"link_states": Counter(), "aligned_library_states": Counter(),
           "cross_states": Counter(), "comparison_states": defaultdict(Counter),
           "n_aligned_rows": 0}
standard_links, aligned_exceptions = [], []
aligned = next(a for a in C["tabular_assets"] if a["role"] == "aligned_quantities")
fieldnames = ["aligned_source_line", "run", "precursor", "feature_id", "sequence", "charge", "group_label",
              "link_state", "n_candidates", "scored_asset", "scored_source_line", "form_agrees",
              "charge_agrees", "precursor_agrees", "target_agrees", "library_state"]
for field in FIELDS:
    fieldnames.extend([field + "_aligned", field + "_scored", field + "_relation", field + "_delta"])
with (ROOT / "results/aligned-feature-links.tsv.gz").open("xb") as binary:
    with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            with Path(aligned["path"]).open(newline="") as stream:
                for line, row in enumerate(csv.DictReader(stream, delimiter="\t"), 2):
                    matches = [] if row["feature_id"] in MISSING else score_index[(row["run"], row["feature_id"])]
                    out = {k: row[k] for k in ["run", "precursor", "feature_id", "sequence", "charge", "group_label"]}
                    out.update(aligned_source_line=line, n_candidates=len(matches))
                    if row["feature_id"] in MISSING:
                        status = "missing_feature_id"
                    elif not matches:
                        status = "unmatched_feature_id"
                    elif len(matches) > 1:
                        status = "ambiguous_feature_id"
                    else:
                        match = matches[0]
                        checks = {"form_agrees": row["sequence"] == match["FullPeptideName"],
                                  "charge_agrees": row["charge"] == match["Charge"],
                                  "precursor_agrees": row["precursor"] == match["transition_group_id"],
                                  "target_agrees": match["decoy"] == "0"}
                        out.update({k: str(v).lower() for k, v in checks.items()})
                        out.update(scored_asset=match["asset"], scored_source_line=match["source_line"])
                        status = "matched_identity" if all(checks.values()) else "id_crosscheck_disagreement"
                    out["link_state"] = status
                    libstate = library_check(row["precursor"], row["sequence"], row["charge"], "0", row["group_label"])
                    out["library_state"] = libstate
                    summary["aligned_library_states"][libstate] += 1
                    if libstate != "matched":
                        aligned_exceptions.append({"source_line": line, "state": libstate})
                    summary["link_states"][status] += 1
                    summary["n_aligned_rows"] += 1
                    summary["cross_states"]["|".join([status, state(row["intensity"]), state(row["m_score"])])] += 1
                    for field, score_field in FIELDS.items():
                        out[field + "_aligned"] = row[field]
                        if len(matches) == 1:
                            b = matches[0][score_field]
                            relation, delta = compare(row[field], b)
                            out[field + "_scored"] = b
                            out[field + "_relation"] = relation
                            out[field + "_delta"] = delta
                            summary["comparison_states"][field][relation] += 1
                    writer.writerow(out)
                    if row["sequence"] in inverse_forms:
                        standard_links.append({"standard": inverse_forms[row["sequence"]], **out})

standard_memberships = []
for name, form in standard_forms.items():
    members = [row for row in library if row["ModifiedPeptideSequence"] == form]
    bare_members = [row for row in library if row["PeptideSequence"] == name]
    standard_memberships.append({"standard": name, "expected_form": form,
                                 "expected_group_form": standard_labels[name],
                                 "library_memberships": members, "bare_sequence_memberships": bare_members,
                                 "n_target_transitions": sum(r["n_transitions"] for r in members if r["Decoy"] == "0"),
                                 "n_decoy_transitions": sum(r["n_transitions"] for r in members if r["Decoy"] == "1"),
                                 "n_scored_candidates": sum(r["FullPeptideName"] == form for matches in score_index.values() for r in matches),
                                 "n_aligned_slots": sum(r["standard"] == name for r in standard_links)})
write("lineage-summary.json", {**summary, "candidate_library_states": candidate_library_states,
                               "n_scored_rows": sum(len(v) for v in score_index.values()),
                               "numeric_tolerance": {"atol": C["atol"], "rtol": C["rtol"]}})
write("candidate-library-exceptions.json", candidate_exceptions)
write("aligned-library-exceptions.json", aligned_exceptions)
write("standard-memberships.json", standard_memberships)
write("standard-links.json", standard_links)
print(json.dumps(summary), flush=True)
