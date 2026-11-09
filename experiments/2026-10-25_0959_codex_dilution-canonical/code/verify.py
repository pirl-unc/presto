"""Independent literal, relational and rational replay of the original sources.

Hitlist is imported only for the final public-store query checks. This verifier
does not import compiler, schema, identifier or numeric helpers.
"""

from collections import Counter, defaultdict
import csv
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sys

from replay_support import (
    MEASURES,
    MISSING,
    PRIMARY,
    Replay,
    digest,
    exhausted,
    literal_rows,
    measures,
    metadata,
    numeric,
    relation,
)
from verify_metadata import verify_metadata

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / "reproduce/launch.json").read_text())
CONTRACT = json.loads((ROOT / "reproduce/source-contract.json").read_text())
PACKAGE = Path(C["package"])
PRIOR = Path(C["prior_audit"])
rep = Replay(C, CONTRACT)

manifest = json.loads((PACKAGE / "manifest.json").read_text())
assert (
    manifest["schema_version"],
    manifest["catalog_schema_version"],
    manifest["package_family"],
    manifest["revision"],
) == (28, 37, "dilution_observer", "pxd024809_dilution_observer_v1")
assert json.loads((PACKAGE / "dilution_observer/contract.json").read_text()) == CONTRACT
inventory = {
    str(p.relative_to(PACKAGE)): {"sha256": digest(p), "size_bytes": p.stat().st_size}
    for p in PACKAGE.rglob("*")
    if p.is_file() and p.name != "manifest.json"
}
assert inventory == manifest["files"]
assert manifest["source_registry"] == CONTRACT["source_registry"]
asset_rows = {r["source_file"]: r for r in rep.rows("assets")}
assert set(asset_rows) == set(C["assets"]) and len(asset_rows) == 17
for asset in CONTRACT["assets"]:
    name = asset["filename"]
    original = Path(C["assets"][name])
    snapshot = PACKAGE / "dilution_observer/source_assets" / name
    assert digest(original) == digest(snapshot) == asset["sha256"]
    assert original.stat().st_size == snapshot.stat().st_size == asset["size_bytes"]
    assert (original.stat().st_dev, original.stat().st_ino) != (
        snapshot.stat().st_dev,
        snapshot.stat().st_ino,
    )
    rep.check(
        "assets",
        asset_rows[name],
        {
            "source_file": name,
            "source_asset_url": asset["source_url"],
            "source_asset_role": asset["role"],
            "asset_sha256": asset["sha256"],
            "size_bytes": asset["size_bytes"],
            "license_status": asset["license_status"],
        },
    )

meta, project, source_inventory, level_cells, paragraphs, commands = metadata(C, CONTRACT)
records = rep.rows("source_records")
meta_records = {}
for _ in range(len(meta)):
    row = next(records)
    locator = row["source_locator"]
    assert locator not in meta_records
    filename, headers, values = meta[locator]
    rep.check(
        "source_records",
        row,
        {
            "source_asset_id": asset_rows[filename]["source_asset_id"],
            "source_file": filename,
            "source_locator": locator,
            "source_row_json": json.dumps(
                {"headers": headers, "values": values}, ensure_ascii=True
            ),
        },
    )
    meta_records[locator] = row
assert set(meta_records) == set(meta)

acq_list = list(rep.rows("acquisitions"))
acquisitions = {r["source_run_label"]: r for r in acq_list}
assert len(acq_list) == len(acquisitions) == 10
lib_fields = (
    "native_library_group_id",
    "modified_sequence_raw",
    "peptide_sequence",
    "charge_raw",
    "decoy_raw",
    "group_label_raw",
)
source_lib_fields = (
    "TransitionGroupId",
    "ModifiedPeptideSequence",
    "PeptideSequence",
    "PrecursorCharge",
    "Decoy",
    "PeptideGroupLabel",
)
lib_rows = list(rep.rows("library_precursors"))
library = {tuple(r[k] for k in lib_fields): r for r in lib_rows}
assert len(library) == len(lib_rows) == 12851
by_form = defaultdict(list)
for row in library.values():
    by_form[(row["modified_sequence_raw"], row["charge_raw"], row["decoy_raw"])].append(row)
assert all(len(members) == 1 for members in by_form.values())


def membership(form, charge, decoy, *, sequence=None, label=None):
    matches = by_form.get((form, charge, decoy), [])
    assert len(matches) == 1, "The independently qualified real source has unique form membership"
    lib = matches[0]
    result = {
        "n_library_candidates": 1,
        "library_precursor_id": lib["library_precursor_id"],
        "library_link_status": "matched",
    }
    if sequence is not None:
        assert sequence == lib["peptide_sequence"]
    if label is not None:
        other = lib["group_label_raw"]
        result["library_label_status"] = (
            "same_literal"
            if label == other
            else "both_missing_different_literal"
            if label in MISSING and other in MISSING
            else "different"
        )
    return result, lib


transitions = rep.rows("library_transitions")
scores = rep.rows("scored_features")
aligned = rep.rows("aligned_quantities")
links = rep.rows("feature_links")
comparisons = rep.rows("comparisons")
lib_counts, lib_first, first_scores, features = Counter(), {}, {}, {}
label_counts, link_counts, comparison_counts = Counter(), Counter(), defaultdict(Counter)
saved_stream = gzip.open(PRIOR / "results/aligned-feature-links.tsv.gz", "rt", newline="")
saved_links = csv.DictReader(saved_stream, delimiter="\t")
prior_inventory = {
    r["asset"]: r for r in json.loads((PRIOR / "results/table-inventory.json").read_text())
}
source_order = (
    [CONTRACT["library"]]
    + sorted(a["filename"] for a in CONTRACT["assets"] if a["role"] == "per_run_scored_candidates")
    + [CONTRACT["aligned"]]
)
roles = {a["filename"]: a["role"] for a in CONTRACT["assets"]}
source_checks = []
n_comparisons = 0
for filename in source_order:
    row_hash = hashlib.sha256()
    n_rows, n_cells = 0, 0
    role = roles[filename]
    for index, header, cells in literal_rows(C["assets"][filename]):
        assert header == prior_inventory[filename]["headers"] == CONTRACT["headers"][role]
        if index == 1:
            row_hash.update((json.dumps(header, ensure_ascii=True) + "\n").encode())
        row_hash.update((json.dumps(cells, ensure_ascii=True) + "\n").encode())
        n_rows += 1
        n_cells += len(cells)
        original = dict(zip(header, cells))
        record = next(records)
        rep.check(
            "source_records",
            record,
            {
                "source_asset_id": asset_rows[filename]["source_asset_id"],
                "source_file": filename,
                "source_locator": f"{filename}:line:{index + 1}",
                "source_row_index": index,
                "source_row_json": json.dumps(
                    {"headers": header, "values": cells}, ensure_ascii=True
                ),
            },
        )
        record_id = record["source_record_id"]
        if role == "master_query_library":
            key = tuple(original[k] for k in source_lib_fields)
            lib_counts[key] += 1
            lib_first.setdefault(key, (record_id, index + 1))
            lib = library[key]
            rep.check(
                "library_transitions",
                next(transitions),
                {
                    "source_record_id": record_id,
                    "library_precursor_id": lib["library_precursor_id"],
                    "native_transition_id": original["TransitionId"],
                    "product_mz_raw": original["ProductMz"],
                    "product_mz": numeric(original["ProductMz"])[0],
                    "product_charge_raw": original["ProductCharge"],
                    "library_intensity_raw": original["LibraryIntensity"],
                    "library_intensity": numeric(original["LibraryIntensity"])[0],
                },
            )
        elif role == "per_run_scored_candidates":
            run = original["filename"][:-5]
            first_scores.setdefault(run, (record_id, original["run_id"]))
            assert first_scores[run][1] == original["run_id"]
            assert original["filename"].replace("#", "_") + ".tsv" == filename
            member, _ = membership(
                original["FullPeptideName"],
                original["Charge"],
                original["decoy"],
                sequence=original["Sequence"],
            )
            row = next(scores)
            rep.check(
                "scored_features",
                row,
                {
                    "source_record_id": record_id,
                    "acquisition_id": acquisitions[run]["acquisition_id"],
                    "source_run_id": original["run_id"],
                    "native_feature_id": original["id"],
                    "native_precursor_id": original["transition_group_id"],
                    "peptide_sequence": original["Sequence"],
                    "modified_sequence_raw": original["FullPeptideName"],
                    "charge_raw": original["Charge"],
                    "decoy_raw": original["decoy"],
                    "observation_kind": "exported_scored_feature_candidate",
                    "dilution_d_score_raw": original["d_score"],
                    "dilution_d_score": numeric(original["d_score"])[0],
                    **measures(original, True),
                    **member,
                },
            )
            key = run, original["id"]
            assert key not in features
            features[key] = filename, index + 1, row
        else:
            run = original["run"]
            member, lib = membership(
                original["sequence"], original["charge"], "0", label=original["group_label"]
            )
            label_counts[member["library_label_status"]] += 1
            row = next(aligned)
            measurement_id = row["measurement_id"]
            rep.check(
                "aligned_quantities",
                row,
                {
                    "source_record_id": record_id,
                    "acquisition_id": acquisitions[run]["acquisition_id"],
                    "native_feature_id": original["feature_id"],
                    "native_precursor_id": original["precursor"],
                    "native_peptide_id": original["peptide_id"],
                    "peptide_sequence": lib["peptide_sequence"],
                    "modified_sequence_raw": original["sequence"],
                    "charge_raw": original["charge"],
                    "decoy_raw": "0",
                    "group_label_raw": original["group_label"],
                    "observation_kind": "aligned_precursor_run_slot",
                    "dilution_alignment_rank_raw": original["alignment_rank"],
                    "dilution_alignment_rank": numeric(original["alignment_rank"])[0],
                    **measures(original, False),
                    **member,
                },
            )
            saved = next(saved_links)
            assert int(saved["aligned_source_line"]) == index + 1
            for field in ("run", "precursor", "feature_id", "sequence", "charge", "group_label"):
                assert saved[field] == original[field]
            candidate = (
                features.get((run, original["feature_id"]))
                if original["feature_id"] not in MISSING
                else None
            )
            expected = {
                "source_record_id": record_id,
                "measurement_id": measurement_id,
                "n_candidates": int(candidate is not None),
            }
            if candidate is None:
                state = (
                    "missing_feature_id"
                    if original["feature_id"] in MISSING
                    else "unmatched_feature_id"
                )
            else:
                scored_file, scored_line, scored = candidate
                assert (
                    saved["scored_asset"] == scored_file
                    and int(saved["scored_source_line"]) == scored_line
                )
                checks = {
                    "form_agrees": original["sequence"] == scored["modified_sequence_raw"],
                    "charge_agrees": original["charge"] == scored["charge_raw"],
                    "precursor_agrees": original["precursor"] == scored["native_precursor_id"],
                    "target_agrees": scored["decoy_raw"] == "0",
                }
                state = "matched_identity" if all(checks.values()) else "id_crosscheck_disagreement"
                assert state == "matched_identity"
                expected.update(identification_id=scored["identification_id"], **checks)
                for field, (raw, _) in MEASURES.items():
                    a, b = original[field], scored[raw]
                    status, delta = relation(
                        a, b, CONTRACT["comparison_atol"], CONTRACT["comparison_rtol"]
                    )
                    comparison = next(comparisons)
                    assert (
                        delta
                        == Fraction(comparison["exact_decimal_delta"])
                        == Fraction(saved[field + "_delta"])
                    )
                    assert saved[field + "_aligned"] == a and saved[field + "_scored"] == b
                    assert saved[field + "_relation"] == status
                    rep.check(
                        "comparisons",
                        comparison,
                        {
                            "source_record_id": record_id,
                            "measurement_id": measurement_id,
                            "identification_id": scored["identification_id"],
                            "compared_source_field": field,
                            "aligned_literal": a,
                            "scored_literal": b,
                            "comparison_status": status,
                            "exact_decimal_delta": comparison["exact_decimal_delta"],
                        },
                    )
                    comparison_counts[field][status] += 1
                    n_comparisons += 1
            assert (
                saved["link_state"] == state
                and int(saved["n_candidates"]) == expected["n_candidates"]
            )
            expected["feature_link_status"] = state
            rep.check("feature_links", next(links), expected)
            link_counts[state] += 1
    assert (n_rows, n_cells, row_hash.hexdigest()) == (
        prior_inventory[filename]["n_rows"],
        prior_inventory[filename]["n_cells"],
        prior_inventory[filename]["literal_rows_sha256"],
    )
    source_checks.append(
        {
            "source_file": filename,
            "n_rows": n_rows,
            "n_cells": n_cells,
            "literal_rows_sha256": row_hash.hexdigest(),
        }
    )
    print(json.dumps(source_checks[-1]), flush=True)
for iterator in (records, transitions, scores, aligned, links, comparisons, saved_links):
    exhausted(iterator)
saved_stream.close()

for key, row in library.items():
    rep.check(
        "library_precursors",
        row,
        {
            **dict(zip(lib_fields, key)),
            "source_record_id": lib_first[key][0],
            "n_transitions": lib_counts[key],
        },
    )
with gzip.open(PRIOR / "results/library-memberships.json.gz", "rt") as stream:
    previous_library = json.load(stream)
independent_library = [
    {
        **dict(zip(source_lib_fields, key)),
        "n_transitions": lib_counts[key],
        "first_source_line": lib_first[key][1],
    }
    for key in sorted(library)
]
assert independent_library == previous_library
previous_lineage = json.loads((PRIOR / "results/lineage-summary.json").read_text())
assert dict(link_counts) == previous_lineage["link_states"]
assert dict(comparison_counts) == previous_lineage["comparison_states"]
assert dict(label_counts) == {"same_literal": 64060, "both_missing_different_literal": 10}
assert n_comparisons == 297876
meta_result = verify_metadata(
    rep,
    meta_records,
    project,
    source_inventory,
    level_cells,
    paragraphs,
    commands,
    acquisitions,
    first_scores,
    library,
)
assert set(rep.visited) == set(PRIMARY)
assert dict(rep.counts) == manifest["reconciliation"]

# The independent checks above establish source semantics. The public client
# checks below establish that downstream consumers can reach the verified data.
sys.path.insert(0, C["hitlist_checkout"])
from hitlist import EvidenceStore

store = EvidenceStore.open(PACKAGE)
query_checks = {}
for name, joins in {
    "aligned_quantities": ["source_record", "acquisition", "library_precursor"],
    "feature_links": ["aligned", "scored"],
    "acquisitions": ["nominal_level", "context"],
    "standard_memberships": ["standard", "library_precursor"],
}.items():
    output = store.query("dilution_" + name, joins=joins, limit=3).table
    assert output.num_rows == 3
    query_checks[name] = {"n_rows": output.num_rows, "columns": output.column_names}
result = {
    "passed": True,
    "table_counts": dict(rep.counts),
    "source_tables": source_checks,
    "n_tabular_rows": sum(r["n_rows"] for r in source_checks),
    "n_literal_cells": sum(r["n_cells"] for r in source_checks),
    "n_numeric_comparisons": n_comparisons,
    "feature_link_states": dict(link_counts),
    "library_label_states": dict(label_counts),
    "metadata_checks": meta_result,
    "public_queries": query_checks,
    "parser": "pandas literal strings, compared with independently verified0844digests",
    "arithmetic": "fractions.Fraction",
    "compiler_helpers_imported": False,
    "training_performed": False,
    "original_native_outer_partitions_accessed": False,
}
(ROOT / "results/verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
