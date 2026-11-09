"""Independent source/recipe audit. No Hitlist imports and no held-out metrics."""

import csv
import gc
import hashlib
import importlib.util
import json
import math
import resource
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
PACKAGE = Path(CFG["output_package"])
OLD = Path(CFG["old_package"])
SOURCE = PACKAGE / "external/wolflevy2018_controlled"
RECIPE = Path(CFG["recipe_output"])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return pq.read_table(SOURCE / "typed" / (name + ".parquet")).to_pylist()


def index(rows, field):
    result = {row[field]: row for row in rows}
    assert len(result) == len(rows)
    return result


def subset(actual, expected):
    for key, value in expected.items():
        assert actual[key] == value, (key, actual[key], value)


def number(value):
    return None if value in ("", "NaN") else float(value)


def connected(peptides, links, identical=False):
    """Independent bipartite traversal; producer uses union-find."""
    neighbors, owners = defaultdict(set), defaultdict(set)
    for pid, parent in links:
        neighbors[pid].add(parent)
        owners[parent].add(pid)
    if identical:
        for pid, row in peptides.items():
            parent = "sequence:" + row["sequence_raw"]
            neighbors[pid].add(parent)
            owners[parent].add(pid)
    assignments = {}
    for pid in sorted(peptides):
        if pid in assignments:
            continue
        stack, members, visited = [pid], {pid}, set()
        while stack:
            node = stack.pop()
            for parent in neighbors[node] - visited:
                visited.add(parent)
                for other in owners[parent] - members:
                    members.add(other)
                    stack.append(other)
        for member in members:
            assignments[member] = min(members)
    return assignments


def main(report):
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    old_manifest = json.loads((OLD / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        path = PACKAGE / name
        assert (
            path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"]
        ), name
    for name, item in old_manifest["files"].items():
        assert manifest["files"][name] == item and sha(OLD / name) == item["sha256"], (
            name
        )
    report["inventory"] = {
        "files_verified": len(manifest["files"]),
        "old_files_unchanged": len(old_manifest["files"]),
    }
    pins = json.loads((SOURCE / "contracts/source_assets.json").read_text())
    for name, item in pins["assets"].items():
        assert sha(SOURCE / "assets" / name) == item["sha256"]
        assert (SOURCE / "assets" / name).stat().st_size == item["size"]
    queries = json.loads((ROOT / "results/controlled-queries.json").read_text())
    for name, query in queries.items():
        digest, count = hashlib.sha256(), 0
        for batch in pq.ParquetFile(PACKAGE / query["artifact"]).iter_batches():
            for row in batch.to_pylist():
                digest.update(
                    (
                        json.dumps(
                            row, sort_keys=True, separators=(",", ":"), allow_nan=False
                        )
                        + "\n"
                    ).encode()
                )
                count += 1
        assert (count, digest.hexdigest()) == (
            query["n_rows"],
            query["canonical_rows_sha256"],
        ), name
    report["canonical_queries"] = {
        "n_relations": len(queries),
        "n_mixed_grain_rows": sum(x["n_rows"] for x in queries.values()),
    }
    print("Inventory, old source bytes and all canonical queries verified", flush=True)

    originals, expected_records = {}, {}
    for project in ("PXD009508", "PXD007263"):
        originals[project] = {}
        for name in ("summary", "peptides", "evidence", "proteinGroups", "parameters"):
            path = (
                Path(CFG["original_control_assets"])
                / project
                / "extracted/txt"
                / (name + ".txt")
            )
            assert sha(path) == sha(SOURCE / "assets" / project / "txt" / path.name)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            originals[project][name] = rows
            for line, row in enumerate(rows, 2):
                expected_records[f"{project.lower()}:{name}:row:{line}"] = (
                    project,
                    name,
                    line,
                    row,
                )
    seen = set()
    for batch in pq.ParquetFile(SOURCE / "typed/source_records.parquet").iter_batches(
        batch_size=1000
    ):
        for row in batch.to_pylist():
            key = row["source_record_id"]
            assert key not in seen
            seen.add(key)
            project, name, line, raw = expected_records[key]
            subset(
                row,
                {
                    "source_dataset": project,
                    "asset_path": f"{project}/txt/{name}.txt",
                    "row_1based": line,
                },
            )
            assert json.loads(row["fields_json"]) == raw
    assert seen == set(expected_records)
    report["source_records"] = len(seen)
    del expected_records
    peptides = index(read("peptide_rows"), "peptide_row_id")
    channels = index(read("channels"), "channel_id")
    groups = index(read("groups"), "group_record_id")
    evidence = index(read("evidence"), "evidence_id")
    assertions = read("source_assertions")
    aliases = index(read("label_aliases"), "channel_id")
    raw_peptides, evidence_values, evidence_ids, counts = (
        {},
        defaultdict(list),
        defaultdict(set),
        defaultdict(Counter),
    )
    for project, tables in originals.items():
        sid = project.lower()
        for line, row in enumerate(tables["peptides"], 2):
            key = f"{sid}:peptide:{row['id']}"
            raw_peptides[key] = row
            subset(
                peptides[key],
                {
                    "source_record_id": f"{sid}:peptides:row:{line}",
                    "source_dataset": project,
                    "peptide_id_raw": row["id"],
                    "sequence_raw": row["Sequence"],
                    "parent_labels_raw": row["Proteins"],
                    "leading_parent_raw": row["Leading razor protein"],
                    "group_ids_raw": row["Protein group IDs"],
                    "n_flank_raw": row["Amino acid before"],
                    "c_flank_raw": row["Amino acid after"],
                    "start_1based_reported": number(row["Start position"]),
                    "end_1based_reported": number(row["End position"]),
                    "pep": number(row["PEP"]),
                    "pep_raw": row["PEP"],
                    "reverse": row["Reverse"] == "+",
                    "potential_contaminant": row["Potential contaminant"] == "+",
                },
            )
        for line, row in enumerate(tables["proteinGroups"], 2):
            key = f"{sid}:proteinGroups:row:{line}"
            subset(
                groups[key],
                {
                    "source_record_id": key,
                    "group_id_raw": row["id"],
                    "protein_ids_raw": row["Protein IDs"],
                    "peptide_ids_raw": row["Peptide IDs"],
                    "reverse": row["Reverse"] == "+",
                    "potential_contaminant": row["Potential contaminant"] == "+",
                },
            )
        for line, row in enumerate(tables["summary"], 2):
            if not row["Experiment"]:
                continue
            experiment = row["Experiment"]
            key = f"{sid}:run:{experiment}"
            channel = channels[key]
            subset(
                channel,
                {
                    "source_record_id": f"{sid}:summary:row:{line}",
                    "raw_file": row["Raw file"],
                    "experiment_raw": experiment,
                    "search_enzyme_mode_raw": row["Enzyme mode"],
                    "source_dataset": project,
                    "measured_species": "Homo sapiens",
                    "measured_cell": "HEK293",
                    "specimen_id": None,
                    "aliquot_id": None,
                    "pair_id": None,
                    "capture_target_gene": None,
                },
            )
            alias = aliases[key]
            column = alias["quantity_column_raw"]
            literal = column[len("Intensity ") :]
            assert literal.translate(
                str.maketrans("", "", "_-")
            ) == experiment.translate(str.maketrans("", "", "_-"))
            assert column in tables["peptides"][0]
            subset(
                alias,
                {
                    "summary_experiment_raw": experiment,
                    "source_dataset": project,
                    "match_rule": "literal_equal"
                    if literal == experiment
                    else "unique_full_identifier_without_hyphen_underscore",
                },
            )
            pieces = experiment.split("_")
            if project == "PXD009508":
                capture, treatment, rep = pieces[2], pieces[3][:-1], int(pieces[3][-1])
                subset(
                    channel,
                    {
                        "capture_label_raw": capture,
                        "capture_role": "candidate_proteasome_capture"
                        if capture == "alpha6"
                        else "capture_control",
                        "treatment_label_raw": treatment,
                        "treatment_name": "epoxomicin"
                        if treatment == "epoxo"
                        else None,
                        "treatment_dose": 1.0 if treatment == "epoxo" else None,
                        "duration_hours": 4.0 if treatment == "epoxo" else None,
                        "replicate_label": rep,
                        "time_hours": None,
                        "treatment_context_status": "run_label_with_publication_level_dose_duration",
                    },
                )
            else:
                subset(
                    channel,
                    {
                        "time_hours": float(pieces[-2]),
                        "replicate_label": int(pieces[-1]),
                        "treatment_name": None,
                        "treatment_dose": None,
                        "treatment_context_status": "literal_TNF_vs_publication_TNFalpha_plus_IFNgamma",
                    },
                )
        for line, row in enumerate(tables["evidence"], 2):
            key, pid, cid = (
                f"{sid}:evidence:{row['id']}",
                f"{sid}:peptide:{row['Peptide ID']}",
                f"{sid}:run:{row['Experiment']}",
            )
            value = number(row["Intensity"])
            subset(
                evidence[key],
                {
                    "source_record_id": f"{sid}:evidence:row:{line}",
                    "peptide_row_id": pid,
                    "channel_id": cid,
                    "sequence_raw": row["Sequence"],
                    "modified_sequence_raw": row["Modified sequence"],
                    "modifications_raw": row["Modifications"],
                    "type_raw": row["Type"],
                    "group_ids_raw": row["Protein group IDs"],
                    "intensity_raw": row["Intensity"],
                    "intensity": value,
                    "intensity_status": "unreported" if value is None else "reported",
                    "msms_count": number(row["MS/MS Count"]),
                    "pep_raw": row["PEP"],
                    "pep": number(row["PEP"]),
                    "reverse": row["Reverse"] == "+",
                    "potential_contaminant": row["Potential contaminant"] == "+",
                },
            )
            assert (
                row["Raw file"] == channels[cid]["raw_file"]
                and row["Sequence"] == raw_peptides[pid]["Sequence"]
            )
            pair = (pid, cid)
            evidence_ids[pair].add(key)
            direct = bool(number(row["MS/MS Count"]) and row["Type"] != "MULTI-MATCH")
            counts[pair].update(
                {
                    "n_source_evidence": 1,
                    "n_numeric_evidence": int(value is not None),
                    "n_direct_msms_evidence": int(direct),
                    "n_positive_direct_msms_evidence": int(
                        direct and value is not None and value > 0
                    ),
                    "n_matched_evidence": int(row["Type"] == "MULTI-MATCH"),
                }
            )
            if value is not None:
                evidence_values[pair].append(value)
        reported_parameters = {
            (r["predicate"], r["value_raw"])
            for r in assertions
            if r["source_dataset"] == project
            and r["scope"] == "deposited_search_parameters"
        }
        assert reported_parameters == {
            (r["Parameter"], r["Value"]) for r in tables["parameters"]
        }
    assert set(peptides) == set(raw_peptides)
    assert len(groups) == sum(len(t["proteinGroups"]) for t in originals.values())
    assert len(evidence) == sum(len(t["evidence"]) for t in originals.values())
    report["metadata"] = {
        "peptides": len(peptides),
        "runs": len(channels),
        "evidence": len(evidence),
        "missing_evidence_intensities": sum(
            r["intensity"] is None for r in evidence.values()
        ),
        "source_groups": len(groups),
        "aliases": len(aliases),
    }
    del originals, evidence, groups
    measurements = index(read("quantity_measurements"), "measurement_id")
    expected_edges, n_results = set(), 0
    for result in read("quantity_results"):
        rid, pid, cid = (
            result["result_id"],
            result["peptide_row_id"],
            result["channel_id"],
        )
        n_results += 1
        pair = (pid, cid)
        subset(result, counts[pair])
        assert all(result[k] == counts[pair][k] for k in result if k.startswith("n_"))
        reported = float(raw_peptides[pid][aliases[cid]["quantity_column_raw"]])
        values = evidence_values[pair]
        total = math.fsum(values)
        assert math.isclose(reported, total, rel_tol=1e-4, abs_tol=1)
        actual = measurements[rid + ":reported_run_intensity"]
        subset(
            actual,
            {
                "value": reported,
                "value_status": "reported_positive"
                if reported > 0
                else "reported_zero_not_absence",
            },
        )
        assert math.isclose(
            actual["recomputed_value"], total, rel_tol=1e-12, abs_tol=1e-6
        )
        summed = measurements[rid + ":evidence_sum"]
        status = (
            "no_reported_evidence"
            if not evidence_ids[pair]
            else "all_evidence_intensities_unreported"
            if not values
            else "partial_evidence_intensity_sum"
            if len(values) < len(evidence_ids[pair])
            else "reported_evidence_sum"
        )
        assert summed["value_status"] == status
        assert (
            summed["value"] is None
            if not values
            else math.isclose(summed["value"], total, rel_tol=1e-12)
        )
        expected_edges.add(
            (
                rid + ":reported_run_intensity",
                rid + ":evidence_sum",
                "arithmetic_replay",
            )
        )
        expected_edges.update(
            (rid + ":evidence_sum", eid, "source_evidence_intensity")
            for eid in evidence_ids[pair]
        )
    actual_edges = read("derivation_inputs")
    assert len(actual_edges) == len(expected_edges)
    assert {
        (r["derived_measurement_id"], r["input_measurement_id"], r["input_role"])
        for r in actual_edges
    } == expected_edges
    assert n_results == 146832 and len(measurements) == 2 * n_results
    report["quantities"] = {
        "results": n_results,
        "measurements": len(measurements),
        "derivation_edges": len(actual_edges),
        "arithmetic_failures": 0,
    }
    del (
        measurements,
        actual_edges,
        expected_edges,
        evidence_values,
        evidence_ids,
        counts,
        raw_peptides,
    )
    gc.collect()
    print(
        "Original rows, metadata, all quantities and contributors verified", flush=True
    )
    reference_checks = verify_references(peptides)
    report["references"] = reference_checks
    verify_recipe(peptides, channels, report)


def verify_references(peptides):
    for name in ("sequence_references", "reference_proteins", "reference_accessions"):
        assert sha(SOURCE / "typed" / (name + ".parquet")) == sha(
            OLD / "external/pxd009092/typed" / (name + ".parquet")
        )
    spec = importlib.util.spec_from_file_location(
        "independent_scanner",
        ROOT / "reproduce/source/independent_reference_scanner.py",
    )
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    all_proteins = read("reference_proteins")
    all_aliases = read("reference_accessions")
    all_occurrences = read("peptide_reference_occurrences")
    all_claims, all_resolutions, all_coverage = (
        read("source_parent_claims"),
        read("source_parent_resolutions"),
        read("peptide_reference_coverage"),
    )
    output = {}
    for ref in read("sequence_references"):
        refid = ref["reference_id"]
        proteins = {
            r["reference_record_id"]: r
            for r in all_proteins
            if r["reference_id"] == refid
        }
        aliases = defaultdict(set)
        for row in all_aliases:
            if row["reference_id"] == refid:
                aliases[row["accession_raw"]].add(row["reference_record_id"])
        hits = scanner.scan(
            {k: r["sequence"] for k, r in proteins.items()},
            {k: r["sequence_raw"] for k, r in peptides.items()},
        )
        occurrences = [r for r in all_occurrences if r["reference_id"] == refid]
        assert len(hits) == len(occurrences)
        assert hits == {
            (r["peptide_row_id"], r["reference_record_id"], r["start_1based"])
            for r in occurrences
        }
        by_peptide, by_parent = defaultdict(set), defaultdict(list)
        for row in occurrences:
            pid, parent, start = (
                row["peptide_row_id"],
                row["reference_record_id"],
                row["start_1based"],
            )
            sequence, pep = proteins[parent]["sequence"], peptides[pid]
            end = start + len(pep["sequence_raw"]) - 1
            listed = set().union(
                *(aliases[x] for x in pep["parent_labels_raw"].split(";") if x)
            )
            subset(
                row,
                {
                    "end_1based": end,
                    "n_flank": sequence[start - 2] if start > 1 else "-",
                    "c_flank": sequence[end] if end < len(sequence) else "-",
                    "source_listed_resolved": parent in listed,
                    "source_leading_resolved": parent
                    in aliases[pep["leading_parent_raw"]],
                },
            )
            by_peptide[pid].add(parent)
            by_parent[(pid, parent)].append(start)
        graph = connected(
            peptides,
            [(pid, proteins[parent]["sequence_sha256"]) for pid, parent, _ in hits],
        )
        claims = {
            (r["peptide_row_id"], r["source_accession_raw"]): r
            for r in all_claims
            if r["reference_id"] == refid
        }
        resolutions = {
            (r["claim_id"], r["reference_record_id"]): r
            for r in all_resolutions
            if r["reference_record_id"] in proteins
        }
        coverage = {
            r["peptide_row_id"]: r for r in all_coverage if r["reference_id"] == refid
        }
        expected_claims, expected_resolutions = set(), set()
        for pid, pep in peptides.items():
            leading = pep["leading_parent_raw"]
            listed = set(filter(None, pep["parent_labels_raw"].split(";")))
            labels = listed | ({leading} if leading else set())
            resolved, coords = set(), []
            for label in labels:
                expected_claims.add((pid, label))
                claim = claims[(pid, label)]
                candidates = aliases[label]
                resolved.update(candidates)
                subset(
                    claim,
                    {
                        "resolution_status": "resolved" if candidates else "missing_id",
                        "n_candidate_records": len(candidates),
                        "listed_in_source": label in listed,
                        "source_leading": label == leading,
                    },
                )
                for parent in candidates:
                    key = (claim["claim_id"], parent)
                    expected_resolutions.add(key)
                    comparison = [None] * 4
                    status = "not_leading_source_claim"
                    if label == leading:
                        a, b = pep["start_1based_reported"], pep["end_1based_reported"]
                        if a is None or b is None:
                            status = "source_coordinates_unreported"
                        else:
                            seq = proteins[parent]["sequence"]
                            ok = 1 <= a <= b <= len(seq)
                            comparison = [
                                ok,
                                ok and seq[a - 1 : b] == pep["sequence_raw"],
                                ok
                                and pep["n_flank_raw"]
                                == (seq[a - 2] if a > 1 else "-"),
                                ok
                                and pep["c_flank_raw"]
                                == (seq[b] if b < len(seq) else "-"),
                            ]
                            status = (
                                "out_of_range"
                                if not ok
                                else "peptide_mismatch"
                                if not comparison[1]
                                else "peptide_and_flanks_match"
                                if all(comparison)
                                else "peptide_match_flank_mismatch"
                            )
                            coords.append(comparison)
                    subset(
                        resolutions[key],
                        {
                            "n_exact_occurrences": len(by_parent[(pid, parent)]),
                            "occurrence_status": "exact_occurrence"
                            if by_parent[(pid, parent)]
                            else "resolved_sequence_absent",
                            "accession_role": "primary"
                            if proteins[parent]["accession"] == label
                            else "recorded_secondary",
                            "coordinate_status": status,
                            **dict(
                                zip(
                                    (
                                        "coordinate_in_range",
                                        "coordinate_peptide_equal",
                                        "coordinate_left_equal",
                                        "coordinate_right_equal",
                                    ),
                                    comparison,
                                )
                            ),
                        },
                    )
            matched = by_peptide[pid]
            subset(
                coverage[pid],
                {
                    "mapping_status": "exact_occurrence"
                    if matched
                    else "no_exact_occurrence",
                    "shared_sequence_component_id": refid + ":component:" + graph[pid],
                    "n_source_claims": len(labels),
                    "n_missing_claims": sum(not aliases[x] for x in labels),
                    "n_source_resolved_records": len(resolved),
                    "n_source_matched_records": len(resolved & matched),
                    "n_reference_records": len(matched),
                    "n_reference_sequences": len(
                        {proteins[p]["sequence_sha256"] for p in matched}
                    ),
                    "n_occurrences": sum(len(by_parent[(pid, p)]) for p in matched),
                    "n_additional_records": len(matched - resolved),
                    "leading_has_occurrence": bool(aliases[leading] & matched),
                    "leading_coordinate_match": any(c[1] for c in coords),
                    "leading_coordinate_flanks_match": any(all(c) for c in coords),
                    "leading_coordinate_conflict": bool(coords)
                    and not any(c[1] for c in coords),
                },
            )
        assert (
            set(claims) == expected_claims
            and set(resolutions) == expected_resolutions
            and set(coverage) == set(peptides)
        )
        output[refid] = {
            "proteins": len(proteins),
            "exact_occurrences": len(hits),
            "claims": len(claims),
            "resolutions": len(resolutions),
            "peptide_coverage_rows": len(coverage),
        }
        print(
            "Independent complete reference scan verified", refid, len(hits), flush=True
        )
    return output


def verify_recipe(peptides, channels, report):
    manifest = json.loads((RECIPE / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        assert (
            sha(RECIPE / name) == item["sha256"]
            and (RECIPE / name).stat().st_size == item["size_bytes"]
        )
    assert (
        manifest["source_package_manifest_sha256"]
        == sha(PACKAGE / "manifest.json")
        == sha(RECIPE / "source_manifest.json")
    )
    # Reconstruct the exact pretty-JSON envelope one native table at a time.
    names = sorted(
        [
            "peptide_rows",
            "channels",
            "quantity_results",
            "quantity_measurements",
            "evidence",
            "derivation_inputs",
            "source_assertions",
            "label_aliases",
            "reference_proteins",
            "peptide_reference_occurrences",
        ]
    )
    digest = hashlib.sha256(b"{\n")
    for i, name in enumerate(names):
        if i:
            digest.update(b",\n")
        digest.update(
            json.dumps({name: read(name)}, sort_keys=True, indent=2, allow_nan=False)[
                2:-2
            ].encode()
        )
    digest.update(b"\n}\n")
    assert digest.hexdigest() == sha(RECIPE / "source_tables.json")
    proteins = {
        r["reference_record_id"]: r["sequence_sha256"]
        for r in read("reference_proteins")
    }
    graph = connected(
        peptides,
        [
            (r["peptide_row_id"], proteins[r["reference_record_id"]])
            for r in read("peptide_reference_occurrences")
        ],
        identical=True,
    )
    groups = defaultdict(set)
    for pid, component in graph.items():
        groups[component].add(peptides[pid]["sequence_raw"])
    identifiers = {
        k: hashlib.sha256("\n".join(sorted(v)).encode()).hexdigest()
        for k, v in groups.items()
    }
    components = index(
        json.loads((RECIPE / "components.json").read_text()), "peptide_row_id"
    )
    assert set(components) == set(peptides)
    for pid, component in graph.items():
        identifier = identifiers[component]
        bucket = (
            int(
                hashlib.sha256(
                    (CFG["recipe_id"] + ":" + identifier).encode()
                ).hexdigest(),
                16,
            )
            % 20
        )
        subset(
            components[pid],
            {
                "component_id": identifier,
                "split": "train" if bucket < 14 else "val" if bucket < 17 else "test",
            },
        )
    dispositions = index(
        json.loads((RECIPE / "dispositions.json").read_text()), "result_id"
    )
    examples = index(json.loads((RECIPE / "examples.json").read_text()), "result_id")
    measurements = {
        r["result_id"]: r
        for r in read("quantity_measurements")
        if r["representation"] == "reported_run_intensity"
    }
    admitted, reasons, project_counts, split_counts = (
        set(),
        Counter(),
        Counter(),
        Counter(),
    )
    for row in read("quantity_results"):
        rid, pid, cid = row["result_id"], row["peptide_row_id"], row["channel_id"]
        pep, measure = peptides[pid], measurements[rid]
        errors = set()
        for flag, reason in (
            ("reverse", "source_reverse"),
            ("potential_contaminant", "source_contaminant"),
        ):
            if pep[flag]:
                errors.add(reason)
        if pep["pep"] is None or not math.isfinite(pep["pep"]) or pep["pep"] > 0.01:
            errors.add("peptide_PEP_unreported_or_above_0.01")
        if len(set(filter(None, pep["group_ids_raw"].split(";")))) != 1:
            errors.add("not_one_source_group")
        if not pep["sequence_raw"] or set(pep["sequence_raw"]) - set(
            "ACDEFGHIKLMNPQRSTVWY"
        ):
            errors.add("unsupported_sequence")
        for field, reason in (
            ("value", "no_reported_positive_quantity"),
            ("recomputed_value", "no_positive_evidence_sum"),
        ):
            if (
                measure[field] is None
                or not math.isfinite(measure[field])
                or measure[field] <= 0
            ):
                errors.add(reason)
        common = {
            **components[pid],
            "channel_id": cid,
            "source_dataset": row["source_dataset"],
            "measurement_id": measure["measurement_id"],
        }
        subset(
            dispositions[rid],
            {
                **common,
                "status": "excluded" if errors else "admitted",
                "reported_value": measure["value"],
                "reported_value_status": measure["value_status"],
            },
        )
        assert set(dispositions[rid]["reasons"]) == errors
        reasons.update(errors)
        if errors:
            continue
        admitted.add(rid)
        subset(
            examples[rid],
            {
                **common,
                "peptide_sequence": pep["sequence_raw"],
                "source_group_ids_raw": pep["group_ids_raw"],
                "source_parent_labels_raw": pep["parent_labels_raw"],
                "source_peptide_PEP": pep["pep"],
                "reported_intensity": measure["value"],
                "recomputed_intensity": measure["recomputed_value"],
                "context": channels[cid],
                "evidence_counts": {k: v for k, v in row.items() if k.startswith("n_")},
            },
        )
        assert examples[rid]["target_log10_intensity"] == math.log10(measure["value"])
        project_counts[row["source_dataset"]] += 1
        split_counts[components[pid]["split"]] += 1
    assert set(examples) == admitted and set(dispositions) == set(measurements)
    summary = json.loads((RECIPE / "summary.json").read_text())
    assert summary == {
        "n_components": len(groups),
        "n_examples": len(admitted),
        "n_excluded_results": len(dispositions) - len(admitted),
        "n_source_results": len(dispositions),
        "n_source_peptides": len(peptides),
        "reason_counts": dict(reasons),
        "examples_by_project": dict(project_counts),
        "examples_by_split": dict(split_counts),
    }
    report["recipe"] = summary
    report["held_out_outcomes"] = (
        "No aggregate response statistics, model fits or predictions; exact source/target arithmetic and eligibility only."
    )
    print(
        "Every recipe row, exclusion, source contributor and split verified", flush=True
    )


if __name__ == "__main__":
    started, report = time.perf_counter(), {"passed": False, "hitlist_imports": False}
    try:
        main(report)
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (ROOT / "results/independent-verification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
