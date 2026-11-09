"""Check canonical source contexts against the admitted original metadata."""

import json
import re

from replay_support import numeric


def verify_metadata(
    rep,
    original,
    project,
    inventory,
    level_cells,
    paragraphs,
    commands,
    acquisitions,
    first_scores,
    library,
):
    c = rep.contract
    contexts = list(rep.rows("material_contexts"))
    by_context = {r["context_id"]: r for r in contexts}
    assert len(contexts) == 13 and len(by_context) == 13
    scopes = {}
    project_contexts = [
        ("project", "source_project", c["source_dataset"], project.get("title")),
        ("matrix", "study_arm", "JY HLA-I eluate matrix", project.get("projectDescription")),
        (
            "standards",
            "study_arm",
            "post-isolation synthetic standard panel",
            project["sampleProcessingProtocol"],
        ),
    ]
    for scope, kind, label, description in project_contexts:
        matches = [r for r in contexts if r["sample_label_raw"] == label]
        assert len(matches) == 1
        row = matches[0]
        scopes[scope] = row["context_id"]
        rep.check(
            "material_contexts",
            row,
            {
                "context_kind": kind,
                "sample_label_raw": label,
                "condition_text": description,
                "condition_status": "source_protocol_context",
                "material_link_precision": "project_context",
                "context_status": "physical_specimen_and_aliquot_unqualified",
                "source_locator": c["project"],
                "notes": "Source context does not create physical material identity",
            },
        )
    levels = list(rep.rows("nominal_levels"))
    by_sample = {r["sample_label_raw"]: r for r in levels}
    assert len(levels) == len(by_sample) == len(level_cells) == 5
    for locator, cells in level_cells:
        factor, sample, concentration = cells
        amount, unit = concentration.split()
        rep.check(
            "nominal_levels",
            by_sample[sample],
            {
                "source_record_id": original[locator]["source_record_id"],
                "sample_label_raw": sample,
                "dilution_factor_raw": factor,
                "nominal_concentration_raw": amount,
                "nominal_concentration": numeric(amount)[0],
                "measurement_unit": unit,
                "nominal_status": "reported_nominal_solution_not_injected_amount",
            },
        )
    raw_names = {r["fileName"] for r in inventory}
    for run, row in acquisitions.items():
        number = re.search(r"_No([0-9]+)_", run).group(1)
        method = int(re.search(r"_DIA#([12])_", run).group(1))
        level = by_sample["No. " + number]
        raw = run.replace("#", "_") + ".raw"
        rep.check(
            "acquisitions",
            row,
            {
                "context_id": row["context_id"],
                "source_record_id": first_scores[run][0],
                "source_run_id": first_scores[run][1],
                "source_run_label": run,
                "raw_file": raw if raw in raw_names else None,
                "nominal_level_id": level["nominal_level_id"],
                "dia_index_from_name": method,
                "method_link_precision": "filename_derived",
                "material_link_precision": "source_acquisition",
                "replicate_status": "different_named_DIA_acquisitions_not_assumed_exchangeable_replicates",
            },
        )
        rep.check(
            "material_contexts",
            by_context[row["context_id"]],
            {
                "context_kind": "source_acquisition",
                "sample_label_raw": run,
                "condition_id": level["nominal_level_id"],
                "condition_status": "filename_derived_nominal_level",
                "material_link_precision": "source_acquisition",
                "context_status": "physical_material_unqualified",
                "source_locator": first_scores[run][0],
                "notes": "DIA index is a method label; no independent specimen or replicate inference",
            },
        )
    links = list(rep.rows("material_links"))
    expected_edges = {
        (r["context_id"], scopes[s]) for r in acquisitions.values() for s in ("matrix", "standards")
    }
    assert len(links) == len(expected_edges) == 20
    for row in links:
        edge = row["source_context_id"], row["target_context_id"]
        assert edge in expected_edges
        expected_edges.remove(edge)
        rep.check(
            "material_links",
            row,
            {
                "source_context_id": edge[0],
                "target_context_id": edge[1],
                "assertion_predicate": "source_protocol_material_context",
                "assertion_status": "source_protocol_not_physical_pairing",
                "material_link_precision": "study_arm",
                "source_locator": c["project"] + "#/sampleProcessingProtocol",
            },
        )
    assert not expected_edges

    expected_claims = {}

    def claim(
        locator,
        predicate,
        value,
        scope="project",
        status="source_reported",
        details=None,
        record_locator=None,
    ):
        key = locator, predicate, scope
        assert key not in expected_claims
        expected_claims[key] = {
            "source_record_id": original[record_locator or locator]["source_record_id"],
            "context_id": scopes[scope],
            "subject_id": scopes[scope],
            "source_scope": scope,
            "assertion_predicate": predicate,
            "assertion_value": value,
            "assertion_status": status,
            "source_locator": locator,
            "details_json": json.dumps(details or {}, sort_keys=True),
        }

    for key, value in project.items():
        claim(
            c["project"] + "#/" + key,
            "project." + key,
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=True),
        )
    for locator, cells in level_cells:
        claim(
            locator,
            "nominal_dilution_grid",
            json.dumps(cells, ensure_ascii=True),
            status="source_reported_conflict_retained",
        )
    for locator, text in paragraphs:
        claim(locator, "article.paragraph", text)
        if "1:10" in text:
            claim(
                locator, "nominal_dilution_grid", text, status="source_reported_conflict_retained"
            )
    for locator, text in commands:
        claim(
            locator,
            "reference_pipeline_command",
            text,
            status="pinned_reference_code_not_actual_run_configuration",
        )
    description = project["projectDescription"]
    for predicate, value, field in (
        ("cell_line_name", "JY" if "JY" in description else None, "projectDescription"),
        ("mhc_class", "I" if "HLA-I" in description else None, "projectDescription"),
        (
            "cellular_mhc_genotype",
            json.dumps(re.findall(r"HLA-[ABC]\*\d+:\d+", description)),
            "projectDescription",
        ),
        ("organism", json.dumps(project.get("organisms", [])), "organisms"),
        ("instrument", json.dumps(project.get("instruments", [])), "instruments"),
    ):
        claim(
            c["project"] + "#/" + field,
            predicate,
            value,
            "matrix",
            "source_reported_matrix_context_not_standard_restriction",
        )
    for predicate in (
        "per_run_injection_volume",
        "certified_standard_purity",
        "standard_lot",
        "physical_aliquot_lineage",
        "per_run_recovery",
        "actual_executable_parameters",
        "APC_machinery_state",
    ):
        claim(
            "curation_contract:" + predicate,
            predicate,
            None,
            status="unqualified_in_this_source_revision",
            details={
                "scope": "No claim that the value is absent from all literature or supplements"
            },
            record_locator=c["project"] + "#/sampleProcessingProtocol",
        )
    assert len(c["protocol_assertions"]) == 15
    for item in c["protocol_assertions"]:
        assert item["evidence_text"] in project[item["project_field"]]
        claim(
            c["project"] + "#/" + item["project_field"],
            item["predicate"],
            item["value_raw"],
            item["subject"],
            "reported_protocol_not_individual_run_receipt",
            {k: item[k] for k in ("unit", "evidence_text", "applicability")},
        )
    for row in rep.rows("assertions"):
        key = row["source_locator"], row["assertion_predicate"], row["source_scope"]
        rep.check("assertions", row, expected_claims.pop(key))
    assert not expected_claims

    standards = list(rep.rows("nominal_standards"))
    panel = re.findall(r"[A-Z]+\[L\(13C6; 15N\)\][A-Z]+", project["sampleProcessingProtocol"])
    by_form = {r["reported_standard_form"]: r for r in standards}
    assert len(panel) == len(standards) == len(by_form) == 10 and set(panel) == set(by_form)
    expected_members = {}
    for reported in panel:
        left, right = reported.split("[L(13C6; 15N)]")
        row = by_form[reported]
        form = left + "L(UniMod:695)" + right
        rep.check(
            "nominal_standards",
            row,
            {
                "source_record_id": original[c["project"] + "#/sampleProcessingProtocol"][
                    "source_record_id"
                ],
                "peptide_sequence": left + "L" + right,
                "reported_standard_form": reported,
                "expected_modified_sequence": form,
                "standard_group_form": left + "L(Label:13C(6)15N(1))" + right,
                "label_position_1based": len(left) + 1,
                "nominal_status": "intended_synthetic_panel_not_measured_restriction",
            },
        )
        members = [r for r in library.values() if r["modified_sequence_raw"] == form] or [None]
        for member in members:
            lib_id = member["library_precursor_id"] if member else None
            key = row["standard_id"], lib_id
            expected_members[key] = {
                "standard_id": row["standard_id"],
                "source_record_id": row["source_record_id"],
                "library_precursor_id": lib_id,
                "decoy_raw": member["decoy_raw"] if member else None,
                "membership_status": ("target" if member["decoy_raw"] == "0" else "decoy")
                if member
                else "absent_from_library",
            }
    for row in rep.rows("standard_memberships"):
        key = row["standard_id"], row["library_precursor_id"]
        rep.check("standard_memberships", row, expected_members.pop(key))
    assert not expected_members
    return {
        "n_contexts": len(contexts),
        "n_material_links": len(links),
        "n_standards": len(standards),
        "n_levels": len(levels),
        "n_protocol_assertions": 15,
    }
