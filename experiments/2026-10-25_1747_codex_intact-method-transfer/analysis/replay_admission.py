"""Independent source-table reconstruction; no Hitlist compiler import or model fit."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from xml.dom import minidom

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def read(directory, name):
    return pq.read_table(directory / (name + ".parquet")).to_pylist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.requests.read_text())
    root = Path(__file__).resolve().parents[1]
    package = Path(request["recipe_directory"])
    source, output = package / "source_tables", package / "typed"
    config = json.loads((root / "reproduce/launch.json").read_text())
    original = Path(config["source_package"])
    manifest = json.loads((original / "manifest.json").read_text())
    assert (
        hashlib.sha256((original / "manifest.json").read_bytes()).hexdigest()
        == config["source_manifest_sha256"]
    )
    for path in source.glob("*.parquet"):
        entries = [
            (name, value)
            for name, value in manifest["files"].items()
            if name.endswith("/typed/" + path.name)
        ]
        assert len(entries) == 1
        relative, evidence = entries[0]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["sha256"]
        assert (
            hashlib.sha256((original / relative).read_bytes()).hexdigest()
            == evidence["sha256"]
        )
    raw = (package / "source_assets/article.xml").read_bytes()
    article_hash = hashlib.sha256(raw).hexdigest()
    assert (
        article_hash
        == "34c2d374b62afe17c48e04bdfd75a5f91e23786db1d537035d632c0f7f411bea"
    )
    document = minidom.parseString(raw)

    def text(node):
        return "".join(
            text(c)
            if c.nodeType == c.ELEMENT_NODE
            else c.data
            if c.nodeType in (c.TEXT_NODE, c.CDATA_SECTION_NODE)
            else ""
            for c in node.childNodes
        )

    assert "10.1073/pnas.2419607122" in [
        text(n) for n in document.getElementsByTagName("article-id")
    ]
    sections = [
        n for n in document.getElementsByTagName("sec") if n.getAttribute("id") == "s15"
    ]
    assert len(sections) == 1 and sections[0].parentNode.getAttribute("id") == "s6"
    section = sections[0]
    titles = [
        text(n)
        for n in section.childNodes
        if n.nodeType == n.ELEMENT_NODE and n.tagName == "title"
    ]
    assert titles == ["ID-MS Data Analysis and Statistics."]
    phrase = "No additional modifications were included"
    paragraphs = [
        text(n) for n in section.getElementsByTagName("p") if phrase in text(n)
    ]
    assert len(paragraphs) == 1
    assert all(
        term in paragraphs[0]
        for term in ("ClipsMS", "10 ppm", "P37840", "4 amino acids", "b-, y-, or by-")
    )
    paragraph_hash = hashlib.sha256(paragraphs[0].encode()).hexdigest()
    assert (
        paragraph_hash
        == "557facde6e07c61a824c7e71d00b1025847dd578d1ac63c24d0c6680f79f7282"
    )
    claim_id = (
        "pxd058121:search_policy:"
        + hashlib.sha256((article_hash + paragraph_hash).encode()).hexdigest()
    )
    assert read(output, "search_policies") == [
        {
            "claim_id": claim_id,
            "source_record_id": None,
            "subject_id": "PXD058121:original_ClipsMS_search_assignments",
            "source_scope": "study_method_for_original_ClipsMS_search_workbooks",
            "assertion_predicate": "additional_search_modifications",
            "assertion_value": "none",
            "assertion_status": "reported_method",
            "source_locator": "article.xml#s15:ID-MS Data Analysis and Statistics",
            "source_asset_sha256": article_hash,
            "source_paragraph_sha256": paragraph_hash,
            "declaration_excerpt": phrase,
        }
    ]
    views = {r["view_id"]: r for r in read(source, "source_views")}
    acq = {r["acquisition_id"]: r for r in read(source, "acquisitions")}
    assignments = {r["assignment_id"]: r for r in read(source, "assignments")}
    old_ids = {
        v["acquisition_id"]
        for v in views.values()
        if v["source_view_role"] == "kinetics" and v["acquisition_id"]
    }
    old_raw = {acq[i]["raw_file"] for i in old_ids if i in acq and acq[i]["raw_file"]}
    reused = set()
    for c in read(source, "reconciliations"):
        if c["assertion_predicate"] == "repeated_acquisition_source_views":
            pair = [
                views.get(c[k], {}) for k in ("source_subject_id", "target_subject_id")
            ]
            if {v.get("source_view_role") for v in pair} == {"kinetics", "organs"}:
                reused.update(
                    v["view_id"] for v in pair if v["source_view_role"] == "organs"
                )
    ids = sorted(
        {
            a["measurement_id"]
            for a in assignments.values()
            if views[a["view_id"]]["source_view_role"] == "organs"
        }
    )
    quantities = {
        q["measurement_id"]: q["quantitative_value"]
        for q in ds.dataset(source / "quantities.parquet")
        .to_table(filter=ds.field("measurement_id").isin(ids))
        .to_pylist()
    }
    reasons, candidates, modification_metadata = {}, {}, {}
    required_organs = ("Brain", "Liver", "Spleen")
    for key, a in assignments.items():
        v, sample = views[a["view_id"]], acq.get(a["acquisition_id"], {})
        errors = []
        if v["source_view_role"] != "organs":
            errors = ["outside_organs_view"]
            modification_metadata[key] = (None, None)
        else:
            value = quantities.get(a["measurement_id"])
            mods = json.loads(a["modifications_json"])
            explicit = mods == {
                "Localized Mod": "0",
                "Unlocalized Mod": "None",
                "Term Mod": "0",
            }
            method = mods == {
                "Localized Mod": "0",
                "Unlocalized Mod": None,
                "Term Mod": "0",
            }
            modification_metadata[key] = (
                ("explicit_row", [])
                if explicit
                else ("reported_method_no_additional_modifications", [claim_id])
                if method
                else ("unsupported", [])
            )
            conditions = {
                "unresolved_acquisition": bool(sample.get("raw_file")),
                "F2_acquisition_overlap": a["acquisition_id"] not in old_ids
                and sample.get("raw_file") not in old_raw,
                "source_backed_reused_view": a["view_id"] not in reused,
                "outside_fixed_organ_scope": sample.get("organ_label")
                in required_organs,
                "outside_declared_times": sample.get("time_minutes")
                in (0, 30, 60, 90, 120),
                "not_positive_finite": value is not None
                and math.isfinite(value)
                and value > 0,
                "span_conflict": a["span_status"] == "matches_source_substrate",
                "input_link_or_quantity_conflict": a["mass_link_status"]
                == "input_id_and_intensity_agree",
                "ambiguous_mass_assignment": a["n_assignments_reported"]
                == a["n_assignments_replayed"]
                == 1,
                "outside_reported_mass_tolerance": a["mass_error_ppm"] is not None
                and math.isfinite(a["mass_error_ppm"])
                and abs(a["mass_error_ppm"]) <= 10,
                "unmodified_search_assignment_not_supported": explicit or method,
            }
            errors = [reason for reason, ok in conditions.items() if not ok]
        reasons[key] = errors
        if not errors:
            candidates[key] = a
    counts = Counter(
        (a["peptide_sequence"], a["acquisition_id"]) for a in candidates.values()
    )
    for key, a in candidates.items():
        if counts[a["peptide_sequence"], a["acquisition_id"]] > 1:
            reasons[key].append("multiple_masses_for_sequence_acquisition")
    candidates = {k: a for k, a in candidates.items() if not reasons[k]}
    parents = defaultdict(list)
    for key, a in candidates.items():
        if (a["start_1based"], a["end_1based"]) == (1, len(a["parent_sequence"])):
            parents[a["acquisition_id"]].append(key)
    parent_ids = {k for values in parents.values() for k in values}
    ratios, normalizer, curves = {}, {}, defaultdict(list)
    for key, a in candidates.items():
        if key in parent_ids:
            continue
        parent = parents[a["acquisition_id"]]
        if len(parent) != 1:
            reasons[key].append("no_unique_admitted_intact_substrate")
            continue
        normalizer[key] = parent[0]
        ratios[key] = math.log10(quantities[a["measurement_id"]]) - math.log10(
            quantities[assignments[parent[0]]["measurement_id"]]
        )
        curve = (
            acq[a["acquisition_id"]]["organ_label"],
            a["parent_sequence"],
            a["start_1based"],
            a["end_1based"],
            a["peptide_sequence"],
        )
        curves[curve].append(key)
    expected, baseline_ids = {}, set()
    for curve, keys in curves.items():
        base = [
            k
            for k in keys
            if acq[assignments[k]["acquisition_id"]]["time_minutes"] == 0
        ]
        if not base:
            for k in keys:
                reasons[k].append("no_eligible_zero_baseline")
            continue
        baseline_ids.update(base)
        mean = math.fsum(ratios[k] for k in base) / len(base)
        for k in set(keys) - set(base):
            expected[k] = (curve, base, mean, ratios[k] - mean)
    actual = {r["assignment_id"]: r for r in read(output, "examples")}
    baselines = {r["assignment_id"]: r for r in read(output, "baselines")}
    decisions = {r["assignment_id"]: r for r in read(output, "decisions")}
    assert actual.keys() == expected.keys()
    assert baselines.keys() == baseline_ids
    assert decisions.keys() == assignments.keys()
    for k, row in decisions.items():
        assert set(json.loads(row["reasons_json"])) == set(reasons[k]), k
        policy, claims = modification_metadata[k]
        assert row["modification_policy"] == policy
        assert (
            json.loads(row["modification_assertion_ids_json"])
            if row["modification_assertion_ids_json"] is not None
            else None
        ) == claims
        status = (
            "admitted"
            if k in expected
            else "baseline_only"
            if k in baseline_ids
            else "normalizer_only"
            if k in parent_ids
            else "excluded"
        )
        assert row["status"] == status, k
        value = quantities.get(assignments[k]["measurement_id"])
        assert row["quantitative_value"] == value, k
    for k, (curve, base, mean, change) in expected.items():
        r = actual[k]
        assert math.isclose(r["target_change"], change, rel_tol=0, abs_tol=1e-12)
        assert r["baseline_log10_ratio"] == mean and r[
            "n_baseline_observations"
        ] == len(base)
        assert set(json.loads(r["baseline_assignment_ids_json"])) == set(base)
        assert set(json.loads(r["baseline_measurement_ids_json"])) == {
            assignments[b]["measurement_id"] for b in base
        }
        assert set(json.loads(r["baseline_normalizer_assignment_ids_json"])) == {
            normalizer[b] for b in base
        }
        assert r["normalizer_assignment_id"] == normalizer[k]
        assert (
            r["normalizer_measurement_id"]
            == assignments[normalizer[k]]["measurement_id"]
        )
        assert r["normalizer_value"] == quantities[r["normalizer_measurement_id"]]
        assert r["target_log10_ratio"] == ratios[k]
    for k, r in baselines.items():
        assert r["target_log10_ratio"] == ratios[k]
        assert r["normalizer_assignment_id"] == normalizer[k]
    for split, times in (("test", (30, 60)), ("extrapolation", (90, 120))):
        selected = {
            k: e
            for k, e in expected.items()
            if acq[assignments[k]["acquisition_id"]]["time_minutes"] in times
        }
        organs = {e[0][0] for e in selected.values()}
        for k, (curve, *_rest) in selected.items():
            n_curves = len({e[0] for e in selected.values() if e[0][0] == curve[0]})
            curve_times = [
                acq[assignments[i]["acquisition_id"]]["time_minutes"]
                for i, e in selected.items()
                if e[0] == curve
            ]
            time = acq[assignments[k]["acquisition_id"]]["time_minutes"]
            weight = 1 / (
                len(organs) * n_curves * len(set(curve_times)) * curve_times.count(time)
            )
            assert actual[k]["split"] == split and actual[k]["weight"] == weight
    summary = json.loads((package / "summary.json").read_text())
    for organ in required_organs:
        for split in ("test", "extrapolation"):
            chosen = [
                r
                for r in actual.values()
                if r["organ_label"] == organ and r["split"] == split
            ]
            expected_support = {
                "n_sequences": len({r["peptide_sequence"] for r in chosen}),
                "n_observations": len(chosen),
                "n_acquisitions": len({r["acquisition_id"] for r in chosen}),
            }
            assert summary["cohort_support"][split][organ] == expected_support
    assert summary["support_passed"] == all(
        v["n_sequences"] >= 20 and v["n_observations"] >= 40
        for v in summary["cohort_support"]["test"].values()
    )
    covariates = {r["assignment_id"]: r for r in read(output, "covariates")}
    assert covariates.keys() == actual.keys()
    forbidden = {
        "quantitative_value",
        "normalizer_value",
        "target_log10_ratio",
        "target_change",
    }
    for k, r in covariates.items():
        assert not forbidden & r.keys()
        assert r == {
            name: value for name, value in actual[k].items() if name not in forbidden
        }
    config = json.loads((root / "reproduce/launch.json").read_text())
    for name, digest in config["model_input_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    modification_counts = Counter(
        a["modifications_json"]
        for a in assignments.values()
        if views[a["view_id"]]["source_view_role"] == "organs"
    )
    result = {
        "F3_modification_counts": dict(modification_counts),
        "raw_source_tables_identical": True,
        "method_evidence_independently_replayed": True,
        "source_article_sha256": article_hash,
        "policy_counts": dict(
            Counter(
                row["modification_policy"]
                for row in decisions.values()
                if row["source_view_role"] == "organs"
            )
        ),
        "status": "passed",
        "n_decisions_replayed": len(decisions),
        "n_baselines_replayed": len(baselines),
        "n_examples_replayed": len(actual),
        "cohort_support": summary["cohort_support"],
        "support_passed": summary["support_passed"],
        "models_fitted_or_scored": False,
    }
    with (root / "results/admission-independent-replay.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
