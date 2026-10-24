"""Independent original-cell and NetworkX audit; no producer curator imported."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
source = Path(config["source_root"]) / "txt"
typed = Path(config["package"]) / "external/pxd009438/typed"
compiled = Path(config["recipe_package"]) / "composition/typed"


def original(name):
    with (source / (name + ".txt")).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def table(directory, name):
    return pq.read_table(directory / (name + ".parquet")).to_pylist()


def compact(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


raw = {
    name: original(name)
    for name in ("peptides", "evidence", "proteinGroups", "summary", "parameters")
}
source_records = {r["source_record_id"]: r for r in table(typed, "source_records")}
n_cells = 0
for name, rows in raw.items():
    for number, row in enumerate(rows, 2):
        actual = source_records[f"pxd009438:{name}:row:{number}"]
        assert actual["asset_path"] == f"txt/{name}.txt" and actual["row_1based"] == number
        assert json.loads(actual["fields_json"]) == row
        n_cells += len(row)
assert len(source_records) == sum(map(len, raw.values()))
runs = {r["Experiment"]: r for r in raw["summary"] if r["Experiment"]}
peptides = {r["id"]: r for r in raw["peptides"]}
assert len(peptides) == len(raw["peptides"]) == 985 and len(runs) == 14
evidence_groups = defaultdict(list)
for row in raw["evidence"]:
    assert row["Sequence"] == peptides[row["Peptide ID"]]["Sequence"]
    assert row["Raw file"] == runs[row["Experiment"]]["Raw file"]
    evidence_groups[(row["Peptide ID"], row["Experiment"])].append(row)

graph = nx.Graph()
for row in raw["peptides"]:
    peptide = "peptide:" + row["Sequence"]
    graph.add_node(peptide)
    graph.add_edges_from((peptide, "parent:" + p) for p in row["Proteins"].split(";") if p)
component = {}
for nodes in nx.connected_components(graph):
    component.update(
        dict.fromkeys(nodes, digest("pxd009438_composition_v1:" + compact(sorted(nodes))))
    )
assert {r["node_id"]: r["component_id"] for r in table(compiled, "components")} == component

channels = {r["experiment_raw"]: r for r in table(typed, "channels")}
for experiment, channel in channels.items():
    assert channel["condition"] == experiment.split("_")[1]
    assert channel["raw_file"] == runs[experiment]["Raw file"]
    assert channel["measured_species_scope"] == "project_organism"
    assert channel["material_link_precision"] == "acquisition_run_unresolved_specimen"
    assert all(
        channel[k] is None for k in ("specimen_id", "aliquot_id", "pair_id", "replicate_label")
    )
for context in table(typed, "material_contexts"):
    assert context["specimen_id"] is context["aliquot_id"] is None

quantities = {r["result_id"]: r for r in table(typed, "quantity_results")}
measurements = {r["measurement_id"]: r for r in table(typed, "quantity_measurements")}
aliases = table(typed, "label_aliases")
assert len(aliases) == len(runs)
decisions, candidates = {}, defaultdict(list)
max_reconciliation = 0.0
for peptide_id, row in peptides.items():
    parents = sorted(set(filter(None, row["Proteins"].split(";"))))
    for experiment in sorted(runs):
        cid = "pxd009438:run:" + experiment
        rid = f"pxd009438:peptide:{peptide_id}:run:{experiment}"
        mid = rid + ":reported_run_intensity"
        number = float(row["Intensity " + experiment])
        ev = evidence_groups[(peptide_id, experiment)]
        numeric = [float(e["Intensity"]) for e in ev if e["Intensity"] != ""]
        direct = [
            e
            for e in ev
            if e["MS/MS Count"] and int(e["MS/MS Count"]) > 0 and e["Type"] != "MULTI-MATCH"
        ]
        positive_direct = [e for e in direct if e["Intensity"] and float(e["Intensity"]) > 0]
        actual = quantities[rid]
        assert actual["n_source_evidence"] == len(ev)
        assert actual["n_numeric_evidence"] == len(numeric)
        assert actual["n_direct_msms_evidence"] == len(direct)
        assert actual["n_positive_direct_msms_evidence"] == len(positive_direct)
        assert measurements[mid]["value"] == number
        replay = sum(numeric)
        error = abs(replay - number) / max(1.0, abs(number))
        max_reconciliation = max(max_reconciliation, error)
        assert error < 1e-4
        reasons = []
        if not math.isfinite(number) or number <= 0:
            reasons.append("not_positive_finite_original_run_intensity")
        if not row["Sequence"] or any(a not in "ACDEFGHIKLMNPQRSTVWY" for a in row["Sequence"]):
            reasons.append("noncanonical_sequence")
        if row["Reverse"] != "" or row["Potential contaminant"] != "":
            reasons.append("peptide_flags_not_explicitly_blank")
        if not row["PEP"] or not 0 <= float(row["PEP"]) < 0.01:
            reasons.append("peptide_PEP_not_in_[0,.01)")
        if row["Identification type " + experiment] != "By MS/MS":
            reasons.append("not_direct_run_MS_MS_identification")
        if not positive_direct:
            reasons.append("no_positive_direct_MS_MS_evidence")
        if len(numeric) != len(ev):
            reasons.append("incomplete_evidence_intensities")
        if len(parents) != 1:
            reasons.append("not_one_distinct_supplied_parent")
        group = (
            "pxd009438:composition-group:" + digest(compact([parents[0], cid]))
            if len(parents) == 1
            else None
        )
        decisions[mid] = {
            "reasons": reasons,
            "component_id": component["peptide:" + row["Sequence"]],
            "group_id": group,
            "cohort": experiment.split("_")[1],
        }
        if not reasons:
            candidates[group].append(
                {
                    "mid": mid,
                    "sequence": row["Sequence"],
                    "value": number,
                    "parent": parents[0],
                    "run": experiment,
                }
            )
groups = {}
for group, rows in candidates.items():
    frequency = Counter(r["sequence"] for r in rows)
    keep = []
    for row in rows:
        if frequency[row["sequence"]] > 1:
            decisions[row["mid"]]["reasons"].append("duplicate_eligible_sequence_in_group")
        else:
            keep.append(row)
    if len(keep) < 3:
        for row in keep:
            decisions[row["mid"]]["reasons"].append("fewer_than_three_distinct_sequences_in_group")
    else:
        groups[group] = keep
for row in table(compiled, "decisions"):
    expected = decisions[row["measurement_id"]]
    assert set(json.loads(row["reasons_json"])) == set(expected["reasons"])
    assert row["status"] == ("excluded" if expected["reasons"] else "admitted")
    for key in ("group_id", "component_id", "cohort"):
        assert row[key] == expected[key]
assert len(decisions) == len(table(compiled, "decisions")) == 13790

examples = {r["measurement_id"]: r for r in table(compiled, "examples")}
errors = []
for cohort in ("CTRL", "LUPUS"):
    selected = {
        g: rows for g, rows in groups.items() if decisions[rows[0]["mid"]]["cohort"] == cohort
    }
    group_counts = Counter(decisions[rows[0]["mid"]]["component_id"] for rows in selected.values())
    for group, rows in selected.items():
        c = decisions[rows[0]["mid"]]["component_id"]
        gw = 1 / len(group_counts) / group_counts[c]
        logs = np.log10([r["value"] for r in rows])
        fractions = np.array([r["value"] for r in rows]) / sum(r["value"] for r in rows)
        for r, y, p in zip(rows, logs - logs.mean(), fractions, strict=True):
            actual = examples[r["mid"]]
            assert (
                actual["peptide_sequence"] == r["sequence"]
                and actual["parent_accession"] == r["parent"]
            )
            assert actual["group_id"] == group and actual["cohort"] == cohort
            assert actual["quantitative_value"] == r["value"]
            errors.extend(
                [
                    abs(actual["target_centered_log10_intensity"] - y),
                    abs(actual["target_intensity_fraction"] - p),
                    abs(actual["group_weight"] - gw),
                    abs(actual["weight"] - gw / len(rows)),
                ]
            )
assert set(examples) == {mid for mid, d in decisions.items() if not d["reasons"]}
assert max(errors, default=0) < 1e-12
covariates = json.loads((ROOT / "results/covariates.json").read_text())
labels = {
    "quantitative_value",
    "log10_intensity",
    "target_centered_log10_intensity",
    "target_intensity_fraction",
}
assert all(not set(r) & labels for r in covariates)
assert {r["measurement_id"] for r in covariates} == set(examples)
receipt = {
    "passed": True,
    "n_original_cells": n_cells,
    "n_decisions": len(decisions),
    "n_admitted_measurements": len(examples),
    "n_graph_nodes": len(component),
    "maximum_target_weight_error": max(errors, default=0),
    "maximum_source_reconciliation_relative_error": max_reconciliation,
    "metadata": "original source labels; no physical specimen or parent-species verification",
}
(ROOT / "results/data-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
