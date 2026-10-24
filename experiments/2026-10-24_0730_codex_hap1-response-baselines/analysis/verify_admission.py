"""Independent arithmetic and graph replay from typed source snapshots (no Hitlist imports)."""

import argparse
import hashlib
import json
import math
import re
import resource
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        yield from batch.to_pylist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    root = args.recipe
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert digest(root / name) == expected["sha256"]
    source_manifest = json.loads((root / "source_manifest.json").read_text())
    assert (
        digest(root / "source_manifest.json")
        == "88cad10d7bc2a638305cb0d37ad96ac2826c3badaf7b3e0aad7928820df5c4b8"
    )
    for path in (root / "source").iterdir():
        assert digest(path) == source_manifest["files"]["hap1/typed/" + path.name]["sha256"]
    typed = root / "source"
    contexts = {r["context_id"]: r for r in read(typed / "contexts.parquet")}
    samples = {r["sample_id"]: r for r in read(typed / "samples.parquet")}
    references = {r["reference_record_id"]: r for r in read(typed / "references.parquet")}
    adjacency = defaultdict(set)

    def edge(a, b):
        adjacency[a].add(b)
        adjacency[b].add(a)

    def tokens(value):
        return set(re.findall(r"[^;,\s]+", value or ""))

    roles, parent_tokens = defaultdict(set), defaultdict(set)
    for reference in references.values():
        parent = reference["sequence_sha256"]
        assert hashlib.sha256(reference["reference_sequence"].encode()).hexdigest() == parent
        roles[parent].add(reference["reference_role"])
        parent_tokens[parent].add(reference["accession_raw"])
        edge("parent:" + parent, "claim:" + reference["accession_raw"])
    group_members, member_groups = {}, defaultdict(set)
    for row in read(typed / "source_records.parquet"):
        if row["source_role"] == "deposited_protein_report":
            raw = json.loads(row["source_row_json"])
            group = raw["Protein"]
            members = tokens(raw.get("Indistinguishable Proteins")) | {group}
            group_members[group] = members
            for member in members:
                member_groups[member].add(group)
                edge("group:" + group, "claim:" + member)
    peptide_parents = defaultdict(set)
    for row in read(typed / "occurrences.parquet"):
        peptide = row["peptide_sequence"]
        node = "peptide:" + peptide
        adjacency[node]
        for token in tokens(row["parent_label_raw"]):
            edge(node, "claim:" + token)
        if row["reference_record_id"] is not None:
            reference = references[row["reference_record_id"]]
            assert (
                reference["reference_sequence"][row["start_1based"] - 1 : row["end_1based"]]
                == peptide
            )
            peptide_parents[peptide].add(reference["sequence_sha256"])
            edge(node, "parent:" + reference["sequence_sha256"])
    derived_components = {
        r["peptide_sequence"]: r for r in read(root / "derived/components.parquet")
    }
    visited = set()
    n_graph_components = 0
    for node in sorted(adjacency):
        if node in visited:
            continue
        queue, nodes = deque([node]), {node}
        visited.add(node)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes.add(neighbor)
                    queue.append(neighbor)
        identifier = hashlib.sha256(
            json.dumps(sorted(nodes), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        bucket = (
            int(
                hashlib.sha256(("hap1_machinery_response_v1" + identifier).encode()).hexdigest(), 16
            )
            % 20
        )
        split = "train" if bucket < 14 else "validation" if bucket < 17 else "test"
        for node2 in nodes:
            if node2.startswith("peptide:"):
                row = derived_components[node2[8:]]
                assert (
                    row["component_id"] == identifier
                    and row["split"] == split
                    and row["n_component_nodes"] == len(nodes)
                )
        n_graph_components += 1
    adjacency.clear()
    visited.clear()
    cells, peptide_positive, protein_positive = {}, defaultdict(list), defaultdict(list)
    background = set()
    for row in read(typed / "quantities.parquet"):
        if (
            row["quantity_kind"] not in {"normalized_peptide_intensity", "protein_MaxLFQ_intensity"}
            or row["value_status"] != "source_positive"
        ):
            continue
        assert (
            row["quantitative_value"] > 0
            and float(row["quantitative_raw"]) == row["quantitative_value"]
        )
        sample = samples[row["sample_id"]]
        assert sample["context_id"] == row["context_id"]
        condition = contexts[row["context_id"]]["condition_id"]
        cells[row["measurement_id"]] = (
            row["quantitative_value"],
            condition,
            row["peptide_sequence"],
            row["parent_label_raw"],
        )
        if row["quantity_kind"] == "normalized_peptide_intensity":
            peptide_positive[row["peptide_sequence"], condition].append(row["measurement_id"])
            if condition == "hap1_b2m_ko":
                background.add(row["peptide_sequence"])
        else:
            protein_positive[row["parent_label_raw"], condition].append(row["measurement_id"])
    inputs = {
        (r["parent_id"], r["condition_id"]): r
        for r in read(root / "derived/precursor_inputs.parquet")
    }
    n_verified_inputs = 0
    for (parent, condition), row in inputs.items():
        group_ids = {g for token in parent_tokens[parent] for g in member_groups[token]}
        assert set(json.loads(row["group_ids_json"])) == group_ids
        if len(group_ids) != 1:
            assert (
                row["status"]
                == ("ambiguous_protein_groups" if group_ids else "missing_protein_group")
                and row["log10_response"] is None
            )
            continue
        group = next(iter(group_ids))
        wt, ko = protein_positive[group, "hap1_wildtype"], protein_positive[group, condition]
        assert set(wt) == set(json.loads(row["wt_measurement_ids_json"]))
        assert set(ko) == set(json.loads(row["ko_measurement_ids_json"]))
        if min(len(wt), len(ko)) < 2:
            assert (
                row["status"] == "insufficient_positive_protein_replicates"
                and row["log10_response"] is None
            )
        else:
            expected = math.log10(math.prod(cells[i][0] for i in ko)) / len(ko) - math.log10(
                math.prod(cells[i][0] for i in wt)
            ) / len(wt)
            assert row["status"] == "complete" and abs(expected - row["log10_response"]) < 1e-12
        n_verified_inputs += 1
    responses = {}
    max_error = 0.0
    for row in read(root / "derived/responses.parquet"):
        identifier = row["measurement_id"]
        assert identifier not in responses
        responses[identifier] = True
        value, condition, peptide, _ = cells[identifier]
        assert row["peptide_sequence"] == peptide and row["condition_id"] == condition
        wt = peptide_positive[peptide, "hap1_wildtype"]
        assert len(wt) >= 2 and set(wt) == set(json.loads(row["wt_measurement_ids_json"]))
        expected = math.log10(value) - math.log10(math.prod(cells[i][0] for i in wt)) / len(wt)
        max_error = max(max_error, abs(expected - row["target_log10_ratio"]))
        assert abs(expected - row["target_log10_ratio"]) < 1e-12
        parents = peptide_parents[peptide]
        assert parents == set(json.loads(row["parent_ids_json"]))
        assert all("human_target" in roles[parent] for parent in parents)
        assert peptide not in background
        for key in ["split", "component_id"]:
            assert row[key] == derived_components[peptide][key]
        parent_values = [inputs[parent, condition] for parent in parents]
        complete = all(p["status"] == "complete" for p in parent_values)
        assert row["precursor_status"] == ("complete" if complete else "incomplete")
        if complete:
            expected_input = sum(p["log10_response"] for p in parent_values) / len(parents)
            assert abs(expected_input - row["precursor_log10_ratio"]) < 1e-12
        else:
            assert row["precursor_log10_ratio"] is None
    counts = Counter()
    quantities, dispositions = (
        read(typed / "quantities.parquet"),
        read(root / "derived/dispositions.parquet"),
    )
    from itertools import zip_longest

    for quantity, disposition in zip_longest(quantities, dispositions):
        assert quantity is not None and disposition is not None
        assert (
            quantity["measurement_id"] == disposition["measurement_id"]
            and quantity["source_record_id"] == disposition["source_record_id"]
        )
        eligible = False
        if quantity["quantity_kind"] == "normalized_peptide_intensity":
            peptide = quantity["peptide_sequence"]
            condition = contexts[quantity["context_id"]]["condition_id"]
            parents = peptide_parents[peptide]
            eligible = bool(
                condition not in {"hap1_wildtype", "hap1_b2m_ko"}
                and quantity["value_status"] == "source_positive"
                and peptide not in background
                and peptide
                and set(peptide) <= set("ACDEFGHIKLMNPQRSTVWY")
                and parents
                and all("human_target" in roles[parent] for parent in parents)
                and len(peptide_positive[peptide, "hap1_wildtype"]) >= 2
            )
        assert (
            eligible
            == (disposition["status"] == "admitted")
            == (quantity["measurement_id"] in responses)
        )
        counts[disposition["status"]] += 1
    result = {
        "status": "passed",
        "recipe_manifest_sha256": digest(root / "manifest.json"),
        "n_quantity_dispositions": sum(counts.values()),
        "n_responses": len(responses),
        "n_parent_inputs": len(inputs),
        "n_sufficient_or_insufficient_group_checks": n_verified_inputs,
        "n_all_graph_components": n_graph_components,
        "n_source_peptide_components_checked": len(derived_components),
        "dispositions": counts,
        "max_response_error": max_error,
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "limits": "Checks typed source faithfulness and endpoint arithmetic; original source audit separate. Does not establish specimen or homology independence.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
