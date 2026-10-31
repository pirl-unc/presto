"""Independent graph, cohort, targets and weights; no model outcomes are assessed."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    config = json.loads((ROOT / "reproduce/launch.json").read_text())
    package = REPO / config["recipe_directory"]
    previous = REPO / "data/mapp2025_source/2026-10-24_2210_baseline_candidates"
    previous_manifest = json.loads((previous / "manifest.json").read_text())
    # Bind to the already independently audited original-candidate generation.
    assert previous_manifest == json.loads(
        (
            REPO
            / "experiments/2026-10-24_2210_codex_mapp2025-canonical-admission/results/recipe-manifest.json"
        ).read_text()
    )
    for name in ("candidates", "decisions"):
        old = previous / f"typed/{name}.parquet"
        assert (
            hashlib.sha256(old.read_bytes()).hexdigest()
            == previous_manifest["files"][f"typed/{name}.parquet"]["sha256"]
        )
        new = package / f"composition/baseline/{name}.parquet"
        for a, b in zip(
            pq.ParquetFile(old).iter_batches(), pq.ParquetFile(new).iter_batches(), strict=True
        ):
            assert a.equals(b)
    graph = nx.Graph()
    for row in rows(package / "source_tables/peptides.parquet"):
        peptide = "peptide:" + row["peptide_sequence"]
        graph.add_node(peptide)
        for parent in set((row["parent_label_raw"] or "").split(";")) - {""}:
            graph.add_edge(peptide, "parent:" + parent)
    components = {}
    for members in nx.connected_components(graph):
        key = digest(
            "goldberg2025_baseline_v1:"
            + json.dumps(sorted(members), separators=(",", ":"), ensure_ascii=True)
        )
        bucket = int(key, 16) % 10
        split = "train" if bucket < 6 else "val" if bucket < 8 else "test"
        components.update({node: (key, split) for node in members})
    actual = {
        r["node_id"]: (r["component_id"], r["split"])
        for r in rows(package / "composition/typed/components.parquet")
    }
    assert actual == components
    grouped = defaultdict(list)
    first_decisions = {}
    for row in rows(previous / "typed/decisions.parquet"):
        parents = sorted(set((row["parent_label_raw"] or "").split(";")) - {""})
        errors = []
        if row["status"] != "candidate":
            errors.append("baseline_" + row["status"])
        pep = row["posterior_error_probability"]
        if pep is None or not math.isfinite(pep) or pep < 0 or pep >= 0.01:
            errors.append("peptide_PEP_not_in_[0,.01)")
        if row["identification_type"] != "By MS/MS":
            errors.append("not_direct_run_MS_MS_support")
        if len(parents) != 1:
            errors.append("not_one_distinct_supplied_parent")
        group = None
        if len(parents) == 1 and row["acquisition_id"]:
            group = "goldberg2025:baseline_composition_group:" + digest(
                json.dumps([parents[0], row["acquisition_id"]], separators=(",", ":"))
            )
        first_decisions[row["measurement_id"]] = (errors, group)
        if not errors:
            grouped[group].append(row)
    small = {g for g, rs in grouped.items() if len({r["peptide_sequence"] for r in rs}) < 3}
    grouped = {g: rs for g, rs in grouped.items() if g not in small}
    for rs in grouped.values():
        assert len(rs) == len({r["peptide_sequence"] for r in rs})
    split_components = defaultdict(set)
    n_groups = Counter()
    for rs in grouped.values():
        component, split = components["peptide:" + rs[0]["peptide_sequence"]]
        split_components[split].add(component)
        n_groups[component] += 1
    expected = {}
    for group, rs in grouped.items():
        component, split = components["peptide:" + rs[0]["peptide_sequence"]]
        values = np.array([float(r["quantitative_raw"]) for r in rs])
        log = np.log10(values)
        y = log - log.mean()
        p = values / values.sum()
        wg = 1 / (len(split_components[split]) * n_groups[component])
        for i, r in enumerate(rs):
            expected[r["measurement_id"]] = {
                "group_id": group,
                "component_id": component,
                "split": split,
                "target_centered_log10_intensity": y[i],
                "target_intensity_fraction": p[i],
                "weight": wg / len(rs),
                "group_weight": wg,
            }
    max_error = 0.0
    checked = 0
    for row in rows(package / "composition/typed/examples.parquet"):
        target = expected.pop(row["measurement_id"])
        for field, value in target.items():
            if isinstance(value, str):
                assert row[field] == value
            else:
                error = abs(row[field] - value)
                max_error = max(max_error, error)
                assert error <= 1e-12
        checked += 1
    assert not expected
    for row in rows(package / "composition/typed/decisions.parquet"):
        errors, group = first_decisions.pop(row["measurement_id"])
        if not errors and group in small:
            errors = errors + ["fewer_than_three_distinct_sequences_in_group"]
        assert json.loads(row["reasons_json"]) == errors
        assert row["status"] == ("excluded" if errors else "admitted")
        assert row["group_id"] == group
    assert not first_decisions
    report = {
        "passed": True,
        "n_graph_nodes": len(graph),
        "n_components_all_source": nx.number_connected_components(graph),
        "n_examples_verified": checked,
        "max_target_weight_error": max_error,
        "n_split_components": {s: len(v) for s, v in split_components.items()},
        "baseline_replay": "Every baseline row equals prior independently source-audited generation",
        "source_manifest_sha256": config["source_manifest_sha256"],
    }
    (ROOT / "results/independent-admission.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
