"""Source-to-query and fixed precursor aggregation checks, without predictive fitting."""

import gzip
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def main():
    start = time.perf_counter()
    cfg = read(FAMILY / "reproduce/launch.json")
    receipt = read(FAMILY / "results/cohort.json")
    assert receipt["status"] == "admitted"
    model = Path(receipt["model_input_directory"])
    source = Path(cfg["source_package"]) / "external/pxd057588/typed"
    names = {
        "external_reference_source_mapping": "reference_source_mappings",
        "external_reference_sequence": "reference_sequences",
        "external_reference_occurrence": "reference_occurrences",
        "external_reference_parent": "reference_parents",
        "external_reference_result_context": "reference_result_context",
        "external_assay_channel": "channels",
        "external_contrast": "contrasts",
        "external_derivation_input": "derivation_inputs",
        "external_context_assertion": "source_assertions",
        "external_source_asset": "source_assets",
    }
    n_relation_rows = 0
    for name, info in read(FAMILY / "results/query-relations.json").items():
        original = pq.read_table(source / (names[name] + ".parquet"))
        exported = pq.read_table(info["path"])
        assert original.equals(exported.rename_columns(original.schema.names)), name
        n_relation_rows += original.num_rows
    mappings = {
        r["peptide_row_id"]: r
        for r in pq.read_table(source / "reference_source_mappings.parquet").to_pylist()
    }
    context = {
        r["result_id"]: r
        for r in pq.read_table(source / "reference_result_context.parquet").to_pylist()
    }
    original = {
        r["measurement_id"]: r
        for r in pq.read_table(
            source / "quantity_measurements.parquet",
            filters=[("representation", "=", "reported_log2_fold_change")],
        ).to_pylist()
    }
    channels = {r["channel_id"]: r for r in pq.read_table(source / "channels.parquet").to_pylist()}
    contrasts = {
        r["contrast_id"]: r for r in pq.read_table(source / "contrasts.parquet").to_pylist()
    }
    derived = defaultdict(set)
    for r in pq.read_table(source / "derivation_inputs.parquet").to_pylist():
        derived[r["derived_measurement_id"]].add(r["input_measurement_id"])
    path = model.parent / "response-recipe/sources/external_quantity_measurement/accepted.parquet"
    exported_rows = pq.read_table(path).to_pylist()
    assert len({r["measurement_id"] for r in exported_rows}) == len(exported_rows)
    for r in exported_rows:
        raw = original[r["measurement_id"]]
        assert all(r["native." + k] == v for k, v in raw.items())
        assert all(
            r["reference_context.native." + k] == v for k, v in context[raw["result_id"]].items()
        )
        assert abs(r["target_value"] - raw["reported_value"] / math.log2(10)) < 1e-12
        assert r["target_unit"] == "log10_ratio"
        contrast = contrasts[raw["contrast_id"]]
        ctx = context[raw["result_id"]]
        expected = {
            ctx["peptide_row_id"] + ":" + c.split(":", 2)[2] + ":raw_tmt_intensity"
            for c in [
                contrast["treated_channel_id"],
                *json.loads(contrast["control_channel_ids_json"]),
            ]
        }
        assert expected == derived[raw["measurement_id"]]
    origins = defaultdict(set)
    for r in pq.read_table(source / "reference_occurrences.parquet").to_pylist():
        origins[r["sequence_id"]].add((r["parent_sequence_sha256"], r["start1"] - 1, r["end1"]))
    digest = defaultdict(list)
    for mid, r in original.items():
        c = context[r["result_id"]]
        if c["table_number"] == 11 and c["component"]:
            contrast = contrasts[r["contrast_id"]]
            label = channels[contrast["treated_channel_id"]]["sample_label_raw"]
            arm = "GL261" if label in {"G1+", "G2+", "G3+"} else "CT2A"
            assert label in {"G1+", "G2+", "G3+", "C1+", "C2+", "C3+"}
            digest[c["sequence_id"], arm].append((mid, r["reported_value"] / math.log2(10)))
    expected_parent = defaultdict(set)
    for sid, arm in digest:
        for parent, _, _ in origins[sid]:
            expected_parent[parent, arm].add(sid)
    precursor = read(model / "precursor-estimates.json.gz")
    for r in precursor.values():
        key = r["parent_sequence_sha256"], r["condition"]
        assert expected_parent[key] == {x["sequence_id"] for x in r["contributors"]}
        assert len(expected_parent[key]) >= 2
        values = []
        for seq in r["contributors"]:
            pairs = digest[seq["sequence_id"], key[1]]
            assert sorted(mid for mid, _ in pairs) == seq["measurement_ids"]
            value = statistics.mean(x for _, x in pairs)
            assert abs(value - seq["mean_log10_ratio"]) < 1e-12
            values.append(value)
        assert abs(statistics.median(values) - r["log10_ratio"]) < 1e-12
    groups = read(model / "groups.json.gz")
    parents = read(model / "parents.json.gz")
    source_parent = {
        r["sequence_sha256"]: r["sequence"]
        for r in pq.read_table(source / "reference_parents.parquet").to_pylist()
        if r["sequence"] is not None
    }
    assert all(source_parent[k] == v for k, v in parents.items())
    seen = set()
    for g in groups:
        assert mappings[g["source_peptide_row_id"]] == g["source_mapping"]
        actual = {
            (o["parent_sequence_sha256"], o["start0"], o["end0_exclusive"]) for o in g["origins"]
        }
        assert actual == origins[g["sequence_id"]]
        n = (
            int(hashlib.sha256((cfg["split_salt"] + g["component_id"]).encode()).hexdigest(), 16)
            % 20
        )
        assert g["split"] == ("train" if n < 14 else "val" if n < 17 else "test")
        for o in g["origins"]:
            p = o["parent_sequence_sha256"]
            assert parents[p][o["start0"] : o["end0_exclusive"]] == g["peptide_sequence"]
            assert (
                o["precursor_log10_ratio"] == precursor[p + ":" + g["condition_id"]]["log10_ratio"]
            )
        for obs in g["observations"]:
            mid = obs["measurement_id"]
            assert mid not in seen
            seen.add(mid)
            assert obs["native.reported_value"] == original[mid]["reported_value"]
            assert context[original[mid]["result_id"]]["table_number"] == 9
    for split in ("train", "val", "test"):
        selected = [g for g in groups if g["split"] == split]
        arms = {g["condition_id"] for g in selected}
        for g in selected:
            comps = {h["component_id"] for h in selected if h["condition_id"] == g["condition_id"]}
            count = sum(
                h["condition_id"] == g["condition_id"] and h["component_id"] == g["component_id"]
                for h in selected
            )
            assert abs(g["group_weight"] - 1 / len(arms) / len(comps) / count) < 1e-15
            assert all(
                abs(o["weight"] - g["group_weight"] / len(g["observations"])) < 1e-15
                for o in g["observations"]
            )
        assert abs(math.fsum(o["weight"] for g in selected for o in g["observations"]) - 1) < 1e-12
    result = {
        "status": "passed",
        "n_canonical_relation_rows_verified": n_relation_rows,
        "n_source_response_rows_verified": len(exported_rows),
        "n_precursor_estimates_verified": len(precursor),
        "n_mhc_observations": len(seen),
        "seconds": time.perf_counter() - start,
        "held_out_metrics_evaluated": False,
        "scope": "source export, derivation membership, precursor aggregation, occurrences, weights and fixed split",
    }
    (FAMILY / "results/cohort-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
