"""Fixed model-input aggregation over canonical response/relationship exports."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def write(path, value):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "wt") as stream:
        json.dump(value, stream, allow_nan=False, indent=None if str(path).endswith("gz") else 2)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relation(root, name):
    return [
        {k.removeprefix("native."): v for k, v in row.items()}
        for row in pq.read_table(root / "relations" / (name + ".parquet")).to_pylist()
    ]


def condition(channel):
    labels = {
        "G1+": "GL261",
        "G2+": "GL261",
        "G3+": "GL261",
        "C1+": "CT2A",
        "C2+": "CT2A",
        "C3+": "CT2A",
    }
    assert channel["table_number"] in (9, 11)
    assert not channel["is_control_for_table"]
    return labels[channel["sample_label_raw"]]


def main():
    start = time.perf_counter()
    cfg = read(FAMILY / "reproduce/launch.json")
    exported = read(FAMILY / "results/export.json")
    assert exported["status"] == "completed"
    root = Path(exported["destination"])
    output = root / "model-inputs"
    assert not output.exists()
    output.mkdir()
    for info in read(FAMILY / "results/query-relations.json").values():
        assert sha(info["path"]) == info["sha256"]
    recipe = root / "response-recipe"
    assert sha(recipe / "manifest.json") == exported["recipe_manifest_sha256"]
    manifest = read(recipe / "manifest.json")
    assert manifest["recipe"] == read(FAMILY / "reproduce/recipe.json")
    for name, item in manifest["files"].items():
        assert sha(recipe / name) == item["sha256"]
    mappings = {r["peptide_row_id"]: r for r in relation(root, "external_reference_source_mapping")}
    parents = {}
    for row in relation(root, "external_reference_parent"):
        key = row["sequence_sha256"]
        assert sha_sequence(row["sequence"]) == key
        if key in parents:
            assert parents[key] == row["sequence"]
        parents[key] = row["sequence"]
    origins = defaultdict(set)
    parent_components = defaultdict(set)
    for row in relation(root, "external_reference_occurrence"):
        key, a, b = row["parent_sequence_sha256"], row["start1"] - 1, row["end1"]
        assert parents[key][a:b] == row["peptide_sequence"]
        origins[row["sequence_id"]].add((key, a, b))
        parent_components[key].add(row["component"])
    assert all(len(v) == 1 for v in parent_components.values())
    channels = {r["channel_id"]: r for r in relation(root, "external_assay_channel")}
    contrasts = {r["contrast_id"]: r for r in relation(root, "external_contrast")}
    assertions = relation(root, "external_context_assertion")
    for table in (9, 11):
        relevant = [r for r in assertions if r["subject_id"] == f"pxd057588:t{table}"]
        assert any(
            "G+ denotes macrophages post co-culture with GL261" in r["value_raw"]
            and "CT2A" in r["value_raw"]
            for r in relevant
        )
    # Admission statistics and input preparation only; no predictive fit/held-out metric.
    accepted = pq.read_table(
        recipe / "sources/external_quantity_measurement/accepted.parquet",
        filters=[("reference_context.native.table_number", "in", [9, 11])],
    ).to_pylist()
    digest, mhc = defaultdict(list), defaultdict(list)
    for row in accepted:
        pid = row["reference_context.native.peptide_row_id"]
        mapping = mappings[pid]
        table = row["reference_context.native.table_number"]
        assert mapping["table_number"] == table
        assert mapping["component"] == row["reference_context.native.component"]
        assert row["reference_context.native.organism_taxon_id"] == "10090"
        assert row["reference_context.native.measured_population"] == "macrophage"
        assert row["reference_context.native.specimen_status"] == "unresolved"
        assert row["target_unit"] == "log10_ratio"
        assert row["target_value"] == row["native.reported_value"] * math.log10(2)
        assert row["measurement_id"] == row["native.measurement_id"]
        contrast = contrasts[row["native.contrast_id"]]
        assert contrast["table_number"] == table
        channel = channels[contrast["treated_channel_id"]]
        assert all(
            channels[c]["is_control_for_table"]
            for c in json.loads(contrast["control_channel_ids_json"])
        )
        c = condition(channel)
        row["model_condition"] = c
        row["source_channel"] = channel
        row["source_contrast"] = contrast
        if table == 11:
            digest[(mapping["sequence_id"], c)].append(row)
        else:
            mhc[(pid, c)].append(row)
    estimates = defaultdict(dict)
    for (sid, c), rows in digest.items():
        if sid not in origins:
            continue
        mean = math.fsum(r["target_value"] for r in rows) / len(rows)
        item = {
            "sequence_id": sid,
            "mean_log10_ratio": mean,
            "measurement_ids": sorted(r["measurement_id"] for r in rows),
            "peptide_row_ids": sorted({r["reference_context.native.peptide_row_id"] for r in rows}),
            "n_candidate_parent_sequences": len({p for p, _, _ in origins[sid]}),
        }
        for parent in {p for p, _, _ in origins[sid]}:
            estimates[(parent, c)][sid] = item
    precursor = {}
    for (parent, c), sequences in estimates.items():
        if len(sequences) >= 2:
            precursor[parent + ":" + c] = {
                "parent_sequence_sha256": parent,
                "condition": c,
                "log10_ratio": float(
                    np.median([r["mean_log10_ratio"] for r in sequences.values()])
                ),
                "n_distinct_digest_sequences": len(sequences),
                "contributors": sorted(sequences.values(), key=lambda r: r["sequence_id"]),
                "material_link_precision": "condition_surrogate_unresolved_specimen",
            }
    groups, disposition = [], []
    for (pid, c), rows in sorted(mhc.items()):
        mapping = mappings[pid]
        locations = sorted(origins.get(mapping["sequence_id"], set()))
        reasons = []
        if not locations:
            reasons.append("no_full_parent_occurrence")
        if not set(mapping["sequence_raw"]) <= AA:
            reasons.append("noncanonical_peptide")
        distinct = sorted({p for p, _, _ in locations})
        if any(not set(parents[p]) <= AA for p in distinct):
            reasons.append("noncanonical_candidate_parent")
        missing = [p for p in distinct if p + ":" + c not in precursor]
        if missing:
            reasons.append("candidate_parent_without_two_digest_sequences")
        component = mapping["component"]
        assert component and all(parent_components[p] == {component} for p in distinct)
        bucket = int(hashlib.sha256((cfg["split_salt"] + component).encode()).hexdigest(), 16) % 20
        split = "train" if bucket < 14 else "val" if bucket < 17 else "test"
        decision = {
            "peptide_row_id": pid,
            "condition": c,
            "component_id": component,
            "split": split,
            "included": not reasons,
            "reasons": reasons,
            "missing_precursor_parent_sequences": missing,
            "measurement_ids": [r["measurement_id"] for r in rows],
        }
        disposition.append(decision)
        if reasons:
            continue
        groups.append(
            {
                "response_group_id": pid + ":" + c,
                "source_peptide_row_id": pid,
                "sequence_id": mapping["sequence_id"],
                "peptide_sequence": mapping["sequence_raw"],
                "condition_id": c,
                "condition": ["mouse", "macrophage", c],
                "modifications": [],
                "component_id": component,
                "split": split,
                "source_mapping": mapping,
                "origins": [
                    {
                        "parent_sequence_sha256": p,
                        "start0": a,
                        "end0_exclusive": b,
                        "precursor_log10_ratio": precursor[p + ":" + c]["log10_ratio"],
                    }
                    for p, a, b in locations
                ],
                "precursor_only_log10_ratio": math.log10(
                    math.fsum(10 ** precursor[p + ":" + c]["log10_ratio"] for p in distinct)
                    / len(distinct)
                ),
                "observations": rows,
            }
        )
    coverage = {}
    for split in ("train", "val", "test"):
        selected = [g for g in groups if g["split"] == split]
        condition_counts = Counter(g["condition_id"] for g in selected)
        component_sets = {
            c: {g["component_id"] for g in selected if g["condition_id"] == c}
            for c in condition_counts
        }
        row_counts = Counter((g["condition_id"], g["component_id"]) for g in selected)
        for group in selected:
            c = group["condition_id"]
            group["group_weight"] = (
                1
                / len(condition_counts)
                / len(component_sets[c])
                / row_counts[c, group["component_id"]]
            )
            for row in group["observations"]:
                row["weight"] = group["group_weight"] / len(group["observations"])
        coverage[split] = {
            "n_peptide_rows": len({g["source_peptide_row_id"] for g in selected}),
            "n_response_groups": len(selected),
            "n_components": len({g["component_id"] for g in selected}),
            "n_observations": sum(len(g["observations"]) for g in selected),
            "n_conditions": len(condition_counts),
            "weight_sum": math.fsum(g["group_weight"] for g in selected),
        }
    used_parents = {o["parent_sequence_sha256"] for g in groups for o in g["origins"]}
    used_precursor = {
        p + ":" + g["condition_id"]
        for g in groups
        for p in {o["parent_sequence_sha256"] for o in g["origins"]}
    }
    write(output / "groups.json.gz", groups)
    write(output / "parents.json.gz", {p: parents[p] for p in sorted(used_parents)})
    write(output / "precursor-estimates.json.gz", {k: precursor[k] for k in sorted(used_precursor)})
    write(output / "dispositions.json.gz", disposition)
    result = {
        "status": "admitted"
        if all(
            coverage[s]["n_peptide_rows"] >= n
            for s, n in (("train", 50), ("val", 20), ("test", 20))
        )
        else "insufficient_fixed_split_support",
        "splits": coverage,
        "n_total_table9_peptide_rows": len({p for p, _ in mhc}),
        "n_source_measurements_table9_and_11": len(accepted),
        "n_admitted_peptide_rows": len({g["source_peptide_row_id"] for g in groups}),
        "exclusion_group_counts": dict(Counter(r for d in disposition for r in d["reasons"])),
        "n_used_parent_sequences": len(used_parents),
        "n_parent_condition_surrogates": len(used_precursor),
        "n_reference_alternatives_retained": sum(len(g["origins"]) for g in groups),
        "model_input_directory": str(output),
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "source_recipe_sha256": exported["recipe_manifest_sha256"],
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "held_out_metrics_evaluated": False,
        "scope": "MHC-bound APC response with condition-surrogate digest input; not free concentrations",
    }
    write(FAMILY / "results/cohort.json", result)
    print(json.dumps(result), flush=True)


def sha_sequence(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


if __name__ == "__main__":
    main()
