"""Independent graph, selection and original-area replay before any model fit."""

import hashlib
import importlib.util
import json
import math
import posixpath
import re
import resource
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pyarrow.parquet as pq
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")
MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token(value):
    pieces = (value or "").split("|")
    key = pieces[1] if pieces[0] in ("sp", "tr") else pieces[0]
    return key if re.fullmatch(r"[A-Z0-9]{6}(?:[A-Z0-9]{4})?(?:-\d+)?", key) else None


def study(filename):
    return "human" if filename[:9] == "PXD051024" else "dc"


def chemical(row):
    body = row["peptide_raw"] or ""
    if re.fullmatch(r"[A-Z-]\..*\.[A-Z-]", body):
        body = body[2:-2]
    pos = 0
    letters = []
    while pos < len(body):
        aa = body[pos]
        if aa not in AA:
            return False
        letters.append(aa)
        pos += 1
        modification = None
        if pos < len(body) and body[pos] == "(":
            end = body.find(")", pos)
            if end < 0:
                return False
            try:
                modification = float(body[pos + 1 : end])
            except ValueError:
                return False
            pos = end + 1
        if aa == "C":
            if modification is None or abs(modification - 57.02) > 0.020000001:
                return False
        elif modification is not None:
            return False
    annotation = (row["peptide_modifications_raw"] or "").strip().lower()
    return (
        bool(letters)
        and "".join(letters) == row["peptide_sequence"]
        and annotation in ("", "carbamidomethylation (c)", "carbamidomethyl (c)")
    )


def main():
    start = time.perf_counter()
    cfg = read(ROOT / "reproduce/launch.json")
    output = ROOT / "results/independent-admission.json"
    assert not output.exists()
    source, recipe = Path(cfg["source_directory"]), Path(cfg["recipe_directory"])
    assert digest(source / "manifest.json") == cfg["source_manifest_sha256"]
    source_manifest, recipe_manifest = (
        read(source / "manifest.json"),
        read(recipe / "manifest.json"),
    )
    for name, entry in recipe_manifest["files"].items():
        assert digest(recipe / name) == entry["sha256"]
    tables = {}
    for path in sorted((recipe / "source_tables").glob("*.parquet")):
        rel = "nanaware_lymph/typed/" + path.name
        assert (
            digest(path)
            == digest(source / rel)
            == source_manifest["files"][rel]["sha256"]
        )
        tables[path.stem] = pq.read_table(path).to_pylist()
    decoder_path = (
        REPO
        / "experiments/2026-10-24_1829_codex_lymph-source-qualification/reproduce/source/verify_workbooks.py"
    )
    spec = importlib.util.spec_from_file_location("original_decoder", decoder_path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    contract = read(source / "nanaware_lymph/contract.json")
    cells_by_block = {}
    for block in contract["blocks"]:
        workbook = source / "nanaware_lymph/source_assets" / block["filename"]
        with zipfile.ZipFile(workbook) as archive:
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                strings = [
                    "".join(n.text or "" for n in item.iter(MAIN + "t"))
                    for item in etree.fromstring(archive.read("xl/sharedStrings.xml"))
                ]
            links = {
                n.get("Id"): n.get("Target")
                for n in etree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            }
            node = next(
                n
                for n in etree.fromstring(archive.read("xl/workbook.xml")).iter(
                    MAIN + "sheet"
                )
                if n.get("name") == block["sheet"]
            )
            target = links[node.get(REL + "id")]
            path = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath("xl/" + target)
            )
            cells_by_block[block["block_id"]] = {
                key: value
                for _, cells in decoder.decode_rows(archive, path, strings, {}, [])
                for key, value in cells.items()
            }
    records = {r["source_record_id"]: r for r in tables["source_records"]}
    for quantity in tables["quantities"]:
        row = records[quantity["source_record_id"]]
        literal = cells_by_block[row["block_id"]].get(
            quantity["source_cell"].split("!")[-1]
        )
        raw = json.loads(row["source_row_json"])["cells"]
        column = re.sub("[0-9]", "", quantity["source_cell"].split("!")[-1])
        assert literal == raw[column]
        if quantity["value_status"] == "positive":
            assert float(literal) == quantity["quantitative_value"] > 0
    refs = {r["reference_record_id"]: r for r in tables["references"]}
    obs = {r["observation_id"]: r for r in tables["observations"]}
    ref_by_key = defaultdict(set)
    evidence = defaultdict(set)
    for key, r in refs.items():
        if token(r["accession_raw"]) and r["reference_sequence"]:
            ref_by_key[study(r["source_file"]), token(r["accession_raw"])].add(key)
    for r in tables["identifications"]:
        if (r["is_decoy_raw"] or "").lower() not in ("true", "1"):
            evidence[study(r["source_file"]), r["peptide_sequence"]].add(
                r["reference_record_id"]
            )
    graph = nx.Graph()
    origins = {}
    parent_seq = {}
    for key, row in obs.items():
        kind = "human" if row["block_id"] == "human_native" else "dc"
        seq = row["peptide_sequence"]
        acc = token(row["parent_label_raw"])
        choices = defaultdict(list)
        for ref in sorted(ref_by_key[kind, acc] | evidence[kind, seq]):
            parent = refs[ref]["reference_sequence"]
            if parent and seq and seq in parent:
                parent_id = hashlib.sha256(parent.encode()).hexdigest()
                parent_seq[parent_id] = parent
                choices[parent_id].append(ref)
        origins[key] = [
            {
                "parent_id": pid,
                "reference_record_ids": ids,
                "occurrences_0based_halfopen": [
                    [m.start(), m.start() + len(seq)]
                    for m in re.finditer("(?=" + re.escape(seq) + ")", parent_seq[pid])
                ],
            }
            for pid, ids in sorted(choices.items())
        ]
        node = "observation:" + key
        graph.add_node(node)
        edges = ["accession:" + kind + ":" + (acc or row["parent_label_raw"] or key)]
        if seq:
            edges.append("peptide:" + seq)
        edges.extend("parent:" + pid for pid in choices)
        graph.add_edges_from((node, other) for other in edges)
    groups = {frozenset(g) for g in nx.connected_components(graph)}
    actual = read(recipe / "dependence_groups.json")
    assert groups == {frozenset(g["nodes"]) for g in actual}
    membership = {node: g["component_id"] for g in actual for node in g["nodes"]}
    split = {}
    for g in actual:
        expected = (
            "nanaware2024:component:"
            + hashlib.sha256(
                json.dumps(
                    [g["nodes"]], sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
        )
        assert expected == g["component_id"]
        bucket = (
            int(
                hashlib.sha256(
                    ("nanaware2024_composition_v1:" + expected).encode()
                ).hexdigest(),
                16,
            )
            % 10
        )
        split[expected] = "train" if bucket < 6 else "val" if bucket < 8 else "test"
        assert g["split"] == split[expected]
    actual_origins = {
        r["observation_id"]: r["candidates"] for r in read(recipe / "origins.json")
    }
    assert origins == actual_origins
    assert parent_seq == {
        r["parent_id"]: r["sequence"] for r in read(recipe / "parents.json")
    }
    choices = defaultdict(list)
    for key, row in obs.items():
        if not chemical(row) or len(origins[key]) != 1:
            continue
        pid = origins[key][0]["parent_id"]
        if not set(parent_seq[pid]) <= AA:
            continue
        score = json.loads(row["confidence_json"]).get("minus_10lgP")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        try:
            charge = float(row["charge_raw"])
        except (ValueError, TypeError):
            charge = math.inf
        if not math.isfinite(charge):
            charge = math.inf
        choices[row["block_id"], row["peptide_sequence"], pid].append(
            (-score, charge, key)
        )
    winners = {min(v)[2] for v in choices.values()}
    chosen = defaultdict(list)
    for row in tables["quantities"]:
        key = row["observation_id"]
        if (
            key in winners
            and row["quantity_kind"] == "area"
            and row["value_status"] == "positive"
        ):
            assert row["quantitative_value"] > 0 and math.isfinite(
                row["quantitative_value"]
            )
            chosen[
                obs[key]["block_id"], row["sample_id"], origins[key][0]["parent_id"]
            ].append(row)
    expected_rows = {
        r["measurement_id"]: r
        for rows in chosen.values()
        if len(rows) >= 3
        for r in rows
    }
    examples = read(recipe / "examples.json")
    assert set(expected_rows) == {r["measurement_id"] for r in examples}
    dispositions = {r["measurement_id"]: r for r in read(recipe / "dispositions.json")}
    assert {key for key, r in dispositions.items() if r["admitted"]} == set(
        expected_rows
    )
    for rows in chosen.values():
        if len(rows) < 3:
            continue
        values = np.array([r["quantitative_value"] for r in rows])
        expected = {
            r["measurement_id"]: (v - np.log10(values).mean(), a / values.sum())
            for r, v, a in zip(rows, np.log10(values), values)
        }
        for row in examples:
            if row["measurement_id"] not in expected:
                continue
            assert (
                row["area"]
                == expected_rows[row["measurement_id"]]["quantitative_value"]
            )
            assert np.allclose(
                [row["target_centered_log10_area"], row["target_area_fraction"]],
                expected[row["measurement_id"]],
                atol=2e-14,
                rtol=0,
            )
            assert (
                row["split"]
                == split[membership["observation:" + row["observation_id"]]]
            )
    for field in ("sequence", "parent_id", "component_id"):
        values = {
            s: {r[field] for r in examples if r["split"] == s}
            for s in ("train", "val", "test")
        }
        assert not (
            values["train"] & values["val"]
            or values["train"] & values["test"]
            or values["val"] & values["test"]
        )
    prepared = recipe.parent / (recipe.name + "_prepared")
    prepared.mkdir()
    support = read(ROOT / "results/admission.json")["cohort_support"]
    for block, info in support.items():
        if not info["supported"]:
            continue
        directory = prepared / block
        directory.mkdir()
        rows = [r for r in examples if r["block_id"] == block]
        covariates = [
            {
                k: v
                for k, v in r.items()
                if k
                not in (
                    "area",
                    "log10_area",
                    "target_centered_log10_area",
                    "target_area_fraction",
                )
            }
            for r in rows
        ]
        (directory / "covariates.json").write_text(
            json.dumps(covariates, indent=2) + "\n"
        )
        (directory / "parents.json").write_text(
            json.dumps(
                {pid: parent_seq[pid] for pid in {r["parent_id"] for r in rows}},
                indent=2,
            )
            + "\n"
        )
        for name in ("train", "val", "test"):
            labels = {
                r["measurement_id"]: {
                    k: r[k]
                    for k in (
                        "area",
                        "target_centered_log10_area",
                        "target_area_fraction",
                    )
                }
                for r in rows
                if r["split"] == name
            }
            (directory / (name + "-labels.json")).write_text(
                json.dumps(labels, indent=2) + "\n"
            )
    result = {
        "status": "passed",
        "n_original_quantity_cells": len(tables["quantities"]),
        "n_original_observations": len(obs),
        "n_verified_targets": len(examples),
        "n_all_source_components": len(groups),
        "prepared_directory": str(prepared),
        "source_reference_replay": "Reused1903original-reference audit bound by unchanged source manifest and source table hashes",
        "files": {
            str(p.relative_to(prepared)): digest(p)
            for p in sorted(prepared.rglob("*"))
            if p.is_file()
        },
        "decoder_sha256": digest(decoder_path),
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
    }
    assert result["runtime_seconds"] < 600 and result["peak_rss_gib"] < 3
    output.write_text(json.dumps(result, indent=2) + "\n")
    shutil.copy2(__file__, ROOT / "reproduce/source/verify_admission.py")
    shutil.copy2(decoder_path, ROOT / "reproduce/source/original_xml_decoder.py")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
