"""Reconcile shared source evidence with independent original XML decoders."""

import gzip
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
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = REPO / "data/lymph_source_qualification/2026-10-24_1829_v1"
PACKAGE = REPO / "data/lymph_source_qualification/2026-10-24_1903_canonical"
MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def state(value):
    if value is None or value == "":
        return None, "missing"
    if isinstance(value, str) and value.startswith("="):
        return None, "formula_not_evaluated"
    if isinstance(value, bool):
        return None, "invalid_literal"
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None, "invalid_literal"
    if not math.isfinite(number):
        return None, "nonfinite"
    return number, "positive" if number > 0 else "zero" if number == 0 else "negative"


def main():
    started = time.perf_counter()
    target = ROOT / "results/independent-verification.json"
    assert not target.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/verify.py")
    decoder_path = (
        REPO
        / "experiments/2026-10-24_1829_codex_lymph-source-qualification/reproduce/source/verify_workbooks.py"
    )
    snapshot = ROOT / "reproduce/source/original_xml_decoder.py"
    shutil.copy2(decoder_path, snapshot)
    spec = importlib.util.spec_from_file_location("original_xml_decoder", snapshot)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    contract = json.loads((PACKAGE / "nanaware_lymph/contract.json").read_text())
    for asset in contract["assets"]:
        data = (RAW / asset["filename"]).read_bytes()
        assert len(data) == asset["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == asset["sha256"]
        assert (
            PACKAGE / "nanaware_lymph/source_assets" / asset["filename"]
        ).read_bytes() == data
    tables = {
        path.stem: pq.read_table(path).to_pylist()
        for path in (PACKAGE / "nanaware_lymph/typed").glob("*.parquet")
    }
    records = {row["source_record_id"]: row for row in tables["source_records"]}
    observations = {row["source_record_id"]: row for row in tables["observations"]}
    samples = {row["sample_id"]: row for row in tables["samples"]}
    assert len(records) == len(tables["source_records"]) == len(observations)
    quantities = {}
    for row in tables["quantities"]:
        key = (row["source_record_id"], row["source_cell"])
        assert key not in quantities
        quantities[key] = row
    checked_quantities = checked_rows = checked_cells = 0
    block_counts = []
    for block in contract["blocks"]:
        expected = {}
        with zipfile.ZipFile(RAW / block["filename"]) as archive:
            assert archive.testzip() is None
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                for item in etree.fromstring(archive.read("xl/sharedStrings.xml")):
                    strings.append(
                        "".join(node.text or "" for node in item.iter(MAIN + "t"))
                    )
            links = {
                node.get("Id"): node.get("Target")
                for node in etree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            }
            sheet = next(
                node
                for node in etree.fromstring(archive.read("xl/workbook.xml")).iter(
                    MAIN + "sheet"
                )
                if node.get("name") == block["sheet"]
            )
            link = links[sheet.get(REL + "id")]
            path = (
                link.lstrip("/")
                if link.startswith("/")
                else posixpath.normpath("xl/" + link)
            )
            for number, cells in decoder.decode_rows(archive, path, strings, {}, []):
                if number > 2 and cells.get(
                    block["columns"]["peptide"] + str(number)
                ) not in (None, ""):
                    expected[number] = cells
        emitted = [
            row for row in records.values() if row["block_id"] == block["block_id"]
        ]
        assert {row["source_row_index"] for row in emitted} == set(expected)
        counter = Counter()
        for row in emitted:
            index = row["source_row_index"]
            cells = expected[index]
            saved = json.loads(row["source_row_json"])
            assert saved["headers"] == block["headers"]
            for column, value in saved["cells"].items():
                original = cells.get(column + str(index))
                assert original == value, (
                    block["block_id"],
                    index,
                    column,
                    original,
                    value,
                )
                assert isinstance(original, bool) == isinstance(value, bool)
                checked_cells += 1
            observation = observations[row["source_record_id"]]
            peptide = cells[block["columns"]["peptide"] + str(index)]
            assert observation["peptide_raw"] == peptide
            assert observation["block_id"] == block["block_id"]
            for quantity in block["quantities"]:
                original = cells.get(quantity["column"] + str(index))
                cell = f"{block['sheet']}!{quantity['column']}{index}"
                actual = quantities[(row["source_record_id"], cell)]
                number, value_status = state(original)
                assert actual["quantitative_value"] == number
                assert actual["value_status"] == value_status
                assert actual["sample_id"] == quantity["sample_id"]
                assert (
                    actual["context_id"] == samples[actual["sample_id"]]["context_id"]
                )
                assert actual["observation_id"] == observation["observation_id"]
                assert actual["quantity_kind"] == quantity["kind"]
                assert actual["peptide_sequence"] == observation["peptide_sequence"]
                assert actual["parent_label_raw"] == observation["parent_label_raw"]
                assert (
                    actual["eligibility_status"]
                    == "source_evidence_not_training_admitted"
                )
                if original is None:
                    assert actual["quantitative_raw"] == ""
                elif number is not None:
                    assert float(actual["quantitative_raw"]) == number
                else:
                    assert actual["quantitative_raw"] == str(original)
                counter[(quantity["kind"], value_status)] += 1
                checked_quantities += 1
            checked_rows += 1
        block_counts.append(
            {
                "block": block["block_id"],
                "n_source_rows": len(emitted),
                "quantity_states": {
                    kind + ":" + status: count
                    for (kind, status), count in sorted(counter.items())
                },
            }
        )
    assert checked_rows == len(records)
    assert checked_quantities == len(quantities)
    references = {
        (row["source_file"], row["source_locator"]): row for row in tables["references"]
    }
    identifications = {
        (row["source_file"], row["source_locator"]): row
        for row in tables["identifications"]
    }
    checked_references = checked_identifications = 0
    for asset in contract["assets"]:
        if asset["role"] != "original_identifications_and_reference_sequences":
            continue
        filename = asset["filename"]
        content = (RAW / filename).read_bytes()
        root = etree.fromstring(
            gzip.decompress(content) if filename.endswith(".gz") else content
        )
        ns = {"m": etree.QName(root).namespace}
        peptides = {
            node.get("id"): node for node in root.xpath("//m:Peptide", namespaces=ns)
        }
        for node in root.xpath("//m:DBSequence", namespaces=ns):
            actual = references[(filename, "DBSequence:" + node.get("id"))]
            sequence = node.findtext("m:Seq", namespaces=ns)
            assert actual["reference_sequence"] == sequence
            assert actual["accession_raw"] == node.get("accession")
            assert actual["source_row_json"] and json.loads(
                actual["source_row_json"]
            ) == dict(node.attrib)
            assert actual["sequence_sha256"] == (
                hashlib.sha256(sequence.encode()).hexdigest() if sequence else None
            )
            checked_references += 1
        for node in root.xpath("//m:PeptideEvidence", namespaces=ns):
            actual = identifications[(filename, "PeptideEvidence:" + node.get("id"))]
            reference = references[
                (filename, "DBSequence:" + node.get("dBSequence_ref"))
            ]
            peptide = peptides[node.get("peptide_ref")]
            assert actual["reference_record_id"] == reference["reference_record_id"]
            assert actual["peptide_sequence"] == peptide.findtext(
                "m:PeptideSequence", namespaces=ns
            )
            assert json.loads(actual["source_row_json"]) == dict(node.attrib)
            modifications = [
                {
                    "attributes": dict(m.attrib),
                    "parameters": [dict(c.attrib) for c in m],
                }
                for m in peptide.findall("m:Modification", namespaces=ns)
            ]
            assert json.loads(actual["peptide_modifications_raw"]) == modifications
            checked_identifications += 1
    assert checked_references == len(references) == len(tables["references"])
    assert (
        checked_identifications
        == len(identifications)
        == len(tables["identifications"])
    )
    inventories = {}
    for dataset in ("PXD051024", "PXD044801", "PXD044885"):
        items = json.loads((RAW / f"{dataset}-files-all.json").read_text())
        count = json.loads((RAW / f"{dataset}-files-count.json").read_text())
        assert len(items) == count == len({item["fileName"] for item in items})
        inventories[dataset] = {
            "n_files": count,
            "names": sorted(item["fileName"] for item in items),
        }
    for dataset in ("MSV000094586", "MSV000094761"):
        html = (RAW / f"{dataset}-files.html").read_text()
        match = re.search(r"var dataset_files\s*=\s*", html)
        first, _ = json.JSONDecoder().raw_decode(html[match.end() :])
        script = next(
            s.get_text()
            for s in BeautifulSoup(html, "html.parser").find_all("script")
            if "var dataset_files =" in s.get_text()
        )
        second = json.loads(script.split("var dataset_files =", 1)[1].split(";", 1)[0])
        assert first == second
        assert len(first["row_data"]) == int(first["total_rows"])
        assert len({row["file_descriptor"] for row in first["row_data"]}) == len(
            first["row_data"]
        )
        inventories[dataset] = {
            "n_files": len(first["row_data"]),
            "collections": dict(
                Counter(row["collection"] for row in first["row_data"])
            ),
        }
    result = {
        "status": "passed",
        "n_assets": len(contract["assets"]),
        "n_source_rows": checked_rows,
        "n_original_cells": checked_cells,
        "n_quantity_cells": checked_quantities,
        "n_reference_entries": checked_references,
        "n_identification_alternatives": checked_identifications,
        "blocks": block_counts,
        "repository_inventories": inventories,
        "model_training_or_predictions": False,
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    assert result["seconds"] <= 600 and result["peak_rss_gib"] <= 3
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
