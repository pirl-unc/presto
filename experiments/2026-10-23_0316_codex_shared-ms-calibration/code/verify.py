"""Independent original-XML, typed-value and prior-generation conservation audit.

No Hitlist imports, source writer calls, or openpyxl. Does not rewrite source data.
"""

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import resource
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pyarrow.parquet as pq

from xml_reference import NS, read_xml, canonical, numeric, colno

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "size_bytes": Path(path).stat().st_size}


def main(report):
    assert json.loads((ROOT / "results/execution.json").read_text())["passed"]
    source = Path(CONFIG["output_package"])
    prior = Path(CONFIG["prior_mapp_package"])
    parent = Path(CONFIG["reference_package"])
    manifest = json.loads((source / "manifest.json").read_text())
    before = json.loads((prior / "manifest.json").read_text())
    assert (
        digest(prior / "manifest.json")["sha256"]
        == CONFIG["prior_mapp_manifest_sha256"]
    )
    assert (
        digest(parent / "manifest.json")["sha256"]
        == CONFIG["reference_manifest_sha256"]
    )
    assert manifest["parent_manifest_sha256"] == CONFIG["reference_manifest_sha256"]
    assert manifest["schema_version"] == 8 and manifest["catalog_schema_version"] == 11
    assert set(manifest["appended_sources"]) == {"pxd009092", "pxd004886"}
    inventory = {
        str(p.relative_to(source)): digest(p)
        for p in source.rglob("*")
        if p.is_file() and p != source / "manifest.json"
    }
    assert inventory == manifest["files"]
    for name, checksum in before["files"].items():
        assert checksum == inventory[name] == digest(prior / name), name
        a, b = (prior / name).stat(), (source / name).stat()
        assert (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino), name
    assert (
        manifest["appended_sources"]["pxd009092"]
        == before["appended_sources"]["pxd009092"]
    )
    report["prior_files_identical_with_independent_inodes"] = len(before["files"])
    report["new_generation_inventory_files"] = len(inventory)
    external = source / "external/pxd004886"
    tables = {
        p.stem: pq.read_table(p).to_pylist()
        for p in (external / "typed").glob("*.parquet")
    }
    report["relation_rows"] = {k: len(v) for k, v in sorted(tables.items())}
    assert len(tables) == 13
    source_cells = {r["cell_id"]: r for r in tables["source_cells"]}
    assert len(source_cells) == len(tables["source_cells"]) == 13561
    originals, nodes = {}, {}
    for sheet in tables["source_sheets"]:
        asset = sheet["asset_path"]
        p = Path(CONFIG["source_roots"]["pxd004886"]) / asset
        assert digest(p) == digest(external / "assets" / asset)
        books, _ = read_xml(p)
        assert list(books) == [sheet["sheet_name"]]
        originals[asset] = books[sheet["sheet_name"]]
        with zipfile.ZipFile(p) as archive:
            wb = ET.fromstring(archive.read("xl/workbook.xml"))
            sn = wb.find("s:sheets/s:sheet", NS)
            rels = {
                x.attrib["Id"]: x.attrib["Target"]
                for x in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            }
            target = rels[
                sn.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
            ]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            xml = ET.fromstring(archive.read(target))
            assert sheet["sheet_state"] == sn.attrib.get("state", "visible")
            assert json.loads(sheet["merged_ranges_json"]) == sorted(
                x.attrib["ref"] for x in xml.findall("s:mergeCells/s:mergeCell", NS)
            )
            nodes[asset] = {
                x.attrib["r"]: x for x in xml.findall(".//s:sheetData/s:row/s:c", NS)
            }
        assert sheet["workbook_title"] == originals[asset]["A1"]["value"]
    assert sum(map(len, originals.values())) == len(source_cells)
    for cell in source_cells.values():
        orig = originals[cell["asset_path"]][cell["cell_ref"]]
        assert json.loads(cell["value_json"]) == orig["value"]
        assert cell["formula"] == (
            "=" + orig["formula"] if orig["formula"] is not None else None
        )
        assert cell["xml_element"] == ET.canonicalize(
            ET.tostring(
                nodes[cell["asset_path"]][cell["cell_ref"]], encoding="unicode"
            ),
            rewrite_prefixes=True,
        )
        match = re.fullmatch(r"([A-Z]+)([0-9]+)", cell["cell_ref"])
        assert cell["row_1based"] == int(match[2]) and cell["column_1based"] == colno(
            match[1]
        )

    def original(cell_id):
        c = source_cells[cell_id]
        return originals[c["asset_path"]][c["cell_ref"]]

    def value(cell_id):
        return original(cell_id)["value"] if cell_id else None

    def at(asset, coordinate):
        return originals[asset].get(coordinate, {}).get("value")

    def verify_value(row, value_key="value", locator=None):
        if locator is None:
            c = source_cells[row["source_cell"]]
            asset, coordinate = c["asset_path"], c["cell_ref"]
        else:
            asset, coordinate = locator.split("#")
        present = coordinate in originals[asset]
        cell = originals[asset].get(coordinate, {})
        v = cell.get("value")
        status = (
            "absent_source_cell"
            if not present
            else "explicit_empty_string"
            if v == ""
            else "empty_formula_cache"
            if v is None and cell.get("formula") is not None
            else "empty_source_value"
            if v is None
            else "reported_zero"
            if numeric(v) and v == 0
            else "reported_positive"
            if numeric(v) and v > 0
            else "reported_negative"
            if numeric(v)
            else "reported_nonnumeric"
        )
        assert row["value_status"] == status, (asset, coordinate, status, row)
        assert bool(row["source_cell"]) == present
        assert row[value_key] == (float(v) if numeric(v) else None), (asset, coordinate)
        if present:
            c = source_cells[row["source_cell"]]
            assert c["asset_path"] == asset and c["cell_ref"] == coordinate
        return v

    for cell in source_cells.values():
        verify_value(
            {**cell, "source_cell": cell["cell_id"]},
            locator=cell["asset_path"] + "#" + cell["cell_ref"],
        )
        orig = original(cell["cell_id"])
        expected_type = (
            "f"
            if orig["formula"] is not None
            else {"inlineStr": "s"}.get(orig["xml_type"], orig["xml_type"])
        )
        assert cell["excel_type"] == expected_type

    standards = {r["standard_id"]: r for r in tables["nominal_standards"]}
    loads = {r["nominal_load_id"]: r for r in tables["nominal_loads"]}
    reporters = {r["reporter_id"]: r for r in tables["reporter_forms"]}
    slots = {r["slot_id"]: r for r in tables["source_slots"]}
    assert (len(standards), len(loads), len(reporters), len(slots)) == (30, 25, 32, 99)
    assert Counter(r["group_label"] for r in standards.values()) == dict.fromkeys(
        "ABCDE", 6
    )
    for row in standards.values():
        assert row["form_label_raw"] == value(row["form_source_cell"])
        assert row["peptide_sequence"] == canonical(row["form_label_raw"], panel=True)
        assert row["protein_label_raw"] == value(row["protein_source_cell"])
        assert row["group_label"] == value(row["group_source_cell"])
    for row in loads.values():
        assert row["value"] == value(row["source_cell"])
        assert row["sample_label"] == value(row["sample_source_cell"])
        assert row["group_label"] == value(row["group_source_cell"])
        assert row["unit"] == "fmol_on_column"
    for row in reporters.values():
        assert (
            row["peptide_sequence"]
            == canonical(row["label_raw"])
            == standards[row["standard_id"]]["peptide_sequence"]
        )
        assert json.loads(row["annotations_json"]) == re.findall(
            r"\[([^]]+)\]", row["label_raw"]
        )
        for cell_id in json.loads(row["source_cells_json"]):
            assert value(cell_id) == row["label_raw"]
    for row in slots.values():
        assert int(row["site_label"]) == value(row["site_source_cell"])
        assert "S4" in value(row["sample_source_cell"])
        header = source_cells[row["site_source_cell"]]["cell_ref"]
        header_col = re.match("[A-Z]+", header)[0]
        assert (
            row["source_slot_ordinal"]
            == colno(row["source_column"]) - colno(header_col) + 1
        )
        assert 1 <= row["source_slot_ordinal"] <= 9
        assert (
            row["run_identity_status"]
            == row["day_status"]
            == row["specimen_status"]
            == "unreported"
        )
    by_form = defaultdict(list)
    by_slot = defaultdict(dict)
    s4_labels = set()
    for row in tables["s4_measurements"]:
        v = verify_value(row, locator=row["source_locator"])
        rep, slot, load = (
            reporters[row["reporter_id"]],
            slots[row["slot_id"]],
            loads[row["nominal_load_id"]],
        )
        asset, coord = row["source_locator"].split("#")
        row_number = re.search("[0-9]+", coord)[0]
        assert asset == "supplementary-data-4.xlsx"
        assert at(asset, "A" + row_number) == rep["label_raw"]
        assert re.match("[A-Z]+", coord)[0] == slot["source_column"]
        assert (
            row["reported_load"]
            == value(row["reported_load_source_cell"])
            == at(asset, "B" + row_number)
        )
        assert (
            load["sample_label"] == "S4"
            and load["group_label"] == standards[rep["standard_id"]]["group_label"]
        )
        assert row["representation"] == (
            "raw" if int(row_number) < 39 else "normalized"
        )
        by_form[row["reporter_id"], row["representation"]].append(v)
        key = (row["reporter_id"], row["representation"])
        assert key not in by_slot[row["slot_id"]]
        by_slot[row["slot_id"]][key] = v
        s4_labels.add(rep["label_raw"])
    assert len(s4_labels) == 31 and len(tables["s4_measurements"]) == 6138
    computed = 0
    for row in tables["summary_measurements"]:
        v = verify_value(row)
        vals = np.array(
            [
                x
                for x in by_form[row["reporter_id"], row["representation"]]
                if numeric(x)
            ]
        )
        if len(vals) > 1:
            result = {
                "mean": vals.mean(),
                "sample_standard_deviation": vals.std(ddof=1),
                "coefficient_of_variation_percent": 100
                * vals.std(ddof=1)
                / vals.mean(),
            }[row["statistic_name"]]
            assert math.isclose(row["recomputed_value"], result, rel_tol=1e-12)
            assert math.isclose(v, result, rel_tol=1e-10)
            computed += 1
    assert computed == 162
    curves = {r["curve_id"]: r for r in tables["curve_rows"]}
    assert len(curves) == 682
    curve_labels = set()
    curve_positions = {}
    for row in curves.values():
        rep = reporters[row["reporter_id"]]
        assert value(row["label_source_cell"]) == rep["label_raw"]
        assert (
            value(row["group_source_cell"])
            == row["group_label"]
            == standards[rep["standard_id"]]["group_label"]
        )
        formula = "=" + original(row["rsq_source_cell"])["formula"]
        assert formula == row["rsq_formula"]
        pairs = re.findall(r"RSQ\(([^,()]+),([^,()]+)\)", formula)
        assert pairs and len(set(pairs)) == 1
        header, response = pairs[0]
        assert (header, response) == (
            row["header_range_raw"],
            row["response_range_raw"],
        )
        ranges = [re.findall(r"\$?([A-Z]+)\$?([0-9]+)", x) for x in (header, response)]
        assert all(len(x) == 2 and x[0][1] == x[1][1] for x in ranges)
        assert [x[0][0] for x in ranges] == [ranges[0][0][0]] * 2
        start, end = colno(ranges[0][0][0]), colno(ranges[0][1][0])
        assert end - start == 4
        curve_positions[row["curve_id"]] = [
            (chr(64 + i) + ranges[0][0][1], chr(64 + i) + ranges[1][0][1])
            for i in range(start, end + 1)
        ]
        curve_labels.add(rep["label_raw"])
        assert row["sample_join_status"] == "unresolved_plot_grid_to_nominal_sample"
        assert row["reported_r_squared"] == (
            value(row["rsq_source_cell"])
            if numeric(value(row["rsq_source_cell"]))
            else None
        )
    assert len(curve_labels) == 31 and len(curve_labels | s4_labels) == 32
    assert s4_labels - curve_labels == {"SMD2.NNTQVLINC[CAM]R[+10]"}
    assert curve_labels - s4_labels == {"SMD2.NNTQVLINC[CAM]R[+10].sum4"}
    positions = set()
    for row in tables["curve_measurements"]:
        verify_value(row, locator=row["source_locator"])
        curve = curves[row["curve_id"]]
        assert (
            row["reporter_id"] == curve["reporter_id"]
            and row["representation"] == curve["representation"]
        )
        grid, response = curve_positions[row["curve_id"]][
            row["source_grid_ordinal"] - 1
        ]
        assert row["source_locator"] == "supplementary-data-8.xlsx#" + response
        assert source_cells[row["grid_amount_source_cell"]]["cell_ref"] == grid
        assert row["reported_grid_amount"] == value(row["grid_amount_source_cell"])
        key = (row["curve_id"], row["source_grid_ordinal"])
        assert key not in positions
        positions.add(key)
    assert len(positions) == 3410
    normalized_only = nonuniform = 0
    for row in tables["normalization_diagnostics"]:
        values = by_slot[row["slot_id"]]
        ids = {key[0] for key in values}
        ratios, only_norm, only_raw = [], 0, 0
        for key in ids:
            raw, norm = values[key, "raw"], values[key, "normalized"]
            rp, np_ = numeric(raw) and raw > 0, numeric(norm) and norm > 0
            if rp and np_:
                ratios.append(norm / raw)
            elif rp:
                only_raw += 1
            elif np_:
                only_norm += 1
        assert (
            row["n_paired_reporters"],
            row["n_normalized_only"],
            row["n_raw_only"],
        ) == (len(ratios), only_norm, only_raw)
        normalized_only += only_norm
        if ratios:
            median = float(np.median(ratios))
            deviation = max(abs(x / median - 1) for x in ratios)
            assert math.isclose(row["median_ratio"], median, rel_tol=1e-12)
            assert math.isclose(row["max_relative_deviation"], deviation, abs_tol=1e-12)
            nonuniform += deviation > 1e-6
    assert normalized_only == 26 and nonuniform == 39
    protocol = tables["protocol_parameters"]
    assert len(protocol) == 520
    for row in protocol:
        v = verify_value(row, locator=row["source_locator"])
        assert json.loads(row["value_json"]) == v
        if row["site_source_cell"]:
            assert int(row["site_label"]) == value(row["site_source_cell"])
        if row["parameter_source_cell"]:
            assert row["parameter_name_raw"] == value(row["parameter_source_cell"])
        if row["stage_source_cell"]:
            assert row["protocol_stage"] == value(row["stage_source_cell"])
        if row["position_source_cell"]:
            assert row["position_value"] == value(row["position_source_cell"])
    assert Counter(r["scope"] for r in protocol) == {
        "site_LC_MS_configuration": 33,
        "reported_column_footnote": 2,
        "source_stage_and_site_gradient": 165,
        "common_SWATH_window_protocol": 320,
    }
    assert (
        len(
            {
                r["position_value"]
                for r in protocol
                if r["scope"] == "common_SWATH_window_protocol"
            }
        )
        == 64
    )
    assert (
        len(
            {
                r["protocol_stage"]
                for r in protocol
                if r["scope"] == "source_stage_and_site_gradient"
            }
        )
        == 2
    )
    article = ET.parse(external / "assets/article.xml")
    anchors = {n.attrib["id"] for n in article.iter() if "id" in n.attrib}
    for row in tables["source_assertions"]:
        for loc in row["source_locator"].split("|"):
            if loc.startswith("article.xml#"):
                assert loc.split("#")[1] in anchors
            if ".xlsx#" in loc:
                asset, coordinate = loc.split("#")
                assert coordinate in originals[asset]
    injection = [
        r
        for r in tables["source_assertions"]
        if r["predicate"] == "reported_common_injection_volume"
    ]
    assert len(injection) == 1 and injection[0]["normalized_value"] == "2 microliters"
    sec12 = article.find(".//*[@id='Sec12']")
    assert re.search(r"2\s*µl", "".join(sec12.itertext()), re.I)
    report.update(
        source_cells_verified=13561,
        source_summary_arithmetic_checks=computed,
        normalized_only_values=normalized_only,
        nonuniform_ratio_columns=nonuniform,
        distinct_cross_asset_reporters=32,
    )

    query = json.loads((ROOT / "results/query-all-added-sources.json").read_text())
    n_query_rows = 0
    for name, record in query.items():
        table = pq.ParquetFile(source / record["artifact"])
        assert record["fields"] == ["native." + f for f in table.schema_arrow.names]
        hasher, count = hashlib.sha256(), 0
        for batch in table.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                hasher.update(
                    (
                        json.dumps(
                            row, sort_keys=True, separators=(",", ":"), allow_nan=False
                        )
                        + "\n"
                    ).encode()
                )
                count += 1
        assert count == record["n_rows"] == table.metadata.num_rows
        assert hasher.hexdigest() == record["canonical_rows_sha256"], name
        n_query_rows += count
    assert len(query) == 31
    catalog = json.loads((ROOT / "results/catalog.json").read_text())
    present = [r for r in catalog["evidence"] if r["status"] == "materialized"]
    assert len(present) == 84
    report.update(
        queried_relations=len(query),
        full_query_rows=n_query_rows,
        materialized_evidence_types=len(present),
    )
    conserved = {}
    for recipe_id, before_path in CONFIG["old_recipes"].items():
        old, new = Path(before_path), Path(CONFIG["recipe_outputs"]) / recipe_id
        old_manifest = json.loads((old / "manifest.json").read_text())
        new_manifest = json.loads((new / "manifest.json").read_text())
        assert (
            new_manifest["source_package_manifest_sha256"]
            == digest(source / "manifest.json")["sha256"]
        )
        files = []
        for name in old_manifest["files"]:
            if name == "source_manifest.json":
                continue
            assert (old / name).read_bytes() == (new / name).read_bytes(), (
                recipe_id,
                name,
            )
            files.append(name)
        for name, expected in new_manifest["files"].items():
            assert digest(new / name) == expected
        conserved[recipe_id] = files
    report["conserved_recipe_files"] = conserved
    report["passed"] = True


if __name__ == "__main__":
    start = time.perf_counter()
    report = {"passed": False, "verifier_sha256": digest(__file__)["sha256"]}
    try:
        main(report)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - start
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (ROOT / "results/independent-verification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
