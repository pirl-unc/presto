"""Independent literal-cell inventory; no calibration fits or biological scores."""

import argparse
import hashlib
import json
import posixpath
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import openpyxl


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(node):
    return "".join(node.itertext()) if node is not None else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    request = json.loads(parser.parse_args().requests.read_text())
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "reproduce/launch.json").read_text())
    data = Path(config["data_directory"])
    manifest = json.loads((data / "PMC8449407.1.json").read_text())
    assert (
        manifest["doi"] == "10.1073/pnas.2111173118"
        and manifest["pmcid"] == "PMC8449407"
    )
    assert (
        hashlib.md5((data / "PMC8449407.1.xml").read_bytes()).hexdigest()
        == manifest["xml_url"].split("md5=")[1]
    )
    xml = ET.parse(data / "PMC8449407.1.xml").getroot()
    assert [
        n.text
        for n in xml.findall(".//article-id")
        if n.attrib.get("pub-id-type") == "doi"
    ] == [manifest["doi"]]
    summaries = []
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    for filename in request["files"]:
        path = data / filename
        url = [u for u in manifest["media_urls"] if "/" + filename + "?" in u]
        assert len(url) == 1
        assert hashlib.md5(path.read_bytes()).hexdigest() == url[0].split("md5=")[1]
        receipt = json.loads((data / (filename + ".receipt.json")).read_text())
        assert (
            receipt["status"] == "retrieved_validated_payload"
            and sha(path) == receipt["sha256"]
        )
        if not filename.endswith(".xlsx"):
            continue
        book = openpyxl.load_workbook(path, data_only=True, read_only=False)
        formula_book = openpyxl.load_workbook(path, data_only=False, read_only=False)
        with ZipFile(path) as archive:
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                strings = [
                    "".join(n.itertext())
                    for n in ET.fromstring(
                        archive.read("xl/sharedStrings.xml")
                    ).findall("s:si", ns)
                ]
            relations = {
                n.attrib["Id"]: n.attrib["Target"]
                for n in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            }
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            for sheet in workbook.findall("s:sheets/s:sheet", ns):
                name = sheet.attrib["name"]
                relation = sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                target = relations[relation]
                member = (
                    posixpath.normpath(posixpath.join("xl", target))
                    if not target.startswith("/")
                    else target.lstrip("/")
                )
                worksheet = ET.fromstring(archive.read(member))
                records = []
                formulas = []
                types = Counter()
                for cell in worksheet.findall(".//s:sheetData/s:row/s:c", ns):
                    coord = cell.attrib["r"]
                    kind = cell.attrib.get("t", "n")
                    value = cell.findtext("s:v", namespaces=ns)
                    if kind == "s":
                        decoded = strings[int(value)] if value is not None else None
                    elif kind == "inlineStr":
                        decoded = "".join(
                            n.text or "" for n in cell.findall(".//s:t", ns)
                        )
                    elif kind == "b":
                        decoded = value == "1" if value is not None else None
                    elif kind in ("str", "e"):
                        decoded = value
                    elif value is None:
                        decoded = None
                    else:
                        decoded = float(value)
                    actual = book[name][coord].value
                    if isinstance(actual, datetime):
                        assert kind == "n" and decoded is not None
                        offset = (
                            decoded
                            if book.epoch.year == 1904
                            else decoded - (1 if decoded >= 60 else 0)
                        )
                        start = (
                            book.epoch
                            if book.epoch.year == 1904
                            else datetime(1899, 12, 31)  # noqa: DTZ001 -- Excel serials have no timezone
                        )
                        assert actual == start + timedelta(days=offset)
                    else:
                        assert actual == decoded, (
                            filename,
                            name,
                            coord,
                            actual,
                            decoded,
                            kind,
                        )
                    formula = cell.find("s:f", ns)
                    if formula is not None:
                        formulas.append(
                            {
                                "cell": coord,
                                "expression": formula.text,
                                "cached_value": value,
                            }
                        )
                        assert formula_book[name][coord].data_type == "f"
                    types[kind] += 1
                    records.append(
                        {
                            "cell": coord,
                            "type": kind,
                            "style": cell.attrib.get("s"),
                            "raw_value": value,
                            "value": str(actual)
                            if isinstance(actual, datetime)
                            else actual,
                        }
                    )
                preview = []
                for row in book[name].iter_rows():
                    values = [
                        {
                            "cell": c.coordinate,
                            "value": str(c.value)
                            if isinstance(c.value, datetime)
                            else c.value,
                        }
                        for c in row
                        if c.value is not None
                    ]
                    if values:
                        preview.append(values)
                    if len(preview) >= 14:
                        break
                summaries.append(
                    {
                        "file": filename,
                        "sheet": name,
                        "dimensions": book[name].calculate_dimension(),
                        "n_stored_cells_replayed": len(records),
                        "cell_types": dict(types),
                        "formulas": formulas,
                        "literal_cells_sha256": hashlib.sha256(
                            json.dumps(
                                records, sort_keys=True, ensure_ascii=False
                            ).encode()
                        ).hexdigest(),
                        "first_nonempty_rows": preview,
                    }
                )
        book.close()
        formula_book.close()
    pages = []
    extraction = None
    result = {
        "status": "literal_inventory_verified_semantic_admission_pending",
        "doi": manifest["doi"],
        "accession": "PXD024917",
        "n_workbooks": 1,
        "n_sheets": len(summaries),
        "n_source_cells_replayed": sum(s["n_stored_cells_replayed"] for s in summaries),
        "workbooks": summaries,
        "pdf_pages": pages,
        "pdf_extractions_and_renders": str(extraction),
        "all_official_md5s_verified": True,
        "model_fitting_or_scoring": False,
    }
    with (root / "results/isotope-inventory.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    semantic(root, data)
    print(json.dumps({k: v for k, v in result.items() if k != "workbooks"}))


def semantic(root, data):
    import math
    import re

    panel_book = openpyxl.load_workbook(
        data / "pnas.2111173118.sd01.xlsx", data_only=True
    )
    panel = {panel_book.active.cell(row, 2).value for row in range(4, 22)}
    assert len(panel) == 18
    source = openpyxl.load_workbook(data / "pnas.2111173118.sd03.xlsx", data_only=True)
    filemap = openpyxl.load_workbook(data / "pnas.2111173118.sd07.xlsx", data_only=True)
    file_rows = [
        list(row)
        for row in filemap.active.iter_rows(min_row=2, max_col=3, values_only=True)
        if row[0] is not None
    ]
    records = []
    groups = []
    forms = {}
    for sheet_name, expected_doses, spike_stage in (
        (
            "DatasetS3A",
            [0.1, 1.0, 10.0, 100.0],
            "free_peptide_after_HLA_peptide_purification",
        ),
        (
            "DatasetS3B",
            [0.1, 1.0, 10.0, 100.0, 1000.0],
            "hipMHC_before_immunoprecipitation",
        ),
    ):
        sheet = source[sheet_name]
        assert str(sheet.merged_cells) == "C3:E3"
        current = None
        current_rows = []
        for row in range(4, sheet.max_row + 1):
            label = sheet.cell(row, 1).value
            dose = sheet.cell(row, 2).value
            if dose is None:
                assert all(sheet.cell(row, col).value is None for col in range(3, 6))
                continue
            if label is not None:
                if current is not None:
                    assert [r[1] for r in current_rows] == expected_doses
                    groups.append((sheet_name, current, current_rows))
                current = str(label)
                current_rows = []
            assert current is not None and isinstance(dose, (float, int))
            sequence = re.sub(r"\[\+[-0-9.]+\]", "", current)
            assert sequence in panel, (sheet_name, current, sequence)
            current_rows.append((row, float(dose)))
            forms[(sheet_name, current)] = sequence
            for col in range(3, 6):
                cell = sheet.cell(row, col)
                value = cell.value
                assert (
                    value is None
                    or isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value >= 0
                )
                records.append(
                    {
                        "sheet": sheet_name,
                        "source_cell": cell.coordinate,
                        "source_peptide_block": current,
                        "sequence": sequence,
                        "dose_cell": sheet.cell(row, 2).coordinate,
                        "added_fmol": float(dose),
                        "raw_signal": value,
                        "signal_status": "not_reported"
                        if value is None
                        else "reported_zero"
                        if value == 0
                        else "reported_positive",
                        "replicate_column_ordinal": col - 2,
                        "acquisition_id": None,
                        "acquisition_link_status": "unlabeled_source_column_no_exact_raw_file_join",
                        "spike_stage": spike_stage,
                        "isotopologue": "1H" if sheet_name.endswith("A") else "2H",
                    }
                )
        assert [r[1] for r in current_rows] == expected_doses
        groups.append((sheet_name, current, current_rows))
    summary = {}
    for sheet_name in ("DatasetS3A", "DatasetS3B"):
        chosen = [r for r in records if r["sheet"] == sheet_name]
        sequences = {r["sequence"] for r in chosen}
        n_supported = sum(
            len(
                {
                    r["added_fmol"]
                    for r in chosen
                    if r["sequence"] == seq
                    and r["signal_status"] == "reported_positive"
                }
            )
            >= 3
            for seq in sequences
        )
        summary[sheet_name] = {
            "n_sequences": len(sequences),
            "n_slots": len(chosen),
            "n_positive": sum(
                r["signal_status"] == "reported_positive" for r in chosen
            ),
            "n_zero": sum(r["signal_status"] == "reported_zero" for r in chosen),
            "n_missing": sum(r["signal_status"] == "not_reported" for r in chosen),
            "n_sequences_with_3_positive_doses": n_supported,
        }
    assert summary["DatasetS3A"]["n_sequences_with_3_positive_doses"] >= 12
    source_exports = [r for r in file_rows if str(r[1]).lower() == "search"]
    result = {
        "status": "known_amount_and_signal_support_qualified_exact_acquisition_joins_pending",
        "cohort_support": summary,
        "n_panel_sequences": 18,
        "filemap_source_exports": source_exports,
        "scope": "selected HLA-A0201 tumor-antigen targeted MS2 response; not MS1 or intracellular free-peptide concentration",
        "metadata_gaps": [
            "unlabeled source response columns need exact acquisition mapping",
            "transition identities/aggregation/interference need deposited transition exports",
            "18 selected targets do not establish broad sequence or protocol transfer",
        ],
        "training_admitted": False,
        "shared_hitlist_ingestion_required": True,
        "amount_assignment": {
            "free_peptide": "reported amino-acid analysis; HPLC purity >97%; isotopologues balanced by LC/MS",
            "hipMHC": "reported stable-complex ELISA assignment after exchange",
            "source": "original supplementary Methods page2; not independently certified values",
        },
        "forward_fill_scope": "peptide label applies to its explicitly delimited table block with complete reported dose grid; original blank cells unchanged",
        "measurement_column_semantics": "ordinal worksheet slot only, not asserted technical injection identity",
    }
    with (root / "results/dose-signal-cells.json").open("x") as f:
        json.dump(records, f, indent=2, allow_nan=False)
        f.write("\n")
    with (root / "results/semantic-qualification.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
