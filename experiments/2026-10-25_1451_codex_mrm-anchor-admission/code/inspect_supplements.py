"""Inspect registered Office supplements and independently replay stored cells."""

import argparse
import hashlib
import io
import json
import math
import posixpath
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def safe_archive(z):
    members = z.infolist()
    assert sum(m.file_size for m in members) <= 512 * 1024**2
    assert len({m.filename for m in members}) == len(members)
    for member in members:
        path = PurePosixPath(member.filename)
        assert not path.is_absolute() and ".." not in path.parts
        assert "\\" not in member.filename and member.file_size <= 256 * 1024**2
        assert (member.external_attr >> 16) & 0o170000 != 0o120000
    assert z.testzip() is None
    return [{"name": m.filename, "bytes": m.file_size} for m in members]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True, type=Path)
    args = parser.parse_args()
    contract = json.loads(args.requests.read_text())
    cfg = json.loads((ROOT / "reproduce/launch-v3.json").read_text())
    data = Path(cfg["data_directory"])
    article = json.loads((data / "s3-PMC6123701.1.json").read_text())
    assert article["doi"] == "10.1038/s42003-018-0087-6"
    assert article["is_pmc_openaccess"] and not article["is_manuscript"]
    assert not article["is_retracted"] and article["license_code"] == "CC BY"
    md5s = {
        Path(urlparse(u).path).name: parse_qs(urlparse(u).query)["md5"][0]
        for u in article["media_urls"]
    }
    requests = json.loads((ROOT / contract["source_requests"]).read_text())["requests"]
    result = {
        "article": article,
        "assets": [],
        "cell_checks": 0,
        "unverified_cells": [],
    }
    for request in requests:
        path = data / request["name"]
        raw = path.read_bytes()
        receipt = json.loads(path.with_name(path.name + ".receipt.json").read_text())
        assert receipt["status"] == "retrieved_validated_payload"
        assert len(raw) == request["expected_bytes"] == receipt["bytes"]
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        assert hashlib.md5(raw).hexdigest() == request["expected_inventory_etag"]
        assert hashlib.md5(raw).hexdigest() == md5s[path.name.removeprefix("s3-")]
        asset = {"file": path.name, "bytes": len(raw), "sha256": receipt["sha256"]}
        with ZipFile(io.BytesIO(raw)) as z:
            asset["archive_members"] = safe_archive(z)
            if path.suffix == ".docx":
                doc = ET.fromstring(z.read("word/document.xml"))
                wn = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                asset["paragraphs"] = [
                    "".join(t.text or "" for t in p.iter(wn + "t"))
                    for p in doc.iter(wn + "p")
                ]
            else:
                book = openpyxl.load_workbook(
                    io.BytesIO(raw), data_only=True, read_only=True
                )
                shared = []
                if "xl/sharedStrings.xml" in z.namelist():
                    shared = [
                        "".join(t.text or "" for t in item.findall(".//s:t", NS))
                        for item in ET.fromstring(z.read("xl/sharedStrings.xml"))
                    ]
                rels = {
                    e.attrib["Id"]: e.attrib["Target"]
                    for e in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
                }
                asset["sheets"] = []
                workbook = ET.fromstring(z.read("xl/workbook.xml"))
                for descriptor in workbook.findall("s:sheets/s:sheet", NS):
                    name = descriptor.attrib["name"]
                    target = rels[descriptor.attrib[REL]]
                    target = (
                        target.lstrip("/")
                        if target.startswith("/")
                        else posixpath.normpath(posixpath.join("xl", target))
                    )
                    sheet_xml = ET.fromstring(z.read(target))
                    cells = {
                        c.coordinate: c.value
                        for row in book[name].iter_rows()
                        for c in row
                        if c.value is not None
                    }
                    sheet = {
                        "name": name,
                        "dimensions": book[name].calculate_dimension(),
                        "nonempty_cached_cells": len(cells),
                        "header_rows": [],
                        "semantic_text_cells": [],
                        "formula_cells": 0,
                        "types": Counter(),
                    }
                    for c in sheet_xml.findall(".//s:sheetData/s:row/s:c", NS):
                        coordinate = c.attrib["r"]
                        kind = c.attrib.get("t", "n")
                        sheet["types"][kind] += 1
                        v = c.findtext("s:v", namespaces=NS)
                        if c.find("s:f", NS) is not None:
                            sheet["formula_cells"] += 1
                        if kind == "s":
                            value = shared[int(v)]
                        elif kind == "inlineStr":
                            value = "".join(
                                t.text or "" for t in c.findall(".//s:t", NS)
                            )
                        elif kind == "b":
                            value = v == "1"
                        elif kind in ("str", "e"):
                            value = v
                        elif kind == "n":
                            value = float(v) if v not in (None, "") else None
                        else:
                            result["unverified_cells"].append(
                                [path.name, name, coordinate, kind]
                            )
                            continue
                        actual = cells.get(coordinate)
                        if isinstance(actual, (date, datetime)):
                            result["unverified_cells"].append(
                                [path.name, name, coordinate, "date_style_conversion"]
                            )
                        else:
                            same = (
                                math.isclose(value, actual, rel_tol=1e-12, abs_tol=0)
                                if isinstance(value, float)
                                and isinstance(actual, (int, float))
                                else value == actual or value == "" and actual is None
                            )
                            assert same, (path.name, name, coordinate, value, actual)
                            result["cell_checks"] += 1
                        if (
                            int(re.sub("[A-Z]", "", coordinate)) <= 6
                            and value is not None
                        ):
                            sheet["header_rows"].append([coordinate, value])
                        if isinstance(value, str) and re.search(
                            r"area|intensit|ratio|response|amount|concentration|purity|fmol",
                            value,
                            re.IGNORECASE,
                        ):
                            sheet["semantic_text_cells"].append([coordinate, value])
                    asset["sheets"].append(sheet)
                book.close()
        result["assets"].append(asset)
    result["status"] = (
        "verified" if not result["unverified_cells"] else "partial_verification"
    )
    with (ROOT / "results/supplement-inspection.json").open("x") as stream:
        json.dump(result, stream, indent=2, default=str)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "assets": len(result["assets"]),
                "cell_checks": result["cell_checks"],
                "unverified_cells": len(result["unverified_cells"]),
            }
        )
    )


if __name__ == "__main__":
    main()
