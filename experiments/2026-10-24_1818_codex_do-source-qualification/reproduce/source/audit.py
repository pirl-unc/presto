"""Inventory retrieved source schemas and independently verify source artifacts."""

import csv
import hashlib
import json
import re
import resource
import shutil
import time
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/do_source_qualification/2026-10-24_1818_v1"


def main():
    start = time.perf_counter()
    destination = ROOT / "results/source-audit.json"
    assert not destination.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/audit.py")
    receipts = []
    for name in [
        "acquisition",
        "metadata-stage2",
        "metadata-stage3",
        "supplement-access",
        "processed-probe",
    ]:
        receipts.extend(
            json.loads((ROOT / f"results/{name}.json").read_text())["assets"]
        )
    successful = [r for r in receipts if r["status"] == "retrieved"]
    for record in successful:
        path = Path(record["path"])
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        if "expected_bytes" in record:
            assert record["bytes"] == record["expected_bytes"]
    html = (RAW / "massive-files.html").read_text()
    offset = re.search(r"var dataset_files\s*=\s*", html).end()
    inventory, _ = json.JSONDecoder().raw_decode(html[offset:])
    saved = json.loads((ROOT / "results/massive-file-inventory.json").read_text())
    assert inventory == saved
    # Independent extraction from the script node, without the producer regex.
    script = next(
        s.get_text()
        for s in BeautifulSoup(html, "html.parser").find_all("script")
        if "var dataset_files =" in s.get_text()
    )
    source_json = script.split("var dataset_files =", 1)[1].lstrip().split(";", 1)[0]
    assert json.loads(source_json) == inventory
    rows = inventory["row_data"]
    assert (
        len(rows)
        == inventory["total_rows"]
        == len({r["file_descriptor"] for r in rows})
    )
    summary = json.loads((ROOT / "results/massive-inventory-summary.json").read_text())
    assert summary["n_files"] == len(rows)
    assert summary["collections"] == dict(Counter(r["collection"] for r in rows))
    assert summary["extensions"] == dict(Counter(Path(r["name"]).suffix for r in rows))
    assert summary["all_declared_bytes"] == sum(r["size"] for r in rows)
    with (RAW / "012916_WT1_1.mzTab").open() as handle:
        table_rows = list(csv.reader(handle, delimiter="\t"))
    counts = Counter(row[0] if row else "" for row in table_rows)
    headers = {row[0]: row[1:] for row in table_rows if row and row[0].endswith("H")}
    lines = (RAW / "012916_WT1_1.mzTab").read_text().splitlines()
    assert counts == Counter(line.split("\t")[0] for line in lines)
    assert headers == {
        line.split("\t")[0]: line.split("\t")[1:]
        for line in lines
        if line.split("\t")[0].endswith("H")
    }
    quant = re.compile(
        r"intensity|abundance|peak.?area|quantity|concentration", re.IGNORECASE
    )
    quantitative_headers = {
        section: [field for field in fields if quant.search(field)]
        for section, fields in headers.items()
    }
    tree = ET.parse(RAW / "012916_WT1_1.mzid")
    cv = Counter(
        (node.get("accession"), node.get("name"))
        for node in tree.iter()
        if node.tag.endswith("}cvParam")
    )
    user = Counter(
        node.get("name") for node in tree.iter() if node.tag.endswith("}userParam")
    )
    counts_xml = Counter(node.tag.rsplit("}", 1)[-1] for node in tree.iter())
    second = etree.parse(str(RAW / "012916_WT1_1.mzid"))
    assert cv == Counter(
        (node.get("accession"), node.get("name"))
        for node in second.xpath('//*[local-name()="cvParam"]')
    )
    assert user == Counter(
        node.get("name") for node in second.xpath('//*[local-name()="userParam"]')
    )
    assert counts_xml == Counter(etree.QName(node).localname for node in second.iter())
    quantity_terms = [name for _, name in cv if quant.search(name or "")]
    quantity_terms.extend(name for name in user if quant.search(name or ""))
    listing = ET.parse(RAW / "pmc-cloud-list.xml")
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    assert listing.findtext("s:IsTruncated", namespaces=ns) == "false"
    assert listing.findtext("s:KeyCount", namespaces=ns) == "0"
    assert not listing.findall("s:Contents", ns)
    epmc = ET.parse(RAW / "supplements.zip")
    error_text = "".join(epmc.getroot().itertext())
    assert "not open access" in error_text
    index_title = BeautifulSoup(
        (RAW / "supplement-index.html").read_text(), "html.parser"
    ).title.get_text()
    assert "reCAPTCHA" in index_title
    schemas = {
        "mzTab_sections": dict(counts),
        "mzTab_headers": headers,
        "mzTab_quantitative_headers": quantitative_headers,
        "mzIdentML_elements": dict(counts_xml),
        "mzIdentML_cv_terms": [
            {"accession": a, "name": n, "count": c} for (a, n), c in cv.items()
        ],
        "mzIdentML_user_terms": dict(user),
        "mzIdentML_quantitative_terms": quantity_terms,
    }
    (ROOT / "results/processed-probe-schemas.json").write_text(
        json.dumps(schemas, indent=2) + "\n"
    )
    result = {
        "status": "source_access_gap_no_quantity_cohort_admitted",
        "successful_asset_hashes_verified": len(successful),
        "downloaded_bytes": sum(r["bytes"] for r in successful),
        "massive_inventory_files_verified": len(rows),
        "inspected_processed_files": ["012916_WT1_1.mzTab", "012916_WT1_1.mzid"],
        "mzTab_psm_rows": counts["PSM"],
        "mzTab_protein_rows": counts["PRT"],
        "mzIdentML_spectrum_identification_items": counts_xml[
            "SpectrumIdentificationItem"
        ],
        "mzIdentML_spectrum_identification_results": counts_xml[
            "SpectrumIdentificationResult"
        ],
        "quantitative_headers_or_terms_found": sum(
            map(len, quantitative_headers.values())
        )
        + len(quantity_terms),
        "quantity_absence_scope": "These two processed files only; remaining repository result files were not decoded.",
        "supplement_api_payload": error_text,
        "supplement_index_payload": index_title,
        "pmc_cloud_article_objects": 0,
        "all_source_quantities_absent_claimed": False,
        "model_training_or_predictions": False,
        "seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
