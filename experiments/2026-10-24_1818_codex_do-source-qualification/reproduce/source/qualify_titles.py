"""Recover numeric title observations without assuming peptide-level quantities."""

import gzip
import hashlib
import json
import math
import re
import resource
import shutil
import time
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/do_source_qualification/2026-10-24_1818_v1"


def main():
    start = time.perf_counter()
    out = ROOT / "results/title-intensity-observations.json.gz"
    assert not out.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/qualify_titles.py")
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
    for record in receipts:
        if record["status"] == "retrieved":
            data = Path(record["path"]).read_bytes()
            assert len(data) == record["bytes"]
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
    path = RAW / "012916_WT1_1.mzid"
    tree = ET.parse(path)
    ns = {"m": tree.getroot().tag.partition("}")[0][1:]}
    records = []
    for node in tree.findall(".//m:SpectrumIdentificationResult", ns):
        titles = [
            p.get("value")
            for p in node.findall("m:cvParam", ns)
            if p.get("accession") == "MS:1000796"
        ]
        assert len(titles) == 1 and titles[0] == node.get("name")
        match = re.search(r"\(intensity=([^()]*)\)$", titles[0])
        raw_value = match.group(1) if match else None
        value = float(raw_value) if raw_value is not None else None
        state = (
            "missing"
            if value is None
            else "nonfinite"
            if not math.isfinite(value)
            else "positive"
            if value > 0
            else "zero"
            if value == 0
            else "negative"
        )
        identifications = [
            {
                key: item.get(key)
                for key in ["id", "peptide_ref", "rank", "passThreshold", "chargeState"]
            }
            for item in node.findall("m:SpectrumIdentificationItem", ns)
        ]
        records.append(
            {
                "result_id": node.get("id"),
                "spectrum_id": node.get("spectrumID"),
                "spectra_data_ref": node.get("spectraData_ref"),
                "title": titles[0],
                "literal_intensity": raw_value,
                "reported_intensity": value,
                "state": state,
                "identifications": identifications,
                "measurement_scope": "one source spectrum result; precursor aggregation/units unresolved",
            }
        )
    second = etree.parse(str(path))
    nodes = second.xpath('//*[local-name()="SpectrumIdentificationResult"]')
    assert len(nodes) == len(records)
    for node, row in zip(nodes, records, strict=True):
        name = node.attrib["name"]
        title = node.xpath('./*[local-name()="cvParam"][@accession="MS:1000796"]')[
            0
        ].attrib["value"]
        assert name == title == row["title"]
        prefix, separator, suffix = name.rpartition("(intensity=")
        expected = (
            float(suffix.removesuffix(")"))
            if separator and suffix.endswith(")")
            else None
        )
        assert expected == row["reported_intensity"]
        assert node.attrib["id"] == row["result_id"]
        for field, key in [
            ("spectrumID", "spectrum_id"),
            ("spectraData_ref", "spectra_data_ref"),
        ]:
            assert node.attrib.get(field) == row[key]
        items = node.xpath('./*[local-name()="SpectrumIdentificationItem"]')
        assert len(items) == len(row["identifications"])
        for item, reported in zip(items, row["identifications"], strict=True):
            assert {key: item.attrib.get(key) for key in reported} == reported
    out.write_bytes(
        gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
    )
    initial = json.loads(
        (ROOT / "results/attempt-field-names-only/source-audit.json").read_text()
    )
    values = Counter(
        row["reported_intensity"] for row in records if row["state"] == "positive"
    )
    summary = {
        **initial,
        "status": "numeric_title_observations_recovered_semantics_pending",
        "supersedes": "results/attempt-field-names-only/source-audit.json",
        "field_name_check_limitation": "No named quantitative CV fields; numeric observations are embedded in values.",
        "spectrum_title_observations_verified": len(records),
        "state_counts": dict(Counter(row["state"] for row in records)),
        "identification_items_preserved": sum(
            len(row["identifications"]) for row in records
        ),
        "positive_unique_values": len(values),
        "positive_values_repeated_across_results": sum(n > 1 for n in values.values()),
        "results_with_nonunique_positive_value": sum(
            n for n in values.values() if n > 1
        ),
        "reported_numeric_range": [min(values), max(values)],
        "source_equivalent_peptide_quantities_admitted": False,
        "quantity_absence_scope": "Superseded: numeric source observations exist inside spectrum titles.",
        "aggregation_or_unit_inference": False,
        "seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    (ROOT / "results/source-audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
