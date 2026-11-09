"""Check original article identity and source contract without training labels."""

import argparse
import hashlib
import json
import xml.dom.minidom as dom
import xml.etree.ElementTree as ET
from pathlib import Path


def node_text(node):
    if node.nodeType in (node.TEXT_NODE, node.CDATA_SECTION_NODE):
        return node.data
    return "".join(node_text(child) for child in node.childNodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.requests.read_text())
    root = Path(__file__).resolve().parents[1]
    article = Path(request["article_path"]).read_bytes()
    metadata = json.loads(Path(request["metadata_path"]).read_text())
    expected_md5 = metadata["xml_url"].split("?md5=")[1]
    assert hashlib.md5(article).hexdigest() == expected_md5
    assert len(article) == 111323
    assert metadata["doi"] == "10.1038/nbt.1524"
    assert metadata["license_code"] == "TDM"
    assert metadata["is_pmc_openaccess"] is False
    et = ET.fromstring(article)
    other = dom.parseString(article)
    a = ["".join(p.itertext()) for p in et.findall(".//p")]
    b = [node_text(p) for p in other.getElementsByTagName("p")]
    assert a == b
    records = []
    for title in ("Method overview", "MRM-MS assay development"):
        sections = [s for s in et.findall(".//sec") if s.findtext("title") == title]
        assert len(sections) == 1
        for i, paragraph in enumerate(sections[0].findall("p")):
            text = "".join(paragraph.itertext())
            records.append(
                {
                    "section_title": title,
                    "paragraph_index_zero_based": i,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    overview = " ".join(a)
    assert "yeast lysate sample" in overview
    assert "proteins were individually digested with trypsin" in overview
    assert "Stable isotope" in overview
    assert "versions of each candidate peptide were synthesized" in overview
    supplements = [
        {"id": s.attrib["id"], "media": [m.attrib for m in s.iter("media")]}
        for s in et.findall(".//supplementary-material")
    ]
    assert len(supplements) == 5
    report = {
        "status": "article_verified_calibration_not_admitted",
        "doi": metadata["doi"],
        "pmcid": metadata["pmcid"],
        "article_bytes": len(article),
        "article_sha256": hashlib.sha256(article).hexdigest(),
        "official_md5_verified": True,
        "independent_parser_paragraphs": len(a),
        "paragraphs_agree": True,
        "contract_source_coordinates": records,
        "supplement_references": supplements,
        "numeric_supplements_retrieved": 0,
        "amount_signal_records_admitted": 0,
        "decision": "Digest-response training data do not isolate MS response; selected MRM validation is mentioned, but its raw amount/signal records remain unqualified.",
        "no_global_absence_claim": True,
        "raw_original_redistribution": False,
        "new_model_or_outer_access": False,
    }
    with (root / "results/source-admission.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("contract_source_coordinates", "supplement_references")
            }
        )
    )


if __name__ == "__main__":
    main()
