"""Read primary declarations and static software artifacts; execute no downloaded code."""

import argparse
import ast
import csv
import hashlib
import io
import json
import xml.dom.minidom as dom
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.requests.read_text())
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "reproduce/launch.json").read_text())
    assets = Path(config["data_directory"])
    receipts = []
    for name in (
        "ClipsMS_v2.0.1.py",
        "No_Unlocalized_Modifications.csv",
        "README.md",
        "LICENSE",
    ):
        raw = (assets / name).read_bytes()
        receipt = json.loads((assets / (name + ".receipt.json")).read_text())
        assert receipt["status"] == "retrieved_validated_payload"
        assert len(raw) == receipt["bytes"]
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        receipts.append({"name": name, "sha256": receipt["sha256"], "bytes": len(raw)})
    raw = Path(request["article_xml"]).read_bytes()
    tree = ET.fromstring(raw)
    phrase = "No additional modifications were included"
    paragraphs = ["".join(p.itertext()) for p in tree.iter("p")]
    hits = [i for i, p in enumerate(paragraphs) if phrase in p]
    assert len(hits) == 1
    paragraph = paragraphs[hits[0]]
    section = tree.find(".//sec[@id='s15']")
    assert (
        "".join(section.find("title").itertext())
        == "ID-MS Data Analysis and Statistics."
    )
    assert paragraph in ["".join(p.itertext()) for p in section.iter("p")]
    assert all(
        term in paragraph for term in ("10 ppm", "P37840", "4 amino acids", "ClipsMS")
    )
    assert any(n.text == "10.1073/pnas.2419607122" for n in tree.iter("article-id"))

    def dom_text(node):
        return "".join(
            c.data if c.nodeType in (c.TEXT_NODE, c.CDATA_SECTION_NODE) else dom_text(c)
            for c in node.childNodes
        )

    second = [dom_text(p) for p in dom.parseString(raw).getElementsByTagName("p")]
    assert paragraphs == second
    csv_bytes = (assets / "No_Unlocalized_Modifications.csv").read_bytes()
    parsed = list(csv.reader(io.StringIO(csv_bytes.decode())))
    assert parsed == [csv_bytes.decode().strip().split(",")]
    assert len(parsed) == 1 and len(parsed[0]) == 12
    assert parsed[0][0] == "None" and float(parsed[0][1]) == 0
    code_bytes = (assets / "ClipsMS_v2.0.1.py").read_bytes()
    commit = json.loads((root / "results/clipsms-source-commit.json").read_text())
    blob = next(
        f["sha"] for f in commit["files"] if f["filename"] == "ClipsMS_v2.0.1.py"
    )
    assert (
        hashlib.sha1(
            b"blob " + str(len(code_bytes)).encode() + b"\0" + code_bytes
        ).hexdigest()
        == blob
    )
    syntax = ast.parse(code_bytes.decode())
    functions = [
        n
        for n in ast.walk(syntax)
        if isinstance(n, ast.FunctionDef) and n.name == "No_Variable_Modifications"
    ]
    assert len(functions) == 1
    literals = {
        n.value
        for n in ast.walk(functions[0])
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert {
        "No_Unlocalized_Modifications.csv",
        "Unlocalized_Modifications.csv",
    } <= literals
    result = {
        "status": "qualified_method_supported_unmodified_search_assignment",
        "study_doi": "10.1073/pnas.2419607122",
        "primary_source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11874349/",
        "source_locator": "article.xml#s15 (ID-MS Data Analysis and Statistics), XML paragraph index43",
        "declaration_excerpt": phrase,
        "article_sha256": hashlib.sha256(raw).hexdigest(),
        "paragraph_sha256": hashlib.sha256(paragraph.encode()).hexdigest(),
        "xml_parsers_agree": True,
        "software_commit": commit["sha"],
        "software_commit_date": commit["commit"]["committer"]["date"],
        "software_no_modification_file": {
            "name": parsed[0][0],
            "mass_shift": float(parsed[0][1]),
        },
        "no_variable_modifications_function_lines": [
            functions[0].lineno,
            functions[0].end_lineno,
        ],
        "software_version_used_by_study": None,
        "blank_export_roundtrip_proven": False,
        "interpretation_basis": "explicit study method declaration, supported by software no-modification convention; not inference from blank alone",
        "required_row_conditions": "reported zero localized and terminal shifts; unlocalized explicitly None or absent; all other original source-quality gates remain",
        "raw_nulls_rewritten": False,
        "biochemical_purity_proven": False,
        "downloaded_code_executed": False,
        "numerical_F3_outcomes_queried": False,
        "model_scoring_or_fitting": False,
        "retrieved_assets": receipts,
        "next_action": "shared provenance-bearing search policy and separately registered frozen-model comparison",
    }
    with (root / "results/qualification.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
