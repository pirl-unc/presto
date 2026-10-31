"""Reproduce the bounded header audit; do not infer total data absence."""

import csv
import hashlib
import json
import zipfile
from pathlib import Path

from lxml import etree

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
BASE = (
    ROOT
    / "data/hap1_quantitative_source_admission/2026-10-24_0612_codex_hap1-quantitative-admission/canonical-package-v2/hap1/source_assets"
)


def main():
    assets = {}
    for name in ["article.xml", "supplements.zip", "Search.zip"]:
        p = BASE / name
        assets[name] = {
            "path": str(p.relative_to(ROOT)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    headers = []
    for name in ["supplements.zip", "Search.zip"]:
        with zipfile.ZipFile(BASE / name) as archive:
            for member in archive.namelist():
                if not member.lower().endswith((".csv", ".tsv", ".txt")):
                    continue
                with archive.open(member) as stream:
                    line = stream.readline().decode("utf-8-sig", errors="replace").rstrip("\r\n")
                fields = next(csv.reader([line], delimiter="\t" if "\t" in line else ","))
                hits = [
                    f
                    for f in fields
                    if any(
                        q in f.lower() for q in ["rank", "hla", "mhc", "affinity", "bind", "allele"]
                    )
                ]
                headers.append(
                    {"archive": name, "member": member, "fields": fields, "keyword_matches": hits}
                )
    tree = etree.parse(str(BASE / "article.xml"))
    assert any("MixMHCpred (version 2.2)" in "".join(p.itertext()) for p in tree.xpath("//p"))
    result = {
        "assets": assets,
        "headers": headers,
        "method": "First line only from listed text exports; does not establish absence from embedded values, other files or nonpublic author outputs.",
        "article_predictor": "MixMHCpred 2.2",
        "role": "predicted selection covariates, not measured restriction or affinity",
        "used_for_fitting": False,
    }
    path = EXP / "results/binding-covariate-source-audit.json"
    assert json.loads(path.read_text()) == result
    print(
        json.dumps(
            {
                "status": "passed",
                "n_header_records": len(headers),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
