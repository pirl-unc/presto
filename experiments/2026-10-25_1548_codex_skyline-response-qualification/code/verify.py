"""Verify the retrieved public inventory; do not infer Skyline file contents."""

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from bs4 import BeautifulSoup


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_pre = False
        self.text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.in_pre = True
        if tag == "a":
            self.links.append(dict(attrs).get("href", ""))

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_pre = False

    def handle_data(self, data):
        if self.in_pre:
            self.text.append(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.requests.read_text())
    root = Path(__file__).resolve().parents[1]
    raw = Path(spec["inventory_path"]).read_bytes()
    receipt = json.loads(
        (root / "results/pass05890-upload-index.html.receipt.json").read_text()
    )
    assert len(raw) == receipt["bytes"]
    assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
    assert receipt["status"] == "retrieved_validated_payload"
    text = raw.decode("latin-1")
    parser_a = Inventory()
    parser_a.feed(text)
    parser_b = BeautifulSoup(text, "html.parser")
    pre_a = "".join(parser_a.text)
    pre_b = "".join(p.get_text() for p in parser_b.find_all("pre"))
    assert pre_a == pre_b
    links_b = [a.get("href", "") for a in parser_b.find_all("a")]
    assert parser_a.links == links_b
    entries = []
    for line in pre_a.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*(\S+)\s+(\w+\s+\d+\s+\d{4})\s+(.+)", line)
        assert match is not None
        size, date, name = match.groups()
        entries.append({"name": name, "display_size": size, "display_date": date})
    target = "CPTAC_Exp1_Curves_CS.sky"
    assert len(entries) == 15
    assert [e["display_size"] for e in entries if e["name"] == target] == ["7.8M"]
    assert not any(target in link for link in parser_a.links)
    certificate = json.loads(
        (root / "results/prior-ftps-certificate-diagnostic.json").read_text()
    )
    cert_text = json.dumps(certificate).lower()
    assert "hostname mismatch" in cert_text and "certificate has expired" in cert_text
    report = {
        "status": "verified_inventory_only_access_unresolved",
        "source_sha256": receipt["sha256"],
        "source_bytes": len(raw),
        "independent_html_inventory_and_link_agreement": True,
        "entries": entries,
        "target": target,
        "target_download_links": 0,
        "skyline_bytes_retrieved": 0,
        "semantic_admission": "not_performed_no_document",
        "training_or_outer_outcome_access": False,
        "decision": "Require repository-supported HTTPS or repaired verified FTPS; no further retries of these routes without changed evidence.",
    }
    with (root / "results/verification.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "entries"}))


if __name__ == "__main__":
    main()
