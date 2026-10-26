"""One public XML request per prespecified paper, preserving failures."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

FAMILY = Path(__file__).resolve().parents[1]
RAW = Path("/private/tmp/presto-mhc2-20261026/ms-transfer-cohort/primary")
PAPERS = {
    "33858848": "PMC8054196",
    "25502872": "PMC4297300",
    "30573663": "PMC6398211",
    "33592498": "PMC8724627",
}


def main():
    RAW.mkdir(parents=True, exist_ok=False)
    receipt = {
        "agent_model": "Codex / GPT-6",
        "utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prior_network_operations": 19,
        "max_network_operations": 24,
        "requests": [],
    }
    destination = FAMILY / "reproduce/primary-retrieval.json"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    for pmid, pmc in PAPERS.items():
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML"
        started = time.monotonic()
        item = {"pmid": pmid, "pmc": pmc, "url": url, "status": "attempting"}
        receipt["requests"].append(item)
        destination.write_text(json.dumps(receipt, indent=2) + "\n")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Presto source qualification/1.0"}
            )
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    body = response.read(10 * 1024**2 + 1)
                    item.update(http_status=response.status, final_url=response.url)
            except urllib.error.HTTPError as error:
                body = error.read(10 * 1024**2 + 1)
                item.update(http_status=error.code, final_url=error.url)
            if len(body) > 10 * 1024**2:
                raise ValueError("response exceeded10MiB cap")
            path = RAW / (pmc + ".xml")
            path.write_bytes(body)
            item.update(path=str(path), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            assert item["http_status"] == 200, "HTTP failure"
            article = ET.fromstring(body)
            assert article.tag == "article", "response is not an article"
            ids = {
                node.get("pub-id-type"): node.text
                for node in article.findall("./front/article-meta/article-id")
            }
            assert ids.get("pmid") == pmid, ids
            item.update(status="verified_original_xml", article_ids=ids)
        except Exception as error:
            item.update(status="failed", error=repr(error))
        item["elapsed_seconds"] = time.monotonic() - started
        destination.write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps(
                {k: item.get(k) for k in ("pmid", "status", "http_status", "bytes", "error")}
            ),
            flush=True,
        )
    if any(item["status"] != "verified_original_xml" for item in receipt["requests"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
