"""Retrieve one explicitly published public PASS processed document, without logging login."""

import argparse
import ftplib
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]


class PublicLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.startswith("ftp://"):
                self.links.append(urlparse(href))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.requests.read_text())
    cfg = json.loads((ROOT / "reproduce/launch-v4.json").read_text())
    data = Path(cfg["data_directory"])
    source = data / "pass05890-sbeams.html"
    source_raw = source.read_bytes()
    source_receipt = json.loads(
        source.with_name(source.name + ".receipt.json").read_text()
    )
    assert hashlib.sha256(source_raw).hexdigest() == source_receipt["sha256"]
    links = PublicLinks()
    links.feed(source_raw.decode("latin-1"))
    matches = [
        u
        for u in links.links
        if u.hostname == "ftp.peptideatlas.org" and u.username == "PASS05890"
    ]
    assert len(matches) == 1 and matches[0].password
    url = matches[0]
    name = request["filename"]
    assert name == "CPTAC_Exp1_Curves_CS.sky" and name in source_raw.decode("latin-1")
    destination = data / name
    partial = destination.with_suffix(destination.suffix + ".part")
    receipt_path = ROOT / "results/pass-skyline-retrieval.json"
    assert not any(p.exists() for p in (destination, partial, receipt_path))
    total = sum(p.stat().st_size for p in data.iterdir() if p.is_file())
    receipt = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": url.hostname,
        "accession": "PASS05890",
        "filename": name,
        "public_route_source_sha256": source_receipt["sha256"],
        "bytes": 0,
        "status": "started",
        "credentials_logged": False,
    }
    started = time.monotonic()
    digest = hashlib.sha256()
    try:
        with ftplib.FTP(timeout=30) as client:
            client.connect(url.hostname, 21, timeout=30)
            client.login(unquote(url.username), unquote(url.password))
            client.voidcmd("TYPE I")
            size = client.size(name)
            receipt["remote_bytes"] = size
            assert size is not None and 0 < size <= 256 * 1024**2
            assert total + size <= 512 * 1024**2
            with partial.open("xb") as output:

                def receive(chunk):
                    if time.monotonic() - started > 120:
                        raise TimeoutError("transfer_wall_time")
                    if receipt["bytes"] + len(chunk) > size:
                        raise ValueError("remote_size_exceeded")
                    if shutil.disk_usage(data).free < cfg["limits"]["disk_floor_bytes"]:
                        raise ValueError("disk_reserve")
                    output.write(chunk)
                    digest.update(chunk)
                    receipt["bytes"] += len(chunk)

                client.retrbinary("RETR " + name, receive, blocksize=128 * 1024)
            assert receipt["bytes"] == size
        partial.rename(destination)
        receipt["xml_root"] = ET.parse(destination).getroot().tag
        assert receipt["xml_root"] == "srm_settings"
        receipt["status"] = "retrieved_validated_processed_document"
    except (
        OSError,
        EOFError,
        ftplib.Error,
        AssertionError,
        ValueError,
        ET.ParseError,
    ) as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error).replace(
            unquote(url.password), "[public dataset password omitted]"
        )
    receipt["sha256"] = digest.hexdigest()
    receipt["elapsed_seconds"] = time.monotonic() - started
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "retrieved_validated_processed_document" else 1


if __name__ == "__main__":
    raise SystemExit(main())
