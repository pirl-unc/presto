"""Fetch one inventory-selected PRIDE asset with Hitlist's shared downloader."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from hitlist.downloads import download_to_file
from hitlist.provenance import file_digest

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()
experiment = Path(__file__).resolve().parents[1]
config = json.loads((experiment / "reproduce/launch.json").read_text())
root = Path(config["source_root"])
rows = []
for path in sorted((root / "retrieval").glob("pride-files-page*/body")):
    rows.extend(json.loads(path.read_text()))
matches = [row for row in rows if row["fileName"] == args.filename]
assert len(matches) == 1, "Require an unambiguous original deposit record"
record = matches[0]
assert 0 < record["fileSizeBytes"] < 500_000_000, "Explicit 500 MB per asset ceiling"
ftp_urls = [p["value"] for p in record["publicFileLocations"] if p["value"].startswith("ftp://")]
assert len(ftp_urls) == 1
parts = urlsplit(ftp_urls[0])
assert parts.hostname == "ftp.pride.ebi.ac.uk" and parts.username is None
url = urlunsplit(("https", parts.netloc, parts.path, parts.query, parts.fragment))
destination = root / "deposits" / args.filename
destination.parent.mkdir(parents=True, exist_ok=True)
result_path = experiment / "results" / f"deposit-{args.filename}.json"
assert not result_path.exists(), "Preserve prior attempt receipts"
shutil.copy2(__file__, experiment / "reproduce/source/fetch_deposit.py")
receipt = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "command": sys.argv,
    "source_record": record,
    "requested_url": url,
    "url_derivation": "HTTPS transport for the exact deposited FTP hostname/path",
    "destination": str(destination),
    "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "presto_commit": config["presto_commit"],
    "hitlist_commit": config["hitlist_base_commit"],
    "environment": config["environment"],
}
started = time.monotonic()
try:
    download_to_file(url, destination, expected_size=record["fileSizeBytes"])
    receipt["actual"] = file_digest(destination)
except Exception as error:
    receipt["error"] = f"{type(error).__name__}: {error}"
receipt["elapsed_seconds"] = time.monotonic() - started
result_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
if "error" in receipt:
    raise SystemExit(1)
