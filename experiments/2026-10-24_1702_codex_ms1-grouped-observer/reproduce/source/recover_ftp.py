"""One bounded anonymous retrieval of the exact public FTP URL in the source index."""

import ftplib
import hashlib
import json
import resource
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/measured_ms1_grouped_observer/2026-10-24_1702_v1"
NAME = "TUM_HLA_63_01_01_DDA-1h-R1-unspecific.zip"
started = time.perf_counter()
receipt = {
    "source_url": "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2021/04/PXD021013/" + NAME,
    "started_utc": datetime.now(UTC).isoformat(),
    "size_bytes": 0,
    "status": "started",
    "authentication": "documented_public_anonymous_FTP",
}
assert not (RAW / NAME).exists()
assert not (RAW / (NAME + ".ftp.receipt.json")).exists()
shutil.copy2(__file__, ROOT / "reproduce/source/recover_ftp.py")
prior_bytes = sum(json.loads(p.read_text())["size_bytes"] for p in RAW.glob("*.receipt.json"))
sha = hashlib.sha256()
try:
    with ftplib.FTP("ftp.pride.ebi.ac.uk", timeout=30) as ftp:
        receipt["welcome"] = ftp.getwelcome()
        receipt["login_response"] = ftp.login()
        ftp.cwd("/pride/data/archive/2021/04/PXD021013")
        ftp.voidcmd("TYPE I")
        receipt["declared_size"] = ftp.size(NAME)
        assert receipt["declared_size"] <= 128 * 1024**2
        partial = RAW / (NAME + ".ftp.partial")
        with partial.open("xb") as stream:

            def receive(data):
                receipt["size_bytes"] += len(data)
                assert receipt["size_bytes"] <= 128 * 1024**2
                assert prior_bytes + receipt["size_bytes"] <= 1024**3
                assert shutil.disk_usage(RAW).free >= 2 * 1024**3 + len(data)
                stream.write(data)
                sha.update(data)

            ftp.retrbinary("RETR " + NAME, receive, blocksize=1024**2)
        assert receipt["size_bytes"] == receipt["declared_size"]
        receipt["sha256"] = sha.hexdigest()
        partial.rename(RAW / NAME)
        receipt["status"] = "downloaded"
except Exception as exc:
    receipt["status"] = "failed"
    receipt["error"] = f"{type(exc).__name__}: {exc}"
    raise
finally:
    receipt["elapsed_seconds"] = time.perf_counter() - started
    receipt["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    (RAW / (NAME + ".ftp.receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
