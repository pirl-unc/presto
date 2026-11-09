"""One bounded retry after a proved public FTP read timeout; preserve prior bytes."""

import ftplib
import hashlib
import json
import shutil
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/measured_ms1_grouped_observer/2026-10-24_1702_v1"
NAME = "TUM_HLA_63_01_01_DDA-1h-R1-unspecific.zip"
started = time.perf_counter()
receipt = {
    "source_url": "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2021/04/PXD021013/" + NAME,
    "started_utc": datetime.now(UTC).isoformat(),
    "status": "started",
    "size_bytes": 0,
    "read_timeout_seconds": 120,
    "overall_deadline_seconds": 180,
}
assert not (RAW / NAME).exists()
assert not (RAW / (NAME + ".ftp-retry.receipt.json")).exists()
shutil.copy2(__file__, ROOT / "reproduce/source/retry_ftp.py")
prior = sum(json.loads(p.read_text())["size_bytes"] for p in RAW.glob("*.receipt.json"))
sha = hashlib.sha256()
ftp = ftplib.FTP(timeout=120)
try:
    ftp.connect("ftp.pride.ebi.ac.uk", timeout=20)
    ftp.login()
    ftp.cwd("/pride/data/archive/2021/04/PXD021013")
    ftp.voidcmd("TYPE I")
    receipt["declared_size"] = ftp.size(NAME)
    receipt["modified_before"] = ftp.sendcmd("MDTM " + NAME)
    assert receipt["declared_size"] <= 128 * 1024**2
    # connect(timeout=20) also sets the FTP object's data-socket timeout.
    ftp.timeout = 120
    partial = RAW / (NAME + ".ftp-retry.partial")
    with partial.open("xb") as output:

        def receive(chunk):
            receipt["size_bytes"] += len(chunk)
            assert time.perf_counter() - started <= 180
            assert (
                receipt["size_bytes"] <= 128 * 1024**2 and prior + receipt["size_bytes"] <= 1024**3
            )
            assert shutil.disk_usage(RAW).free >= 2 * 1024**3 + len(chunk)
            output.write(chunk)
            sha.update(chunk)

        ftp.retrbinary("RETR " + NAME, receive, blocksize=65536)
    receipt["modified_after"] = ftp.sendcmd("MDTM " + NAME)
    assert receipt["modified_before"] == receipt["modified_after"]
    assert receipt["size_bytes"] == receipt["declared_size"]
    first = (RAW / (NAME + ".ftp.partial")).read_bytes()
    with partial.open("rb") as stream:
        assert stream.read(len(first)) == first
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
    receipt["sha256"] = sha.hexdigest()
    receipt["verified_original_partial_prefix_bytes"] = len(first)
    receipt["zip_crc"] = "passed"
    partial.rename(RAW / NAME)
    receipt["status"] = "downloaded"
except Exception as exc:
    receipt["status"] = "failed"
    receipt["error"] = f"{type(exc).__name__}: {exc}"
    raise
finally:
    ftp.close()
    receipt["elapsed_seconds"] = time.perf_counter() - started
    (RAW / (NAME + ".ftp-retry.receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
