"""Finish frozen public assets with verified replay and bounded official routes."""

import ftplib
import hashlib
import json
import resource
import shutil
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests
from hitlist.proteometools_ms1 import BASE_URL, CORE
from hitlist.provenance import file_digest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/measured_ms1_grouped_observer/2026-10-24_1702_v1"
GIB = 1024**3
S3 = "https://www.ebi.ac.uk/pride/ws/archive-file-downloader/files/s3/PXD021013/"
FTP = "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2021/04/PXD021013/"


def main():
    started = time.perf_counter()
    assert not (ROOT / "reproduce/archive-receipts.json").exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/finish_acquisition.py")
    selection = json.loads((ROOT / "results/selection.json").read_text())
    assert len(selection["selected"]) == 24
    for spec in json.loads(CORE.read_text())["assets"]:
        assert file_digest(RAW / spec["filename"]) == {
            k: spec[k] for k in ("sha256", "size_bytes")
        }
    total = sum(
        json.loads(p.read_text())["size_bytes"] for p in RAW.glob("*.receipt.json")
    )
    archives = []
    for number, pool in enumerate(selection["selected"], 1):
        name = pool["source_name"]
        path = RAW / name
        previous = [
            json.loads(p.read_text()) for p in RAW.glob(name + "*.receipt.json")
        ]
        if path.exists():
            successful = [r for r in previous if r.get("status") == "downloaded"]
            assert successful
            digest = file_digest(path)
            assert all(digest == {k: r[k] for k in digest} for r in successful)
        else:
            attempted = {r["source_url"] for r in previous}
            for route, url in (
                ("https", BASE_URL + name),
                ("s3", S3 + name),
                ("ftp", FTP + name),
            ):
                if url in attempted:
                    continue
                receipt_path = RAW / f"{name}.finish-{route}.receipt.json"
                partial = RAW / f"{name}.finish-{route}.partial"
                assert not receipt_path.exists() and not partial.exists()
                receipt = {
                    "pool": pool["pool"],
                    "source_url": url,
                    "deposited_url": BASE_URL + name,
                    "started_utc": datetime.now(UTC).isoformat(),
                    "status": "started",
                    "size_bytes": 0,
                }
                tick = time.perf_counter()
                sha = hashlib.sha256()

                def receive(chunk, receipt=receipt, tick=tick, sha=sha):
                    nonlocal total
                    receipt["size_bytes"] += len(chunk)
                    total += len(chunk)
                    assert time.perf_counter() - tick <= 180, (
                        "Per-route deadline reached"
                    )
                    assert receipt["size_bytes"] <= 128 * 1024**2 and total <= GIB, (
                        "Transfer cap"
                    )
                    assert shutil.disk_usage(RAW).free >= 2 * GIB + len(chunk), (
                        "Disk floor"
                    )
                    output.write(chunk)
                    sha.update(chunk)

                try:
                    if route == "ftp":
                        client = ftplib.FTP(timeout=120)
                        try:
                            client.connect("ftp.pride.ebi.ac.uk", timeout=20)
                            receipt["login_response"] = client.login()
                            client.cwd("/pride/data/archive/2021/04/PXD021013")
                            client.voidcmd("TYPE I")
                            receipt["declared_size"] = client.size(name)
                            receipt["modified_before"] = client.sendcmd("MDTM " + name)
                            assert receipt["declared_size"] <= 128 * 1024**2
                            client.timeout = 120
                            with partial.open("xb") as output:
                                client.retrbinary(
                                    "RETR " + name, receive, blocksize=65536
                                )
                            receipt["modified_after"] = client.sendcmd("MDTM " + name)
                            assert (
                                receipt["modified_before"] == receipt["modified_after"]
                            )
                            assert receipt["declared_size"] == receipt["size_bytes"]
                        finally:
                            client.close()
                    else:
                        with requests.get(
                            url, stream=True, timeout=(20, 60)
                        ) as response:
                            receipt.update(
                                http_status=response.status_code,
                                final_url=response.url,
                                headers=dict(response.headers),
                            )
                            response.raise_for_status()
                            if "Content-Length" in response.headers:
                                assert (
                                    int(response.headers["Content-Length"])
                                    <= 128 * 1024**2
                                )
                            with partial.open("xb") as output:
                                for chunk in response.iter_content(65536):
                                    if chunk:
                                        receive(chunk)
                    assert receipt["size_bytes"] > 0
                    with zipfile.ZipFile(partial) as archive:
                        assert archive.testzip() is None
                    receipt.update(
                        status="downloaded", sha256=sha.hexdigest(), zip_crc="passed"
                    )
                    partial.rename(path)
                except (
                    requests.RequestException,
                    ftplib.Error,
                    TimeoutError,
                    OSError,
                ) as exc:
                    receipt.update(
                        status="failed", error=f"{type(exc).__name__}: {exc}"
                    )
                except Exception as exc:
                    receipt.update(
                        status="failed", error=f"{type(exc).__name__}: {exc}"
                    )
                    raise
                finally:
                    receipt.update(
                        ended_utc=datetime.now(UTC).isoformat(),
                        elapsed_seconds=time.perf_counter() - tick,
                    )
                    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                    print(json.dumps(receipt), flush=True)
                if path.exists():
                    break
            assert path.exists(), f"All documented routes exhausted for {name}"
        archives.append({"filename": name, "source_name": name, **file_digest(path)})
        print(f"Verified {number}/24: {name}", flush=True)
    (ROOT / "reproduce/archive-receipts.json").write_text(
        json.dumps(archives, indent=2) + "\n"
    )
    report = {
        "n_archives": len(archives),
        "transfer_bytes_including_prior_attempts": total,
        "seconds_this_resume": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB,
        "argv": sys.argv,
        "python": sys.executable,
        "selection": file_digest(ROOT / "results/selection.json"),
        "original_receipts": [str(p) for p in sorted(RAW.glob("*.receipt.json"))],
    }
    assert report["peak_rss_gib"] <= 3
    (ROOT / "results/acquisition-summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
