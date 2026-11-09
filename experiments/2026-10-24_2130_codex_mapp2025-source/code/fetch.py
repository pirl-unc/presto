"""Bounded original-asset retrieval for baseline MAPP source qualification."""

import hashlib
import json
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/mapp2025_source/2026-10-24_2130"
ASSETS = {
    "project.json": ("https://www.ebi.ac.uk/pride/ws/archive/v2/projects/PXD052314", 20),
    "files.json": (
        "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/PXD052314/files?page=0&pageSize=1000",
        20,
    ),
    "supplement.zip": (
        (
            "https://media.springernature.com/original/springer-static/esm/"
            "art%3A10.1038%2Fs41586-025-08615-w/MediaObjects/41586_2025_8615_MOESM5_ESM.zip"
        ),
        256,
    ),
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    (EXP / "results").mkdir(exist_ok=True)
    start = time.monotonic()
    receipts = []
    for name, (url, cap_mib) in ASSETS.items():
        dst = RAW / name
        if dst.exists():
            raise FileExistsError(dst)
        receipt = {"name": name, "url": url, "utc": datetime.now(timezone.utc).isoformat()}
        t0 = time.monotonic()
        try:
            with requests.get(url, stream=True, timeout=(20, 60)) as response:
                receipt["status"] = response.status_code
                receipt["final_url"] = response.url
                receipt["headers"] = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower() in {"content-type", "content-length", "last-modified", "etag"}
                }
                response.raise_for_status()
                cap = cap_mib * 1024**2
                if int(response.headers.get("content-length", "0")) > cap:
                    raise ValueError("declared content exceeds cap")
                size = 0
                sha = hashlib.sha256()
                with dst.with_suffix(dst.suffix + ".partial").open("wb") as handle:
                    for chunk in response.iter_content(1024**2):
                        size += len(chunk)
                        if size > cap:
                            raise ValueError("stream exceeds cap")
                        handle.write(chunk)
                        sha.update(chunk)
                dst.with_suffix(dst.suffix + ".partial").rename(dst)
                receipt.update(bytes=size, sha256=sha.hexdigest(), path=str(dst), ok=True)
        except (requests.RequestException, OSError, ValueError) as exc:
            receipt.update(ok=False, error=str(exc))
        receipt["seconds"] = time.monotonic() - t0
        receipts.append(receipt)
        (EXP / "results/retrieval.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    stats = {
        "seconds": time.monotonic() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (EXP / "results/retrieval-runtime.json").write_text(json.dumps(stats, indent=2) + "\n")
    assert all(r["ok"] for r in receipts), "asset failures retained in retrieval.json"


if __name__ == "__main__":
    main()
