"""Bounded public primary-article retrieval; one operation, no automatic retry."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

import psutil

FAMILY = Path(__file__).resolve().parents[1]
RAW = Path("/private/tmp/presto-mhc2-20261026/human-cell-contrast/primary")
URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9724678/fullTextXML"


def main():
    receipt = FAMILY / "reproduce/primary-operation06.json"
    assert not receipt.exists()
    state = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "url": URL,
        "logical_operation": 6,
        "total_operation_budget": 16,
        "prior_operations": "four GitHub issue reads; one PMC web open returned browser challenge",
        "argv": sys.argv,
        "python": sys.version,
        "agent_model": "Codex / GPT-6",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            key: os.environ.get(key)
            for key in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "limits": {
            "seconds": 60,
            "bytes": 15 * 1024**2,
            "rss_bytes": 2 * 1024**3,
            "start_ram_bytes": 13 * 1024**3,
            "floor_ram_bytes": 10 * 1024**3,
        },
    }
    receipt.write_text(json.dumps(state, indent=2) + "\n")
    RAW.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def timeout(_signal, _frame):
        raise TimeoutError("primary fetch exceeded60seconds")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    destination = RAW / "PMC9724678.xml"
    try:
        assert psutil.virtual_memory().available >= 13 * 1024**3
        assert shutil.disk_usage(FAMILY).free >= 15 * 1024**3
        n = 0
        with urlopen(URL, timeout=20) as response, destination.open("xb") as output:
            state.update(
                http_status=response.status, content_type=response.headers.get("content-type")
            )
            while chunk := response.read(65536):
                n += len(chunk)
                assert n <= 15 * 1024**2
                assert psutil.virtual_memory().available >= 10 * 1024**3
                assert psutil.Process().memory_info().rss <= 2 * 1024**3
                output.write(chunk)
        content = destination.read_bytes()
        assert b"<article" in content and b"36215666" in content
        state.update(
            status="completed",
            path=str(destination),
            bytes=n,
            sha256=hashlib.sha256(content).hexdigest(),
        )
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        state.update(
            elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        receipt.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state))


if __name__ == "__main__":
    main()
