"""Verify downloaded canonical bytes; this does not admit training observations."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]
RAW = Path("/private/tmp/presto-mhc2-20261026/ms-source")


def main():
    checks = {}
    for line in (RAW / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            sha, name = line.split()
            name = name.lstrip("*")
            if name in checks:
                raise ValueError(f"duplicate checksum entry: {name}")
            checks[name] = sha
    files = []
    for name in ("training.parquet", "manifest.json", "QUALIFICATION.md"):
        path = RAW / name
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha != checks[name]:
            raise ValueError(f"published checksum mismatch: {name}")
        files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha})
    if sum(row["bytes"] for row in files) > 256 * 1024**2:
        raise ValueError("registered download byte ceiling exceeded")
    table = pq.ParquetFile(RAW / "training.parquet")
    report = {
        "agent_model": "Codex / GPT-6",
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_release": "human-cohort-corrected-v1",
        "all_published_checksums_match": True,
        "files": files,
        "checksum_file_sha256": hashlib.sha256((RAW / "SHA256SUMS").read_bytes()).hexdigest(),
        "n_source_rows": table.metadata.num_rows,
        "schema": str(table.schema_arrow),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "admitted_training_rows": 0,
        "qualification": "byte verification only; inspect release constraints before admission",
    }
    destination = FAMILY / "reproduce/source-verification.json"
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    snapshot = FAMILY / "reproduce/source/verify_source.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "verified_files": len(files),
                "n_source_rows": table.metadata.num_rows,
                "admitted_training_rows": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
