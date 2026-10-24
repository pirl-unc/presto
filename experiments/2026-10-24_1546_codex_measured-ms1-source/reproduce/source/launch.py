"""Bounded original-source retrieval only; never turns data into training labels."""

import argparse
import datetime
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = REPO / "data/measured_ms1_source_qualification/2026-10-24_1546_v1"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.initialize:
        RAW.mkdir(parents=True, exist_ok=False)
        for directory in ["results", "launch_logs", "reproduce", "reproduce/source"]:
            (ROOT / directory).mkdir(exist_ok=True)
        shutil.copyfile(__file__, ROOT / "reproduce/source/launch.py")
        shutil.copyfile(
            REPO / "experiments/agents/codex/plans/measured-ms1-source-qualification.md",
            ROOT / "reproduce/plan.md",
        )
        metadata = {
            "agent": "Codex / GPT-6",
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty_tracked": subprocess.check_output(
                ["git", "status", "--short", "--untracked-files=no"], text=True
            ),
            "new_untracked_experiment_and_plan": True,
            "raw_directory": str(RAW),
            "argv": sys.argv,
            "python": sys.executable,
            "platform": platform.platform(),
            "environment_overrides": {},
            "training": "none",
            "prediction_dumps": "not applicable",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "per_file_cap_bytes": 100 * 1024**2,
            "total_cap_bytes": 500 * 1024**2,
        }
        write(ROOT / "manifest.json", metadata)
        write(ROOT / "reproduce/launch.json", metadata)
        (ROOT / "reproduce/launch.sh").write_text(
            f"#!/bin/sh\nset -eu\ncd '{REPO}'\n'{sys.executable}' '{ROOT}/code/launch.py' --initialize\n"
        )
        print(json.dumps(metadata))
        return
    assert args.url and args.name and Path(args.name).name == args.name
    target = RAW / args.name
    receipt_path = RAW / f"{args.name}.receipt.json"
    assert RAW.is_dir() and not target.exists() and not receipt_path.exists()
    cap = min(
        100 * 1024**2, 500 * 1024**2 - sum(p.stat().st_size for p in RAW.iterdir() if p.is_file())
    )
    assert cap > 0
    started = time.perf_counter()
    receipt = {
        "argv": sys.argv,
        "url": args.url,
        "name": args.name,
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    try:
        request = urllib.request.Request(
            args.url, headers={"User-Agent": "Presto-public-source-audit/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            receipt.update(
                status=response.status,
                final_url=response.url,
                headers={
                    k: v
                    for k, v in response.headers.items()
                    if k.lower()
                    in {"content-type", "content-length", "etag", "last-modified", "date"}
                },
            )
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > cap:
                raise ValueError("Declared asset size exceeds registered cap")
            payload = response.read(cap + 1)
            if len(payload) > cap:
                raise ValueError("Observed asset size exceeds registered cap")
        target.write_bytes(payload)
        receipt.update(
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            path=str(target),
            access_status="retrieved_not_yet_qualified",
        )
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        receipt.update(error=str(error), access_status="failed_or_over_cap_not_absent")
    receipt["seconds"] = time.perf_counter() - started
    write(receipt_path, receipt)
    print(json.dumps(receipt))
    if not target.exists():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
