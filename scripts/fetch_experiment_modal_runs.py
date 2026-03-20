#!/usr/bin/env python
"""Fetch completed Modal run directories for a manifest-backed experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise SystemExit(f"Expected list payload in {path}")
    rows: list[dict[str, Any]] = []
    for row in payload:
        if isinstance(row, dict) and row.get("run_id"):
            rows.append(row)
    return rows


def _required_files(row: dict[str, Any]) -> tuple[str, ...]:
    values = row.get("required_files")
    if isinstance(values, list):
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if cleaned:
            return tuple(cleaned)
    return ("summary.json",)


def _remote_artifacts(volume: str, run_id: str) -> list[str]:
    result = _run(["modal", "volume", "ls", volume, run_id])
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _completed_run_ids(volume: str) -> set[str]:
    result = _run(["modal", "volume", "ls", volume])
    if result.returncode != 0:
        return set()
    run_ids: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        run_ids.add(line.split("/", 1)[0])
    return run_ids


def _local_ready(dest_root: Path, run_id: str, required_files: tuple[str, ...]) -> bool:
    run_dir = dest_root / run_id
    return all((run_dir / name).exists() for name in required_files)


def _remote_ready(remote_artifacts: list[str], required_files: tuple[str, ...]) -> bool:
    present = {Path(path).name for path in remote_artifacts}
    return all(name in present for name in required_files)


def _fetch_run(volume: str, run_id: str, dest_root: Path, required_files: tuple[str, ...]) -> bool:
    dest_root.mkdir(parents=True, exist_ok=True)
    result = _run(["modal", "volume", "get", "--force", volume, f"{run_id}/", str(dest_root)])
    return result.returncode == 0 and _local_ready(dest_root, run_id, required_files)


def _status_row(
    *,
    run_id: str,
    required_files: tuple[str, ...],
    local_ready: bool,
    remote_artifacts: list[str],
    fetched: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "required_files": list(required_files),
        "local_ready": local_ready,
        "remote_artifact_count": len(remote_artifacts),
        "remote_ready": _remote_ready(remote_artifacts, required_files),
        "fetched": fetched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch completed Modal experiment runs into results/runs.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--results-subdir", default="results/runs")
    parser.add_argument("--volume", default="presto-checkpoints")
    parser.add_argument("--poll-interval-sec", type=float, default=60.0)
    parser.add_argument("--max-wait-sec", type=float, default=0.0)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    manifest_path = experiment_dir / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    results_root = experiment_dir / args.results_subdir
    results_root.mkdir(parents=True, exist_ok=True)
    status_path = experiment_dir / "results" / "fetch_status.json"
    deadline = time.time() + float(args.max_wait_sec) if float(args.max_wait_sec) > 0 else None

    fetched_run_ids: set[str] = set()
    while True:
        completed_remote = _completed_run_ids(str(args.volume))
        rows: list[dict[str, Any]] = []
        pending = 0
        for item in manifest:
            run_id = str(item["run_id"])
            required_files = _required_files(item)
            local_ready = _local_ready(results_root, run_id, required_files)
            remote_artifacts = _remote_artifacts(str(args.volume), run_id) if run_id in completed_remote else []
            fetched = local_ready
            if not local_ready and run_id in completed_remote:
                fetched = _fetch_run(str(args.volume), run_id, results_root, required_files)
            if fetched:
                fetched_run_ids.add(run_id)
            else:
                pending += 1
            rows.append(
                _status_row(
                    run_id=run_id,
                    required_files=required_files,
                    local_ready=_local_ready(results_root, run_id, required_files),
                    remote_artifacts=remote_artifacts,
                    fetched=fetched,
                )
            )

        payload = {
            "experiment_dir": str(experiment_dir),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "fetched_runs": len(fetched_run_ids),
            "total_runs": len(manifest),
            "rows": rows,
        }
        status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "fetched_runs": len(fetched_run_ids),
                    "pending_runs": pending,
                    "status_path": str(status_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        if pending == 0:
            return
        if not args.wait:
            return
        if deadline is not None and time.time() >= deadline:
            return
        time.sleep(float(args.poll_interval_sec))


if __name__ == "__main__":
    main()
