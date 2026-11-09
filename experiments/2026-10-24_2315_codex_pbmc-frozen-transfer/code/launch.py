"""Single ordered stage with RSS/time/disk guards and immutable receipts."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    stage = sys.argv[1]
    if stage not in ("source", "admit", "audit", "predict", "evaluate", "replay", "closure"):
        raise ValueError(stage)
    receipt = ROOT / f"results/{stage}-resource.json"
    if receipt.exists():
        raise FileExistsError("Preserve the prior attempt; no automatic retry")
    config = json.loads((ROOT / "reproduce/launch.json").read_text())
    limits = config["limits"]
    script = ROOT / ("analysis" if stage in ("audit", "replay") else "code") / f"{stage}.py"
    (ROOT / f"reproduce/source/{stage}.py").write_bytes(script.read_bytes())
    initial_free = shutil.disk_usage(ROOT).free
    if initial_free < limits["free_disk_floor_bytes"] + 32 * 1024**2:
        raise RuntimeError("Insufficient bounded-stage disk headroom")
    start = time.monotonic()
    peak = 0
    minimum_free = initial_free
    reason = None
    command = [config["python"], str(script)]
    with (ROOT / f"launch_logs/{stage}.log").open("w") as stream:
        process = subprocess.Popen(
            command,
            env={**os.environ, **config["environment_overrides"]},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        probe = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, probe.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            free = shutil.disk_usage(ROOT).free
            minimum_free = min(minimum_free, free)
            if peak > limits["rss_bytes"]:
                reason = "rss_limit"
            elif time.monotonic() - start > limits["stage_seconds"]:
                reason = "stage_time_limit"
            elif free < limits["free_disk_floor_bytes"]:
                reason = "free_disk_floor"
            elif initial_free - free > limits["new_physical_bytes"]:
                reason = "conservative_volume_free_delta_cap"
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                break
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        code = process.wait(timeout=30)
    result = {
        "command": command,
        "exit_code": code,
        "wall_seconds": time.monotonic() - start,
        "peak_polled_rss_bytes": peak,
        "free_before_bytes": initial_free,
        "minimum_free_bytes": minimum_free,
        "free_after_bytes": shutil.disk_usage(ROOT).free,
        "termination_reason": reason,
        "passed": code == 0 and reason is None,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "disk_metric": "conservative shared-volume free-space delta; includes other activity",
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
