"""Bounded source admission stages, preserving every failed attempt."""

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
    if stage not in {"build", "audit"}:
        raise ValueError(stage)
    config = json.loads((ROOT / "reproduce/launch.json").read_text())
    for name, expected in config["runtime_sha256"].items():
        assert hashlib.sha256((Path(config["hitlist"]) / name).read_bytes()).hexdigest() == expected
    result_path = ROOT / f"results/{stage}-resource.json"
    lock = ROOT / f"results/{stage}-started.json"
    script = ROOT / ("analysis" if stage == "audit" else "code") / f"{stage}.py"
    with lock.open("x") as stream:
        json.dump({"stage": stage, "started_epoch": time.time()}, stream)
    (ROOT / f"reproduce/source/{stage}.py").write_bytes(script.read_bytes())
    limits = config["limits"]
    initial = shutil.disk_usage(ROOT).free
    assert initial >= limits["free_disk_floor_bytes"] + 32 * 1024**2
    start, peak, minimum, reason = time.monotonic(), 0, initial, None
    command = [config["python"], str(script)]
    with (ROOT / f"launch_logs/{stage}.log").open("x") as stream:
        process = subprocess.Popen(command, env={**os.environ, **config["environment_overrides"]},
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        probe = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, probe.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            free = shutil.disk_usage(ROOT).free
            minimum = min(minimum, free)
            if peak > limits["rss_bytes"]:
                reason = "rss_limit"
            elif time.monotonic() - start > limits["stage_seconds"]:
                reason = "stage_time_limit"
            elif free < limits["free_disk_floor_bytes"]:
                reason = "free_disk_floor"
            elif initial - free > limits["new_physical_bytes"]:
                reason = "conservative_shared_volume_delta_cap"
            if reason:
                os.killpg(process.pid, signal.SIGTERM)
                break
            try:
                process.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass
        code = process.wait(timeout=30)
    result = {"command": command, "exit_code": code, "passed": code == 0 and reason is None,
              "wall_seconds": time.monotonic() - start, "peak_polled_rss_bytes": peak,
              "free_before_bytes": initial, "minimum_free_bytes": minimum,
              "free_after_bytes": shutil.disk_usage(ROOT).free, "termination_reason": reason,
              "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
