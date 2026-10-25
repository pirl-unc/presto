"""Bounded, non-overwriting launcher for the training-only diagnostic."""

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
config = json.loads((ROOT / "reproduce/launch.json").read_text())
stage = sys.argv[1]
assert stage in config["stages"]
for name, expected in config["inputs"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
script = ROOT / ("analysis" if stage == "replay" else "code") / f"{stage}.py"
assert script.is_file()
limits = config["limits"]
assert shutil.disk_usage(ROOT).free >= limits["free_disk_floor_bytes"] + 32 * 1024**2
for directory in ("results", "launch_logs", "reproduce/source"):
    (ROOT / directory).mkdir(parents=True, exist_ok=True)
position = config["stages"].index(stage)
for prerequisite in config["stages"][:position]:
    assert json.loads((ROOT / f"results/{prerequisite}-resource.json").read_text())["passed"]
with (ROOT / f"results/{stage}-started.json").open("x") as stream:
    json.dump({"stage": stage, "started_epoch": time.time()}, stream)
snapshot = ROOT / f"reproduce/source/{stage}"
snapshot.mkdir()
for p in (ROOT / "code").glob("*.py"):
    shutil.copyfile(p, snapshot / p.name)
if stage == "replay":
    shutil.copyfile(script, snapshot / script.name)
start = time.monotonic()
peak = 0
minimum_free = shutil.disk_usage(ROOT).free
reason = None
with (ROOT / f"launch_logs/{stage}.log").open("x") as stream:
    process = subprocess.Popen([config["python"], str(script)],
                               env={**os.environ, **config["environment_overrides"]},
                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    probe = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            peak = max(peak, probe.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        minimum_free = min(minimum_free, shutil.disk_usage(ROOT).free)
        logical = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
        if peak > limits["rss_bytes"]:
            reason = "rss_cap"
        elif time.monotonic() - start > limits["stage_seconds"]:
            reason = "time_cap"
        elif minimum_free < limits["free_disk_floor_bytes"]:
            reason = "disk_floor"
        elif logical > limits["new_logical_bytes"]:
            reason = "artifact_cap"
        if reason:
            os.killpg(process.pid, signal.SIGTERM)
            break
        try:
            process.wait(timeout=.2)
        except subprocess.TimeoutExpired:
            pass
    code = process.wait(timeout=30)
result = {"stage": stage, "passed": code == 0 and reason is None, "exit_code": code,
          "termination_reason": reason, "wall_seconds": time.monotonic() - start,
          "peak_polled_rss_bytes": peak, "minimum_free_bytes": minimum_free,
          "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
(ROOT / f"results/{stage}-resource.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
if not result["passed"]:
    raise SystemExit(code or 1)
