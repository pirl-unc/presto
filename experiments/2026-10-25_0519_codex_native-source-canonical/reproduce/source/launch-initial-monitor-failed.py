"""Bounded local source replay, with durable terminal receipts."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
stage = sys.argv[1]
assert stage in {"build", "verify"}
receipt = ROOT / "results" / (stage + "-runtime.json")
assert not receipt.exists(), "Do not overwrite a completed or failed attempt"
for filename, digest in CONFIG["hitlist_overlay_sha256"].items():
    assert hashlib.sha256((Path(CONFIG["hitlist_repo"]) / filename).read_bytes()).hexdigest() == digest
for file in Path(__file__).parent.glob("*.py"):
    frozen = ROOT / "reproduce/source" / file.name
    if frozen.exists():
        assert frozen.read_bytes() == file.read_bytes(), "Register changed executed source separately"
    else:
        shutil.copy2(file, frozen)
limits = CONFIG["limits"]
assert shutil.disk_usage(ROOT).free >= limits["disk_floor_bytes"] + limits["start_margin_bytes"]
environment = {**os.environ, **CONFIG["non_default_environment_overrides"]}
command = [CONFIG["python"], "-B", str(ROOT / "code" / (stage + ".py"))]
started = time.monotonic()
peak, violation = 0, None
with (ROOT / "launch_logs" / (stage + ".log")).open("x") as log:
    child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    process = psutil.Process(child.pid)
    while child.poll() is None:
        try:
            rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])
            peak = max(peak, rss)
        except psutil.NoSuchProcess:
            rss = 0
        if time.monotonic() - started > limits["stage_seconds"]:
            violation = "stage_time_limit"
        elif rss > limits["rss_bytes"]:
            violation = "process_tree_rss_limit"
        elif shutil.disk_usage(ROOT).free < limits["disk_floor_bytes"]:
            violation = "disk_reserve"
        if violation:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
            break
        time.sleep(1)
    returncode = child.wait()
result = dict(stage=stage, argv=command, elapsed_seconds=time.monotonic()-started,
              peak_observed_process_tree_rss_bytes=peak, exit_code=returncode,
              resource_violation=violation, free_bytes_after=shutil.disk_usage(ROOT).free)
receipt.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
sys.exit(1 if violation or returncode else 0)
