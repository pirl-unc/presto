"""Bounded local prospective contrast fitting, with durable terminal receipts."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import resource

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
stage = sys.argv[1]
assert stage in CONFIG["stages"]
for earlier in CONFIG["stages"][: CONFIG["stages"].index(stage)]:
    prior = json.loads((ROOT / "results" / (earlier + "-runtime.json")).read_text())
    assert prior["exit_code"] == 0 and prior["resource_violation"] is None
assert stage != "qualify", "Preserve the completed original qualification"
if stage in {"fit", "evaluate", "replay"}:
    bindings = json.loads((ROOT / "results/code-bindings.json").read_text())
    evidence_path = ROOT / "reproduce/merged-code-evidence.json"
    assert (
        hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        == bindings["evidence_sha256"]
    )
receipt = ROOT / "results" / (stage + "-runtime.json")
assert not receipt.exists(), "Do not overwrite a completed or failed attempt"
for filename, digest in CONFIG["input_sha256"].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
for file in Path(__file__).parent.glob("*.py"):
    frozen = ROOT / "reproduce/source" / file.name
    if frozen.exists():
        assert frozen.read_bytes() == file.read_bytes(), (
            "Register changed executed source separately"
        )
    else:
        shutil.copy2(file, frozen)
limits = CONFIG["limits"]
assert (
    shutil.disk_usage(ROOT).free
    >= limits["disk_floor_bytes"] + limits["start_margin_bytes"]
)
environment = {**os.environ, **CONFIG["non_default_environment_overrides"]}
command = [
    CONFIG["python"],
    "-B",
    str(ROOT / "code" / ("prepare_local.py" if stage == "prepare" else stage + ".py")),
]
process_probe = psutil.Process()
process_probe.memory_info()
process_probe.children(
    recursive=True
)  # Fail before launch if monitoring is unavailable.
started = time.monotonic()
peak, violation = 0, None
with (ROOT / "launch_logs" / (stage + ".log")).open("x") as log:
    child = subprocess.Popen(
        command, env=environment, stdout=log, stderr=subprocess.STDOUT
    )
    (ROOT / "results" / (stage + "-process.json")).write_text(
        json.dumps({"pid": child.pid, "command": command}) + "\n"
    )
    process = psutil.Process(child.pid)
    while child.poll() is None:
        try:
            rss = sum(
                p.memory_info().rss
                for p in [process, *process.children(recursive=True)]
            )
            peak = max(peak, rss)
        except psutil.NoSuchProcess:
            rss = 0
        except Exception as error:
            violation = "monitor_error: " + repr(error)
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
        time.sleep(0.05)
    returncode = child.wait()
child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
child_peak_bytes = int(child_peak if sys.platform == "darwin" else child_peak * 1024)
if child_peak_bytes > limits["rss_bytes"]:
    violation = "terminal_child_peak_rss_limit"
result = dict(
    stage=stage,
    argv=command,
    elapsed_seconds=time.monotonic() - started,
    peak_observed_process_tree_rss_bytes=peak,
    peak_child_process_rss_bytes=child_peak_bytes,
    rss_poll_interval_seconds=0.05,
    exit_code=returncode,
    resource_violation=violation,
    free_bytes_after=shutil.disk_usage(ROOT).free,
)
receipt.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
sys.exit(1 if violation or returncode else 0)
