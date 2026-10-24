"""Freeze and record one source-audit phase, preserving failed executions."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

family = Path(__file__).resolve().parents[1]
phase = sys.argv[1]
assert phase in {"audit", "canonical_audit"}
source = family / "code" / (phase + ".py")
receipt_path = family / "reproduce" / (phase + "-execution.json")
assert not receipt_path.exists()
snapshot = family / "reproduce/source" / source.name
assert not snapshot.exists()
shutil.copyfile(source, snapshot)
config = json.loads((family / "reproduce/launch.json").read_text())
env = dict(os.environ, **config["environment"])
command = [config["python"], str(source)]
receipt = dict(command=command, environment=config["environment"],
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
tick = time.perf_counter()
with (family / "launch_logs" / (phase + ".txt")).open("w") as log:
    result = subprocess.run(command, env=env, cwd=family.parents[1], stdout=log,
                            stderr=subprocess.STDOUT, timeout=180 if phase == "canonical_audit" else 720)
receipt.update(returncode=result.returncode, seconds=time.perf_counter()-tick)
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
raise SystemExit(result.returncode)
