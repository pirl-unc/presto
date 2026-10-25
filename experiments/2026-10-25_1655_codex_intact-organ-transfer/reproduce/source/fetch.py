"""Run a separately pinned local analysis through the unchanged resource guard."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--requests", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
request = json.loads(args.requests.read_text())
relative = request["task_source"]
task = (root / relative).resolve()
assert task.is_relative_to(root / "analysis") and task.suffix == ".py"
assert (
    hashlib.sha256(task.read_bytes()).hexdigest() == request["source_sha256"][relative]
)
runpy.run_path(str(task), run_name="__main__")
