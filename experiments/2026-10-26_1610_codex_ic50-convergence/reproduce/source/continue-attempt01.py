"""Finish the fixed family sequentially, preserving failures and terminal evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FAMILY = Path(__file__).resolve().parents[1]
REPRO = FAMILY / "reproduce"
ROOT = FAMILY.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def check_sources():
    contract = read(REPRO / "continuation-contract.json")
    for filename, expected in contract["source_hashes"].items():
        actual = hashlib.sha256((FAMILY / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("continuation source changed: " + filename)
    return contract


def wait_for_receipt(path, pid):
    while True:
        receipt = read(path)
        if receipt["status"] == "completed":
            return
        if receipt["status"] != "starting":
            raise RuntimeError("existing condition failed: " + str(receipt.get("failure")))
        # Do not mistake an orphaned starting receipt for a live fit.
        os.kill(pid, 0)
        started = datetime.fromisoformat(receipt["utc"])
        if (datetime.now(timezone.utc) - started).total_seconds() > receipt["limits"][
            "seconds"
        ] + 120:
            raise TimeoutError("existing condition exceeded its registered wall-time guard")
        time.sleep(10)


def execute(script, action, log):
    check_sources()
    # Exclusive logs prevent an automatic retry from overwriting a prior attempt.
    with log.open("x") as stream:
        subprocess.run(
            [sys.executable, str(FAMILY / "code" / script), action],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-receipt", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    contract = check_sources()
    destination = REPRO / "continuation-state.json"
    if destination.exists():
        raise FileExistsError("continuation already recorded; inspect it before any retry")
    state = dict(
        status="running",
        pid=os.getpid(),
        argv=sys.argv,
        started_utc=datetime.now(timezone.utc).isoformat(),
        contract=contract,
        events=[],
        current_action="waiting for seed42",
    )
    write(destination, state)

    def mark(action):
        state["current_action"] = action
        state["events"].append({"utc": datetime.now(timezone.utc).isoformat(), "action": action})
        write(destination, state)
        print(action, flush=True)

    try:
        mark("waiting for seed42")
        wait_for_receipt(args.wait_receipt, args.pid)
        for condition in ("seed42", "seed43", "seed44", "permuted42"):
            if condition != "seed42":
                mark("fitting " + condition)
                execute("launch.py", condition, FAMILY / "launch_logs" / f"{condition}.log")
            mark("verifying " + condition)
            execute("verify.py", condition, FAMILY / "launch_logs" / f"verify-{condition}.log")
            mark("reporting " + condition)
            execute("report.py", condition, FAMILY / "launch_logs" / f"report-{condition}.log")
        state["status"] = "completed"
        state["current_action"] = "fixed family verified; scientific gates require review"
    except BaseException as exc:
        state.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(destination, state)
        # Refresh saved-output status even after a failure, without repeating fits/evaluations.
        subprocess.run(
            [sys.executable, str(FAMILY / "code/report.py"), "refresh"], cwd=ROOT, check=False
        )


if __name__ == "__main__":
    main()
