"""Wait for verified IC50 completion, then execute the three frozen species seeds."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
REPRO = FAMILY / "reproduce"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check():
    contract = read(REPRO / "continuation-contract.json")
    for name, expected in contract["hashes"].items():
        if hashlib.sha256((FAMILY / name).read_bytes()).hexdigest() != expected:
            raise ValueError("continuation source changed: " + name)
    return contract


def execute(arguments, log, timeout):
    check()
    with (FAMILY / "launch_logs" / log).open("x") as stream:
        subprocess.run(
            arguments,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def main():
    contract = check()
    destination = REPRO / "continuation-state.json"
    if destination.exists():
        raise FileExistsError("existing continuation must be reviewed, never overwritten")
    state = dict(
        status="running",
        pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(),
        current_action="waiting for IC50 family",
        contract=contract,
        events=[],
    )
    write(destination, state)

    def mark(action):
        state["current_action"] = action
        state["events"].append({"utc": datetime.now(timezone.utc).isoformat(), "action": action})
        write(destination, state)
        print(action, flush=True)

    try:
        mark("waiting for IC50 family")
        while True:
            predecessor = read(Path(contract["predecessor"]))
            if predecessor["status"] == "completed":
                break
            if predecessor["status"] != "running":
                raise RuntimeError(
                    "IC50 continuation stopped; review required before neural launch"
                )
            elapsed = (
                datetime.now(timezone.utc) - datetime.fromisoformat(predecessor["started_utc"])
            ).total_seconds()
            if elapsed > 25 * 60 * 60:
                raise TimeoutError("predecessor exceeded the explicit waiting allowance")
            time.sleep(10)
        baseline = read(FAMILY / "results/baselines/independent-verification.json")
        if baseline["status"] != "passed":
            raise RuntimeError("fixed baseline verification must pass")
        for condition in ("seed42", "seed43", "seed44"):
            mark("fitting " + condition)
            execute(["bash", str(REPRO / "launch.sh"), condition], condition + ".log", 10890)
            mark("reporting " + condition)
            execute(
                ["bash", str(REPRO / "launch.sh"), "report"], "report-" + condition + ".log", 660
            )
            mark("verifying " + condition)
            execute(
                ["bash", str(REPRO / "verify-v2.sh"), condition],
                "verify-" + condition + ".log",
                660,
            )
            execute(
                ["bash", str(REPRO / "launch.sh"), "report"],
                "verified-report-" + condition + ".log",
                660,
            )
            execute(
                [contract["python"], str(FAMILY / "code/close.py")],
                "close-" + condition + ".log",
                120,
            )
        state.update(
            status="completed",
            current_action="fixed species family verified; scientific screens require review",
        )
    except BaseException as exc:
        state.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(destination, state)
        # Never restart a fit or rescore test on failure; publish saved evidence only.
        try:
            execute([contract["python"], str(FAMILY / "code/close.py")], "close-terminal.log", 120)
        except BaseException as exc:
            state.update(status="failed", closure_failure=repr(exc))
            write(destination, state)


if __name__ == "__main__":
    main()
