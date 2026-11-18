"""Explicit remaining arms after zero-update preflight failure; no action retry."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

FAMILY = Path(__file__).resolve().parents[1]
STATE = FAMILY / "reproduce/remaining-controller-state.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert not STATE.exists(), "inspect existing state; no automatic recovery"
    pin = json.loads((FAMILY / "reproduce/remaining-launch.json").read_text())
    assert sha(__file__) == pin["controller_sha256"]
    logs = FAMILY / "remaining_logs"
    logs.mkdir(exist_ok=False)
    state = {"status": "running", "events": [], "admission_samples": []}
    commands = [
        ("fit-native102", ["streaming_recovery.py", "fit", "native102"]),
        ("score-native102", ["streaming_recovery.py", "score", "native102"]),
        ("fit-permuted101", ["streaming_recovery.py", "fit", "permuted101"]),
        ("score-permuted101", ["streaming_recovery.py", "score", "permuted101"]),
        ("logistic-features", ["logistic_controls_streaming.py", "features"]),
        ("logistic-fit", ["logistic_controls_streaming.py", "fit"]),
    ]

    def save():
        STATE.write_text(json.dumps(state, indent=2) + "\n")

    try:
        for name, arguments in commands:
            state["current_action"] = "admission:" + name
            started, sufficient_since = time.monotonic(), None
            while True:
                now = time.monotonic()
                available = psutil.virtual_memory().available
                if available >= 20 * 1024**3:
                    sufficient_since = now if sufficient_since is None else sufficient_since
                else:
                    sufficient_since = None
                state["admission_samples"].append(
                    {
                        "action": name,
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": now - started,
                        "available_ram_bytes": available,
                        "sufficient_seconds": 0
                        if sufficient_since is None
                        else now - sufficient_since,
                    }
                )
                save()
                if sufficient_since is not None and now - sufficient_since >= 60:
                    break
                if now - started > 1800:
                    raise RuntimeError("admission deadline; no action launched or retried")
                time.sleep(5)
            for filename, expected in pin["code_hashes"].items():
                assert sha(FAMILY / "code" / filename) == expected
            command = [sys.executable, str(FAMILY / "code" / arguments[0]), *arguments[1:]]
            state["current_action"] = name
            state["events"].append({"utc": datetime.now(timezone.utc).isoformat(), "argv": command})
            save()
            with (logs / (name + ".log")).open("x") as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                state["pid"] = process.pid
                save()
                result = process.wait()
            if result:
                raise RuntimeError(f"{name} exited {result}; no next action or automatic retry")
        state["status"] = "all_arms_scored_independent_comparison_pending"
    except BaseException as error:
        state.update(status="stopped_on_failure", error=repr(error))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
