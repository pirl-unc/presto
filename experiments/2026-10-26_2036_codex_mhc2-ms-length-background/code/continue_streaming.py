"""Run the registered serialization recovery once; never retry automatically."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from presto.data.modular_sources import file_sha256

FAMILY = Path(__file__).resolve().parents[1]
STATE = FAMILY / "reproduce/streaming-controller-state.json"


def main():
    if STATE.exists():
        raise FileExistsError("existing recovery state; inspect before any recovery")
    pin = json.loads((FAMILY / "reproduce/streaming-recovery-launch.json").read_text())
    logs = FAMILY / "streaming_logs"
    logs.mkdir(exist_ok=False)
    state = {"status": "running", "events": []}
    commands = [
        ("recover-score-native101", ["streaming_recovery.py", "score", "native101"]),
        ("fit-native102", ["streaming_recovery.py", "fit", "native102"]),
        ("score-native102", ["streaming_recovery.py", "score", "native102"]),
        ("fit-permuted101", ["streaming_recovery.py", "fit", "permuted101"]),
        ("score-permuted101", ["streaming_recovery.py", "score", "permuted101"]),
        ("logistic-features", ["logistic_controls_streaming.py", "features"]),
        ("logistic-fit", ["logistic_controls_streaming.py", "fit"]),
    ]
    try:
        for name, arguments in commands:
            for filename, expected in pin["code_hashes"].items():
                assert file_sha256(FAMILY / "code" / filename) == expected
            command = [sys.executable, str(FAMILY / "code" / arguments[0]), *arguments[1:]]
            state["current_action"] = name
            state["events"].append(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "argv": command,
                }
            )
            STATE.write_text(json.dumps(state, indent=2) + "\n")
            with (logs / (name + ".log")).open("x") as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                state["pid"] = process.pid
                STATE.write_text(json.dumps(state, indent=2) + "\n")
                result = process.wait()
            if result:
                raise RuntimeError(f"{name} exited{result}; no next action or automatic retry")
        state["status"] = "all_arms_scored_independent_comparison_pending"
    except BaseException as error:
        state.update(status="stopped_on_failure", error=repr(error))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        STATE.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
