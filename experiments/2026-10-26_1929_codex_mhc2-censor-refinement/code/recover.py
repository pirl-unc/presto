"""Run the two frozen conditions serially; stop on any child failure."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

FAMILY = Path(__file__).resolve().parents[1]
STATE = FAMILY / "reproduce/recovery-state-attempt02.json"


def main():
    if STATE.exists():
        raise FileExistsError("existing controller state; inspect before any recovery")
    logs = FAMILY / "recovery_logs"
    logs.mkdir(exist_ok=False)
    state = {"status": "running", "events": []}
    commands = [
        ("resume-censor", [sys.executable, str(FAMILY / "code/resume_censor.py")]),
        (
            "score-censor",
            [sys.executable, str(FAMILY / "code/launch.py"), "score", "censor_refinement"],
        ),
        ("verify-comparison", [sys.executable, str(FAMILY / "code/verify.py")]),
    ]
    try:
        for name, command in commands:
            state["current_action"] = name
            state["events"].append(
                {"utc": datetime.now(timezone.utc).isoformat(), "action": name, "argv": command}
            )
            STATE.write_text(json.dumps(state, indent=2) + "\n")
            with (logs / (name + ".log")).open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                state["pid"] = process.pid
                STATE.write_text(json.dumps(state, indent=2) + "\n")
                code = process.wait()
            if code:
                raise RuntimeError(f"{name} exited {code}; no next condition launched")
        state["status"] = "all_conditions_verified"
    except BaseException as error:
        state.update(status="stopped_on_failure", error=repr(error))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        STATE.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
