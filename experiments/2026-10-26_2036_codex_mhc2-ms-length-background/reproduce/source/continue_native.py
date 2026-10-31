"""Run each prespecified native arm once, stopping on any failed action."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

FAMILY = Path(__file__).resolve().parents[1]
STATE = FAMILY / "reproduce/native-controller-state.json"


def main():
    if STATE.exists():
        raise FileExistsError("existing controller state; inspect before recovery")
    logs = FAMILY / "native_logs"
    logs.mkdir(exist_ok=False)
    state = {"status": "running", "events": []}
    try:
        for arm in ("native101", "native102", "permuted101"):
            for action in ("fit", "score"):
                name = action + "-" + arm
                command = [sys.executable, str(FAMILY / "code/native.py"), action, arm]
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
                    status = process.wait()
                if status:
                    raise RuntimeError(f"{name} exited{status}; stopped without retry")
        state["status"] = "native_arms_scored_comparison_verification_pending"
    except BaseException as error:
        state.update(status="stopped_on_failure", error=repr(error))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        STATE.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
