"""Wait for native fits, then run the frozen logistic controls serially."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from presto.data.modular_sources import file_sha256

FAMILY = Path(__file__).resolve().parents[1]
STATE = FAMILY / "reproduce/logistic-controller-state.json"
NATIVE = FAMILY / "reproduce/native-controller-state.json"


def main():
    if STATE.exists():
        raise FileExistsError("existing logistic controller; no automatic restart")
    pin = json.loads((FAMILY / "reproduce/logistic-launch.json").read_text())
    state = {"status": "waiting_for_native", "events": []}
    started = time.monotonic()
    logs = FAMILY / "logistic_logs"
    logs.mkdir(exist_ok=False)
    try:
        while True:
            if time.monotonic() - started > 14 * 3600:
                raise TimeoutError("native completion wait exceeded fourteen hours")
            native = json.loads(NATIVE.read_text())
            if native["status"] == "stopped_on_failure":
                raise RuntimeError("native controller failed; controls not launched")
            if native["status"] == "native_arms_scored_comparison_verification_pending":
                break
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            STATE.write_text(json.dumps(state, indent=2) + "\n")
            time.sleep(15)
        for action in ("features", "fit"):
            for name, expected in pin["code_sha256"].items():
                assert file_sha256(FAMILY / "code" / name) == expected
            command = [sys.executable, str(FAMILY / "code/logistic_controls.py"), action]
            state.update(status="running", current_action=action)
            state["events"].append(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "argv": command,
                }
            )
            STATE.write_text(json.dumps(state, indent=2) + "\n")
            with (logs / (action + ".log")).open("x") as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                state["pid"] = process.pid
                STATE.write_text(json.dumps(state, indent=2) + "\n")
                code = process.wait()
            if code:
                raise RuntimeError(f"logistic {action} exited{code}; stopped without retry")
        state["status"] = "controls_scored_independent_comparison_pending"
    except BaseException as error:
        state.update(status="stopped_on_failure", error=repr(error))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        STATE.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
