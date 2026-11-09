"""Own the terminal transport handoff only; never launch or retry training."""

from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

F = Path(__file__).resolve().parents[1]
PID = 25349
EXPECTED = (
    "Tue Oct 27 23:21:16 2026     "
    "/Users/iskander/code/shared-virtual-env/bin/python "
    "/Users/iskander/code/shared-virtual-env/bin/modal run --detach "
    "experiments/2026-10-28_0250_codex_assay-transfer-mechanisms/code/launch.py"
)


def identity():
    result = subprocess.run(
        ["ps", "-p", str(PID), "-o", "lstart=,command="],
        text=True,
        capture_output=True,
    )
    if (
        result.returncode == 1
        and not result.stdout.strip()
        and not result.stderr.strip()
    ):
        return None
    assert result.returncode == 0 and not result.stderr.strip(), result.stderr
    return result.stdout.strip()


def read(relative):
    return json.loads((F / relative).read_text())


def main():
    assert identity() == EXPECTED, "original collector identity differs"
    handoff = F / "analysis/collection_handoff.py"
    handoff_hash = hashlib.sha256(handoff.read_bytes()).hexdigest()
    receipt = F / "reproduce/handoff-watcher.json"
    assert not receipt.exists()
    receipt.write_text(
        json.dumps(
            dict(
                argv=sys.argv,
                executable=sys.executable,
                started_unix=time.time(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                handoff_sha256=handoff_hash,
                collector_pid=PID,
                collector_identity=EXPECTED,
                scope="Normal termination of this exact owned collector only after every original fit call returns and all original archive hashes pass; then original collector with local transport. No training or duplicate closure.",
            ),
            indent=2,
        )
        + "\n"
    )
    while True:
        controller = read("reproduce/controller.json")
        if (
            controller.get("status") == "completed"
            and len(controller["completed"]) == 12
        ):
            print(
                "Original collection completed independently; no handoff.", flush=True
            )
            return
        workers = read("results/live-workers.json")
        cache = read("results/artifact-readback-state.json")
        if workers["all_terminal"] and cache["status"] == "completed":
            break
        time.sleep(15)
    assert hashlib.sha256(handoff.read_bytes()).hexdigest() == handoff_hash
    command = [sys.executable, str(handoff)]
    subprocess.run(command + ["--prepare"], check=True)
    controller = read("reproduce/controller.json")
    if controller.get("status") == "completed" and len(controller["completed"]) == 12:
        print("Original collection completed during preparation; no handoff.", flush=True)
        return
    current = identity()
    if current is not None:
        assert current == EXPECTED, "PID identity changed; refuse termination"
        print(
            "All original calls returned; terminating only the verified collector.",
            flush=True,
        )
        os.kill(PID, signal.SIGTERM)
        for _ in range(60):
            if identity() is None:
                break
            time.sleep(1)
        else:
            raise RuntimeError("collector did not exit; no second writer started")
    subprocess.run(
        command + ["--finish", "--former-controller-pid", str(PID)], check=True
    )
    print("Handoff complete; existing closure watcher owns verification.", flush=True)


if __name__ == "__main__":
    main()
