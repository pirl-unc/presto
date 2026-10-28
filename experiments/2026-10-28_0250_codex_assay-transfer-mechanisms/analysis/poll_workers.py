"""Read-only progress polling independent of the original archive collector."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import modal

F = Path(__file__).resolve().parents[1]
NAMES = tuple(json.loads((F / "reproduce/source-pins.json").read_text())["conditions"])


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    receipt = F / "reproduce/worker-poller.json"
    if receipt.exists():
        raise ValueError("poller already attempted; inspect existing process/receipt")
    atomic(
        receipt,
        dict(
            argv=sys.argv,
            executable=sys.executable,
            started_unix=time.time(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            interval_seconds=10,
            maximum_concurrent_reads=4,
            scope="Read existing family Dict keys only; no fit calls, GPU allocations, or artifact writes on Modal.",
        ),
    )
    live = modal.Dict.from_name("presto-cross-class-live")
    states = {}
    previous = None
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            errors = {}
            futures = {
                pool.submit(live.get, f"{F.name}:{name}", None): name for name in NAMES
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    state = future.result()
                    if state is not None:
                        states[name] = state
                except Exception as exc:
                    errors[name] = repr(exc)
            controller = read(F / "reproduce/controller.json", {})
            for failure in controller.get("failed", []):
                states[failure["condition"]] = dict(
                    condition=failure["condition"],
                    status=dict(status="failed", error=failure["error"]),
                )
            terminal = [
                name
                for name in NAMES
                if states.get(name, {}).get("status", {}).get("status")
                in ("completed", "failed")
            ]
            atomic(
                F / "results/live-workers.json",
                dict(
                    conditions=states,
                    errors=errors,
                    updated_unix=time.time(),
                    terminal=terminal,
                    all_terminal=len(terminal) == len(NAMES),
                ),
            )
            current = {
                name: states.get(name, {}).get("status", {}).get("status", "queued")
                for name in NAMES
            }
            if current != previous or errors:
                print(json.dumps(dict(status=current, errors=errors)), flush=True)
                previous = current
            if len(terminal) == len(NAMES):
                print(
                    "All worker calls terminal; original controller still owns archive collection.",
                    flush=True,
                )
                return
            time.sleep(10)


if __name__ == "__main__":
    main()
