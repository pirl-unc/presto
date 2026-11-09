"""One explicitly reviewed continuation after RAM admission failure; no numerical changes."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import launch
from support import FAMILY, CONDITIONS, RUNTIMES, ROOT, GIB, atomic, read, sha, guarded


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "profile-worker":
        launch.configure()
        with guarded("profile-three_aa-after-stable-ram"):
            launch.profile("three_aa")
        return
    import psutil

    path = FAMILY / "reproduce" / "controller.json"
    previous = read(path)
    assert previous["status"] == "failed" and previous["completed"] == []
    archived = FAMILY / "reproduce" / "controller-attempt1.json"
    if archived.exists():
        raise ValueError("continuation already attempted")
    shutil.copy2(path, archived)
    invocation = dict(
        agent_model="Codex / GPT-6",
        argv=[sys.executable, str(Path(__file__).resolve())],
        cwd=str(ROOT),
        source_sha256=sha(__file__),
        original_controller_sha256=sha(archived),
        reason="first profile admission failed before any optimizer update",
        minimum_available_gib=8,
        stable_seconds=60,
        wait_limit_seconds=1200,
        original_launch=str(FAMILY / "reproduce" / "launch.json"),
    )
    atomic(FAMILY / "reproduce" / "continuation-launch.json", invocation)
    shutil.copy2(__file__, FAMILY / "reproduce" / "source" / "continue_after_ram.py")
    started = time.monotonic()
    since = None
    done = []
    samples = FAMILY / "reproduce" / "ram-admission.jsonl"
    try:
        while True:
            now = time.monotonic()
            available = psutil.virtual_memory().available
            since = since or now if available >= 8 * GIB else None
            stable = now - since if since is not None else 0
            sample = dict(
                unix=time.time(), available_gib=available / GIB, stable_seconds=stable
            )
            with samples.open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            atomic(path, dict(status="waiting_for_stable_RAM", completed=[], **sample))
            if stable >= 60:
                break
            if now - started >= 1200:
                raise RuntimeError("stable RAM admission timed out; zero new launches")
            time.sleep(5)
        runtime = RUNTIMES["three_aa"]
        env = {
            **os.environ,
            "PYTHONPATH": str(runtime.parent),
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        }
        atomic(path, dict(status="profiling", active="three_aa", completed=[]))
        with (FAMILY / "launch_logs" / "profile-three_aa-after-stable-ram.log").open(
            "x"
        ) as stream:
            subprocess.run(
                [sys.executable, __file__, "profile-worker"],
                env=env,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        for label in ("full", "shared"):
            atomic(path, dict(status="profiling", active=label, completed=[]))
            launch.child("profile", label)
        for condition, spec in CONDITIONS.items():
            atomic(path, dict(status="running", active=condition, completed=done))
            launch.child("fit", spec["runtime"], condition)
            done.append(condition)
        atomic(path, dict(status="fits_completed", completed=done, finished_unix=time.time()))
    except BaseException as exc:
        atomic(path, dict(status="failed", completed=done, error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
