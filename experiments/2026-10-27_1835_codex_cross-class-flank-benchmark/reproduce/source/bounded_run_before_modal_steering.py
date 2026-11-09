"""Prospectively revised resource admission; frozen models and numerical experiment."""

from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import launch
from support import CONDITIONS, FAMILY, ROOT, RUNTIMES, LIMITS, GIB, atomic, read, sha, guarded


def worker(action, label, condition):
    import psutil

    launch.configure()
    LIMITS.update(startup_ram=int(5.5 * GIB), rss=2 * GIB)
    name = "bounded-" + (condition or f"profile-{label}")
    start = time.monotonic()
    since = None
    while True:
        now = time.monotonic()
        available = psutil.virtual_memory().available
        since = (since or now) if available >= LIMITS["startup_ram"] else None
        if since is not None and now - since >= 15:
            break
        if now - start >= 300:
            raise RuntimeError("bounded admission timed out before worker start")
        time.sleep(1)
    with guarded(name):
        if action == "profile":
            launch.profile(label)
        else:
            launch.fit(condition)


def controller():
    state = FAMILY / "reproduce" / "controller.json"
    previous = read(state)
    assert previous["status"] == "failed" and previous["completed"] == []
    archive = FAMILY / "reproduce" / "controller-attempt2-interrupted.json"
    if archive.exists():
        raise ValueError("bounded continuation already attempted")
    shutil.copy2(state, archive)
    source = Path(__file__).resolve()
    frozen = FAMILY / "reproduce" / "source" / source.name
    shutil.copy2(source, frozen)
    override = dict(startup_ram=int(5.5 * GIB), rss=2 * GIB)
    atomic(
        FAMILY / "reproduce" / "bounded-launch.json",
        dict(
            agent_model="Codex / GPT-6",
            argv=[sys.executable, str(source)],
            cwd=str(ROOT),
            source_sha256=sha(source),
            limits={**LIMITS, **override},
            original_launch=str(FAMILY / "reproduce" / "launch.json"),
            original_runtime_pins_sha256=sha(FAMILY / "reproduce" / "runtime-pins.json"),
            reason="explicit bounded profiling-first resource revision before optimizer updates",
        ),
    )
    done = []
    try:
        for action, label, condition in [
            ("profile", label, None) for label in ("three_aa", "full", "shared")
        ] + [("fit", spec["runtime"], name) for name, spec in CONDITIONS.items()]:
            active = condition or f"profile-{label}"
            atomic(
                state,
                dict(
                    status="profiling" if action == "profile" else "running",
                    active=active,
                    completed=done,
                    resource_protocol="bounded-resource-revision.md",
                ),
            )
            runtime = RUNTIMES[label]
            env = {
                **os.environ,
                "PYTHONPATH": str(runtime.parent),
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            }
            command = [sys.executable, str(source), "--worker", action, "--runtime", label]
            if condition:
                command += ["--condition", condition]
            with (FAMILY / "launch_logs" / f"bounded-{active}.log").open("x") as stream:
                subprocess.run(
                    command,
                    env=env,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if condition:
                done.append(condition)
        atomic(state, dict(status="fits_completed", completed=done, finished_unix=time.time()))
    except BaseException as exc:
        atomic(
            state,
            dict(
                status="failed",
                completed=done,
                error=repr(exc),
                resource_protocol="bounded-resource-revision.md",
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("profile", "fit"))
    parser.add_argument("--runtime", choices=tuple(RUNTIMES))
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.runtime, args.condition)
    else:
        controller()


if __name__ == "__main__":
    main()
