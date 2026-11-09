"""Frozen resource guards and receipts for the MHC species family."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch

from presto.data.modular_sources import file_sha256
from presto.training.modular_resume import SOURCE_FILES, atomic_json

ROOT = Path(__file__).resolve().parents[3]
FAMILY = Path(__file__).resolve().parents[1]
RAW = Path(
    os.environ.get("PRESTO_MHC_SPECIES_ARTIFACT_ROOT", ROOT / "artifacts" / FAMILY.name)
).resolve()
ORIGINAL_SHA = "4e4d4a5d2b3d47fc16ddf63ddeeaf6111836138661356ab6805945f13c7ef06c"
CONDITIONS = {"seed42": 42, "seed43": 43, "seed44": 44}
GIB = 1024**3
LIMITS = dict(
    startup_ram=10 * GIB,
    ram_floor=5 * GIB,
    startup_disk=15 * GIB,
    disk_floor=10 * GIB,
    rss=6 * GIB,
    seconds=3 * 60 * 60,
)


def write(path, value):
    atomic_json(path, value)


def receipt():
    return {
        "agent_model": "Codex / GPT-6",
        "utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "packages": {name: version(name) for name in ("scipy", "scikit-learn", "psutil")},
        "device": "cpu",
        "requested_modal_gpu": None,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PRESTO_MHC_SPECIES_ARTIFACT_ROOT",
            )
        },
        "runtime_hashes": {p: file_sha256(ROOT / p) for p in SOURCE_FILES},
        "guard_sha256": file_sha256(__file__),
        "code_hashes": {p.name: file_sha256(p) for p in sorted((FAMILY / "code").glob("*.py"))},
        "limits": LIMITS,
    }


@contextmanager
def guarded(name, seconds=600):
    """Monitor this single-process fit; BLAS/PyTorch use threads, not child trainers."""
    LIMITS["seconds"] = seconds
    destination = FAMILY / "reproduce" / f"{name}-{time.time_ns()}.json"
    state = receipt()
    started = time.monotonic()
    process = psutil.Process()
    state.update(
        startup_ram=psutil.virtual_memory().available,
        startup_disk=shutil.disk_usage(ROOT).free,
        status="starting",
    )
    write(destination, state)
    stop = threading.Event()
    peak = [process.memory_info().rss]
    violation = [None]

    def fail(_signum, _frame):
        raise RuntimeError("resource guard: " + str(violation[0]))

    def monitor():
        while not stop.wait(0.25):
            try:
                rss = process.memory_info().rss
                peak[0] = max(peak[0], rss)
                if rss > LIMITS["rss"]:
                    violation[0] = "rss"
                elif psutil.virtual_memory().available < LIMITS["ram_floor"]:
                    violation[0] = "available_ram"
                elif shutil.disk_usage(ROOT).free < LIMITS["disk_floor"]:
                    violation[0] = "disk_reserve"
                elif time.monotonic() - started > LIMITS["seconds"]:
                    violation[0] = "wall_time"
            except Exception as exc:
                violation[0] = "monitor_error: " + repr(exc)
            if violation[0]:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    old_handler = signal.signal(signal.SIGUSR1, fail)
    thread = threading.Thread(target=monitor, daemon=True)
    try:
        if state["startup_ram"] < LIMITS["startup_ram"]:
            raise RuntimeError("startup available RAM below registered floor")
        if state["startup_disk"] < LIMITS["startup_disk"]:
            raise RuntimeError("startup free disk below registered floor")
        thread.start()
        yield state
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        stop.set()
        if thread.ident is not None:
            thread.join()
        signal.signal(signal.SIGUSR1, old_handler)
        state.update(
            elapsed_seconds=time.monotonic() - started,
            peak_polled_rss_bytes=peak[0],
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource_violation=violation[0],
            final_disk_free=shutil.disk_usage(ROOT).free,
        )
        write(destination, state)
