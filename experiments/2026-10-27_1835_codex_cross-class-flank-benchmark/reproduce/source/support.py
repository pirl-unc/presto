"""Benchmark bookkeeping and explicit unsupported-domain diagnostics; no training loop."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import json
import os
import resource
import shutil
import signal
import threading
import time

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
FULL = Path("/private/tmp/presto-unified-core-20261027/presto")
OLD = Path("/private/tmp/presto-shared-core-20261027/presto")
RUNTIMES = {"shared": ROOT, "full": FULL, "three_aa": OLD}
RAW = ROOT / "artifacts" / FAMILY.name
GIB = 2**30
LIMITS = dict(
    startup_ram=6 * GIB,
    ram_floor=3 * GIB,
    startup_disk=15 * GIB,
    disk_floor=10 * GIB,
    rss=4 * GIB,
    seconds=3600,
)
CONDITIONS = {
    f"{variant}-s{seed}": dict(seed=seed, classes=classes, runtime=runtime)
    for seed in (42, 43)
    for variant, classes, runtime in (
        ("shared-I", ["I"], "shared"),
        ("shared-II", ["II"], "shared"),
        ("shared-joint", ["I", "II"], "shared"),
        ("full-joint", ["I", "II"], "full"),
        ("three-aa-joint", ["I", "II"], "three_aa"),
    )
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


@contextmanager
def guarded(name):
    import psutil

    start = time.monotonic()
    process = psutil.Process()
    destination = FAMILY / "reproduce" / "receipts" / f"{name}.json"
    if destination.exists():
        raise ValueError(f"refusing to overwrite resource receipt: {name}")
    state = dict(
        status="starting",
        pid=os.getpid(),
        limits=LIMITS,
        startup_ram=psutil.virtual_memory().available,
        startup_disk=shutil.disk_usage(ROOT).free,
        peak_rss=process.memory_info().rss,
    )
    atomic(destination, state)
    done = threading.Event()
    violation = [None]

    def handler(_signum, _frame):
        raise RuntimeError("resource guard: " + str(violation[0]))

    def monitor():
        while not done.wait(0.25):
            try:
                state["peak_rss"] = max(state["peak_rss"], process.memory_info().rss)
                if state["peak_rss"] > LIMITS["rss"]:
                    violation[0] = "rss"
                elif psutil.virtual_memory().available < LIMITS["ram_floor"]:
                    violation[0] = "available_ram"
                elif shutil.disk_usage(ROOT).free < LIMITS["disk_floor"]:
                    violation[0] = "disk_reserve"
                elif time.monotonic() - start > LIMITS["seconds"]:
                    violation[0] = "wall_time"
            except Exception as exc:
                violation[0] = "monitor_error: " + repr(exc)
            if violation[0]:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    previous = signal.signal(signal.SIGUSR1, handler)
    thread = threading.Thread(target=monitor, daemon=True)
    try:
        if state["startup_ram"] < LIMITS["startup_ram"]:
            raise RuntimeError("startup RAM below frozen 6 GiB floor")
        if state["startup_disk"] < LIMITS["startup_disk"]:
            raise RuntimeError("startup disk below frozen 15 GiB floor")
        state["status"] = "running"
        atomic(destination, state)
        thread.start()
        yield state
        state["status"] = "passed"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        done.set()
        if thread.is_alive():
            thread.join(timeout=2)
        signal.signal(signal.SIGUSR1, previous)
        state.update(
            elapsed_seconds=time.monotonic() - start,
            peak_rss=max(state["peak_rss"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            resource_violation=violation[0],
        )
        atomic(destination, state)


def diagnostic_view(source):
    """Expand only routing; every tensor stays identical, with no target training."""
    import torch
    from presto.models import Presto

    config = replace(source.config, mhc_classes=("I", "II"))
    with torch.random.fork_rng():
        target = Presto(config)
    target.load_state_dict(source.state_dict(), strict=True)
    for name, before in source.state_dict().items():
        if not torch.equal(before, target.state_dict()[name]):
            raise ValueError("diagnostic view changed a tensor: " + name)
    for name in (
        "support",
        "trained_parameters",
        "trained_features",
        "exposed_features",
        "source_label_support",
        "frozen_targets",
        "frozen_parameters",
    ):
        setattr(target, name, copy.deepcopy(getattr(source, name)))
    target.eval()
    return target, dict(
        protocol="identical_weights_expanded_routing",
        new_parameters=[],
        all_weights_preserved=True,
        target_class_labels_used=0,
        capability_status="unsupported_class_transfer_diagnostic",
    )


def prediction_dump(path, records):
    import gzip

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            for record in records:
                stream.write(
                    (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
                )


def load_dump(path):
    import gzip

    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def instrument_steps(runner, destination):
    """Observe the canonical step without changing losses, optimizer, sampler or model."""
    original = runner.train_step
    started = time.monotonic()
    state = dict(steps=0, examples=0, optimizer_seconds=0.0)

    def observed(model, optimizer, rows, scales, **kwargs):
        before = time.monotonic()
        result = original(model, optimizer, rows, scales, **kwargs)
        state["optimizer_seconds"] += time.monotonic() - before
        state["steps"] += 1
        state["examples"] += len(rows)
        if state["steps"] == 1 or state["steps"] % 10 == 0:
            import psutil

            point = {
                **state,
                "elapsed_seconds": time.monotonic() - started,
                "latest_training_loss": result["loss"],
                "rss": psutil.Process().memory_info().rss,
                "updated_unix": time.time(),
            }
            atomic(destination, point)
            with Path(destination).with_suffix(".jsonl").open("a") as stream:
                stream.write(json.dumps(point, allow_nan=False) + "\n")
        return result

    runner.train_step = observed
    return original
