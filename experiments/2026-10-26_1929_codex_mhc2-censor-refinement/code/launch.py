"""Fixed-update canonical affinity refinement; never change the older runners."""

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
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

import psutil
import torch

from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training.modular import load_checkpoint
from presto.training.modular_evaluation import evaluate
from presto.training.modular_resume import SOURCE_FILES, atomic_json
from presto.training.modular_run import fit_index, preflight

ROOT = Path(__file__).resolve().parents[3]
FAMILY = Path(__file__).resolve().parents[1]
RAW = Path("/private/tmp/presto-mhc2-20261026/censor-refinement")
COHORT = Path(
    "/Users/iskander/code/presto/.worktrees/presto-modular-model/"
    "data/modular_cohorts/2026-10-21-full-ic50"
)
PARENT = Path(
    "/Users/iskander/code/presto/artifacts/"
    "2026-10-26_1610_codex_ic50-convergence/seed42/selected.pt"
)
PARENT_SHA = "db2585e63489cb5aac8c7a32fd7e522bcbc51035846c05f8cd2794c4449e3dc5"
SOURCES = {
    "exact_continuation": (
        "manifest-exact.json",
        "914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be",
    ),
    "censor_refinement": (
        "manifest-with-censor.json",
        "7fd143e22082dc18c1b7332954f66608bb7b0840813e5e12d74b3c85806dfc7c",
    ),
}
TRAINING = {
    "seed": 101,
    "batch_size": 64,
    "max_epochs": 5,
    "max_steps": 1476,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "identity_weight": 0,
    "min_epochs": 5,
    "patience": 100,
    "checkpoint_every_steps": 50,
}
LIMITS = {
    "startup_ram": 13 * 1024**3,
    "ram_floor": 10 * 1024**3,
    "startup_disk": 15 * 1024**3,
    "disk_floor": 10 * 1024**3,
    "rss": 4 * 1024**3,
    "seconds": 7200,
}


def read(path):
    return json.loads(Path(path).read_text())


def runtime_hashes():
    return {path: file_sha256(ROOT / path) for path in SOURCE_FILES}


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
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "requested_modal_gpu": None,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "runtime_hashes": runtime_hashes(),
        "launcher_sha256": file_sha256(__file__),
        "limits": LIMITS,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTORCH_ENABLE_MPS_FALLBACK",
            )
        },
    }


@contextmanager
def guarded(action):
    path = FAMILY / "reproduce" / f"{action}-{time.time_ns()}.json"
    state = receipt()
    started = time.monotonic()
    process = psutil.Process()
    state.update(
        status="starting",
        startup_ram=psutil.virtual_memory().available,
        startup_disk=shutil.disk_usage(RAW.parent).free,
    )
    atomic_json(path, state)
    stop = threading.Event()
    peak = [process.memory_info().rss]
    violation = [None]

    def fail(_signum, _frame):
        raise RuntimeError("resource guard: " + str(violation[0]))

    def monitor():
        while not stop.wait(0.25):
            try:
                memory = process.memory_info().rss
                peak[0] = max(peak[0], memory)
                if memory > LIMITS["rss"]:
                    violation[0] = "rss"
                elif psutil.virtual_memory().available < LIMITS["ram_floor"]:
                    violation[0] = "available_ram"
                elif shutil.disk_usage(RAW.parent).free < LIMITS["disk_floor"]:
                    violation[0] = "disk_reserve"
                elif time.monotonic() - started > LIMITS["seconds"]:
                    violation[0] = "wall_time"
            except Exception as error:
                violation[0] = "monitor_error: " + repr(error)
            if violation[0]:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    old_handler = signal.signal(signal.SIGUSR1, fail)
    thread = threading.Thread(target=monitor, daemon=True)
    try:
        if (
            state["startup_ram"] < LIMITS["startup_ram"]
            or state["startup_disk"] < LIMITS["startup_disk"]
        ):
            raise RuntimeError("startup resource reserve is unavailable")
        thread.start()
        yield state
        assert runtime_hashes() == state["runtime_hashes"], "runtime changed during action"
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
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
        )
        atomic_json(path, state)


def tensor_hashes(payload):
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in payload["state_dict"].items()
    }


def prepare():
    destination = FAMILY / "reproduce/launch.json"
    if destination.exists():
        raise FileExistsError(destination)
    assert file_sha256(PARENT) == PARENT_SHA
    parent, scales, payload = load_checkpoint(PARENT)
    manifests = {}
    for condition, (name, expected) in SOURCES.items():
        source_path = COHORT / name
        assert file_sha256(source_path) == expected
        manifest = read(source_path)
        for source in manifest["sources"]:
            source["path"] = str((source_path.parent / source["path"]).resolve())
            assert file_sha256(source["path"]) == source["sha256"]
        manifest["training"] = TRAINING
        manifest["pretrained"] = {"path": str(PARENT), "sha256": PARENT_SHA, "freeze": []}
        path = FAMILY / "reproduce" / f"{condition}-manifest.json"
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, manifest)
        index = ObservationIndex(path)
        config, _, current_scales, admission = preflight(index)
        expected_counts = Counter(
            {
                "train:assay": 21636 if condition == "exact_continuation" else 23555,
                "validation:assay": 2925,
                "test:assay": 2803,
            }
        )
        assert (
            Counter({key: len(value) for key, value in index.partitions.items()}) == expected_counts
        )
        assert current_scales == scales
        assert config == parent.config
        manifests[condition] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "source_manifest_sha256": expected,
            "counts": dict(expected_counts),
            "model": asdict(config),
            "admission": admission,
        }
    source_dir = FAMILY / "reproduce/source"
    source_dir.mkdir(exist_ok=True)
    for path in (FAMILY / "code").glob("*.py"):
        shutil.copyfile(path, source_dir / path.name)
    reference = ROOT / "experiments/2026-10-26_1610_codex_ic50-convergence/code/verify.py"
    shutil.copyfile(reference, source_dir / "metric_reference.py")
    state = receipt()
    state.update(
        manifests=manifests,
        parent={
            "path": str(PARENT),
            "sha256": PARENT_SHA,
            "scales": {key: asdict(value) for key, value in scales.items()},
            "tensor_hashes": tensor_hashes(payload),
        },
        training=TRAINING,
        raw_artifacts=str(RAW),
        primary_checkpoint="last.pt at update 1476",
        secondary_checkpoint="canonical validation-selected checkpoint",
        metric_reference_sha256=file_sha256(reference),
        code_hashes={path.name: file_sha256(path) for path in (FAMILY / "code").glob("*.py")},
    )
    atomic_json(destination, state)
    del parent
    print(
        json.dumps(
            {
                "prepared": list(manifests),
                "updates_per_condition": 1476,
                "scaler": state["parent"]["scales"],
            }
        ),
        flush=True,
    )


def index_for(condition):
    launch = read(FAMILY / "reproduce/launch.json")
    assert launch["runtime_hashes"] == runtime_hashes()
    for name, expected in launch["code_hashes"].items():
        assert file_sha256(FAMILY / "code" / name) == expected, name
    pin = launch["manifests"][condition]
    assert file_sha256(pin["path"]) == pin["sha256"]
    assert file_sha256(PARENT) == PARENT_SHA
    return ObservationIndex(pin["path"])


def fit(condition):
    index = index_for(condition)
    result = fit_index(index, RAW / condition, device="cpu")
    last = torch.load(RAW / condition / "last.pt", map_location="cpu", weights_only=True)
    assert last["metadata"]["steps"] == TRAINING["max_steps"]
    assert result["coverage_gate"]["status"] == "passed"
    atomic_json(FAMILY / "results" / f"{condition}-fit.json", result)


def score(condition):
    index = index_for(condition)
    destination = RAW / condition / "primary"
    destination.mkdir(exist_ok=False)
    portable = FAMILY / "results" / condition
    portable.mkdir(exist_ok=False)
    model, scales, payload = load_checkpoint(RAW / condition / "last.pt", device="cpu")
    assert payload["metadata"]["steps"] == 1476
    metrics, artifacts = {}, {}
    for split in ("train", "validation", "test"):
        metrics[split], rows = evaluate(
            model, index, index.partitions[f"{split}:assay"], scales, batch_size=64
        )
        path = destination / f"{split}-predictions.jsonl"
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        compressed = portable / (path.name + ".gz")
        with path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target)
        artifacts[split] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "compressed": str(compressed),
            "compressed_sha256": file_sha256(compressed),
            "n": len(rows),
        }
        del rows
    result = {
        "condition": condition,
        "primary_checkpoint": str(RAW / condition / "last.pt"),
        "primary_checkpoint_sha256": file_sha256(RAW / condition / "last.pt"),
        "steps": 1476,
        "scales": {key: asdict(value) for key, value in scales.items()},
        "metrics": metrics,
        "artifacts": artifacts,
    }
    atomic_json(portable / "primary.json", result)
    print(
        json.dumps({"condition": condition, "fixed_final_scored": True, "steps": 1476}), flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "fit", "score"))
    parser.add_argument("condition", choices=tuple(SOURCES), nargs="?")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    RAW.mkdir(exist_ok=True)
    (FAMILY / "reproduce").mkdir(exist_ok=True)
    (FAMILY / "results").mkdir(exist_ok=True)
    with guarded(args.action + "-" + (args.condition or "family")):
        if args.action == "prepare":
            prepare()
        elif args.condition is None:
            parser.error("fit and score require a condition")
        elif args.action == "fit":
            fit(args.condition)
        else:
            score(args.condition)


if __name__ == "__main__":
    main()
