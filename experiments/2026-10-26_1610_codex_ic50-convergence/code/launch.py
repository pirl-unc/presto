"""Frozen local convergence conditions; no model or recipe search."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
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

import numpy as np
import psutil
import torch

from presto.data.modular import AA, observation_target
from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training.modular_evaluation import summarize
from presto.training.modular_resume import SOURCE_FILES, atomic_json
from presto.training.modular_run import fit_index, preflight

ROOT = Path(__file__).resolve().parents[3]
FAMILY = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("PRESTO_IC50_ARTIFACT_ROOT", ROOT / "artifacts" / FAMILY.name)).resolve()
ORIGINAL_SHA = "914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be"
CONDITIONS = {"seed42": 42, "seed43": 43, "seed44": 44, "permuted42": 42}
GIB = 1024**3
LIMITS = dict(
    startup_ram=10 * GIB,
    ram_floor=5 * GIB,
    startup_disk=15 * GIB,
    disk_floor=10 * GIB,
    rss=6 * GIB,
    seconds=6 * 60 * 60,
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
                "PRESTO_IC50_ARTIFACT_ROOT",
            )
        },
        "runtime_hashes": {p: file_sha256(ROOT / p) for p in SOURCE_FILES},
        "launcher_sha256": file_sha256(__file__),
        "limits": LIMITS,
    }


@contextmanager
def guarded(name):
    """Monitor this single-process fit; BLAS/PyTorch use threads, not child trainers."""
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


def prepare(source_manifest):
    source_manifest = Path(source_manifest).resolve()
    if file_sha256(source_manifest) != ORIGINAL_SHA:
        raise ValueError("original qualified manifest differs")
    if (FAMILY / "reproduce/launch.json").exists():
        raise ValueError("preparation already exists; preserve it")
    original = json.loads(source_manifest.read_text())
    for source in original["sources"]:
        source["path"] = str((source_manifest.parent / source["path"]).resolve())
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError("qualified observation source changed")
    RAW.mkdir(parents=True, exist_ok=True)
    # A new control file is explicit; the measured source bytes stay untouched.
    train_source = next(s for s in original["sources"] if "train-exact" in s["path"])
    train_rows = [json.loads(line) for line in Path(train_source["path"]).read_text().splitlines()]
    rng = np.random.default_rng(424242)
    assignment = []
    for mhc_class in ("I", "II"):
        positions = [i for i, r in enumerate(train_rows) if r["mhc"][0]["mhc_class"] == mhc_class]
        donors = rng.permutation(positions)
        values = [train_rows[int(i)]["value"] for i in donors]
        for index, donor, value in zip(positions, donors, values, strict=True):
            row = train_rows[index]
            assignment.append(
                {
                    "observation_id": row["observation_id"],
                    "donor_observation_id": train_rows[int(donor)]["observation_id"],
                    "class": mhc_class,
                    "original_value": row["value"],
                    "value": value,
                }
            )
            row["provenance"]["label_permutation_control"] = assignment[-1]
            row["value"] = value
    permutation = RAW / "permuted-train.jsonl"
    with permutation.open("x") as stream:
        for row in train_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write(FAMILY / "reproduce/permutation-assignments.json", assignment)
    manifests = {}
    for name, seed in CONDITIONS.items():
        manifest = json.loads(json.dumps(original))
        manifest["training"].update(
            seed=seed,
            batch_size=64,
            max_epochs=100,
            min_epochs=10,
            patience=15,
            learning_rate=3e-4,
            weight_decay=1e-4,
            identity_weight=0,
            checkpoint_every_steps=50,
        )
        if name == "permuted42":
            source = next(s for s in manifest["sources"] if "train-exact" in s["path"])
            source.update(
                path=str(permutation),
                sha256=file_sha256(permutation),
                label_policy="within-class permutation control; original values in provenance",
            )
        path = FAMILY / "reproduce" / f"{name}-manifest.json"
        write(path, manifest)
        manifests[name] = {"path": str(path), "sha256": file_sha256(path)}
    for filename in (*SOURCE_FILES, str(Path(__file__).relative_to(ROOT))):
        target = FAMILY / "reproduce/source" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, target)
    launch = receipt()
    launch.update(
        original_manifest_path=str(source_manifest),
        original_manifest_sha256=ORIGINAL_SHA,
        manifests=manifests,
        raw_artifacts=str(RAW),
    )
    write(FAMILY / "reproduce/launch.json", launch)
    print(json.dumps({"prepared": list(manifests), "raw": str(RAW)}), flush=True)


def manifest_index(name):
    launch = json.loads((FAMILY / "reproduce/launch.json").read_text())
    for path, expected in launch["runtime_hashes"].items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError("runtime source changed: " + path)
    pin = launch["manifests"][name]
    if file_sha256(pin["path"]) != pin["sha256"]:
        raise ValueError("condition manifest changed")
    return ObservationIndex(pin["path"])


def features(rows):
    return np.asarray(
        [
            [
                len(r.peptide),
                *[r.peptide.count(a) / len(r.peptide) for a in AA],
                float(r.mhc[0].mhc_class == "I"),
                float(r.mhc[0].mhc_class == "II"),
            ]
            for r in rows
        ],
        dtype=float,
    )


def baselines():
    index = manifest_index("seed42")
    _, _, scales, admission = preflight(index)
    output = FAMILY / "results/baselines"
    output.mkdir(exist_ok=False)
    train = [index[i] for i in index.partitions["train:assay"]]
    y = np.asarray([observation_target(r) for r in train])
    x = features(train)
    center, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = np.column_stack([np.ones(len(x)), (x - center) / scale])
    penalty = np.eye(z.shape[1])
    penalty[0, 0] = 0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    write(
        output / "coefficients.json",
        {
            "mean": float(y.mean()),
            "ridge": coef.tolist(),
            "feature_mean": center.tolist(),
            "feature_sd": scale.tolist(),
            "alpha": 1,
        },
    )
    write(output / "admission.json", admission)
    metrics = {}
    for split in ("train", "validation", "test"):
        rows = [index[i] for i in index.partitions[f"{split}:assay"]]
        matrix = np.column_stack([np.ones(len(rows)), (features(rows) - center) / scale])
        for method, predictions in (
            ("mean", np.full(len(rows), y.mean())),
            ("ridge", matrix @ coef),
        ):
            records = []
            for row, prediction in zip(rows, predictions, strict=True):
                target = observation_target(row)
                diff = prediction - target
                if row.qualifier in ("<", "<="):
                    diff = max(diff, 0)
                elif row.qualifier in (">", ">="):
                    diff = min(diff, 0)
                records.append(
                    dict(
                        observation_id=row.observation_id,
                        endpoint=row.endpoint,
                        source=row.source,
                        group_id=row.group_id,
                        split=split,
                        peptide=row.peptide,
                        qualifier=row.qualifier,
                        mhc_classes=[m.mhc_class for m in row.mhc],
                        target=target,
                        prediction=float(prediction),
                        raw_value=row.value,
                        raw_unit=row.unit,
                        target_space="log10(nM)",
                        observation=row.to_dict(),
                        loss=float((diff / scales["ic50"].scale) ** 2),
                    )
                )
            with gzip.open(output / f"{method}-{split}-predictions.jsonl.gz", "wt") as stream:
                for record in records:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            metrics.setdefault(method, {})[split] = summarize(records)
    write(output / "metrics.json", metrics)
    print("Fixed mean/ridge baselines saved", flush=True)


def fit(name, resume):
    index = manifest_index(name)
    output = RAW / name
    summary = fit_index(index, output, device="cpu", resume=resume)
    target = FAMILY / "results" / name
    target.mkdir(exist_ok=False)
    for filename in (
        "summary.json",
        "coverage.json",
        "history.json",
        "admission.json",
        "resolved-config.json",
    ):
        shutil.copyfile(output / filename, target / filename)
    for path in output.glob("*-predictions.jsonl"):
        with (
            path.open("rb") as source,
            gzip.open(target / (path.name + ".gz"), "wb") as destination,
        ):
            shutil.copyfileobj(source, destination)
    write(
        target / "artifacts.json",
        {
            str(path.relative_to(output)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != ".training.lock"
        },
    )
    print(
        json.dumps(
            {
                "condition": name,
                "selected_epoch": summary["selected_epoch"],
                "coverage": summary["coverage_gate"]["status"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "baselines", *CONDITIONS))
    parser.add_argument("--source-manifest")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.resume and args.action not in CONDITIONS:
        parser.error("resume is only valid for a neural condition")
    if args.action == "prepare" and not args.source_manifest:
        parser.error("prepare requires --source-manifest")
    with guarded(args.action):
        if args.action == "prepare":
            prepare(args.source_manifest)
        elif args.action == "baselines":
            baselines()
        else:
            fit(args.action, args.resume)


if __name__ == "__main__":
    main()
