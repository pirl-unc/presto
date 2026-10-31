#!/usr/bin/env python
"""Local hardware diagnostic using Presto's unchanged canonical IC50 trainer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

from presto.data.modular import AA, observation_target
from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.models.modular import ModularPresto
from presto.training.modular import save_checkpoint, train_step
from presto.training.modular_evaluation import summarize
from presto.training.modular_resume import SOURCE_FILES
from presto.training.modular_run import fit_index, preflight, _accumulate_coverage

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "experiments/2026-10-22_1745_codex_local-ic50-feasibility"
RAW = ROOT / "artifacts" / FAMILY.name
MANIFEST = ROOT / "data/modular_cohorts/2026-10-21-full-ic50/manifest-exact.json"
MANIFEST_HASH = "914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def memory(device):
    result = {
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "current_process_rss_bytes": psutil.Process().memory_info().rss,
        "system_available_bytes": psutil.virtual_memory().available,
    }
    if device == "mps":
        result.update(
            mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
            mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        )
    return result


def register(args):
    if file_sha256(MANIFEST) != MANIFEST_HASH:
        raise ValueError("admitted manifest changed")
    key = f"{args.action}-{args.device}"
    receipt_path = FAMILY / f"reproduce/{key}.json"
    if receipt_path.exists():
        raise ValueError("refusing to overwrite execution receipt")
    receipt = {
        "agent_model": "Codex / GPT-6",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "executable": sys.executable,
        "cwd": str(ROOT),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": psutil.virtual_memory().total,
        "device": args.device,
        "mps_available": torch.backends.mps.is_available(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "environment": {k: os.environ.get(k) for k in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH")},
        "manifest": {"path": str(MANIFEST), "sha256": MANIFEST_HASH},
        "source_hashes": {},
        "memory": memory(args.device),
    }
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = FAMILY / "reproduce/source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) != file_sha256(source):
            raise ValueError("implementation changed within diagnostic")
        shutil.copy2(source, target)
        receipt["source_hashes"][relative] = file_sha256(source)
    shutil.copy2(__file__, FAMILY / "reproduce/source/launch.py")
    dump(receipt_path, receipt)
    if not (FAMILY / "reproduce/launch.json").exists():
        dump(FAMILY / "reproduce/launch.json", receipt)
    return receipt


def benchmark(device):
    output = RAW / f"benchmark-{device}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    index = ObservationIndex(MANIFEST)
    config, settings, scales, admission = preflight(index)
    torch.manual_seed(42)
    model = ModularPresto(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    order = list(index.partitions["train:assay"])
    random.Random(42).shuffle(order)
    save_checkpoint(output / "initial.pt", model, scales, optimizer=optimizer)
    measurements, coverage = [], {}
    for step in range(12):
        rows = [index[i] for i in order[step * 64:(step + 1) * 64]]
        sync(device)
        tick = time.monotonic()
        result = train_step(model, optimizer, rows, scales, identity_weight=0)
        sync(device)
        seconds = time.monotonic() - tick
        _accumulate_coverage(coverage, result)
        record = {"step": step + 1, "warmup": step < 4, "seconds": seconds,
                  "loss": result["loss"], "n": len(rows),
                  "observation_ids": [r.observation_id for r in rows], "memory": memory(device)}
        measurements.append(record)
        dump(output / "steps.json", measurements)
        print(f"{device} step={step + 1} seconds={seconds:.3f} loss={result['loss']:.6f}", flush=True)
    save_checkpoint(output / "last.pt", model, scales, optimizer=optimizer)
    median = statistics.median(r["seconds"] for r in measurements if not r["warmup"])
    result = {"device": device, "median_measured_step_seconds": median,
              "training_only_epoch_estimate_seconds": median * 339,
              "elapsed_seconds": time.monotonic() - started, "memory": memory(device),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "parameter_tensors": len(list(model.parameters())), "steps": measurements,
              "coverage": coverage, "admission": admission, "raw_output": str(output)}
    dump(FAMILY / f"results/benchmark-{device}.json", result)


def features(rows):
    return np.asarray([[len(r.peptide), *[r.peptide.count(a) / len(r.peptide) for a in AA],
                        float(r.mhc[0].mhc_class == "I"), float(r.mhc[0].mhc_class == "II")]
                       for r in rows], dtype=float)


def baselines(index, scales):
    output = RAW / "baselines"
    output.mkdir(parents=True, exist_ok=False)
    train = [index[i] for i in index.partitions["train:assay"]]
    y = np.asarray([observation_target(r) for r in train])
    x = features(train)
    center, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = np.column_stack([np.ones(len(x)), (x - center) / scale])
    penalty = np.eye(z.shape[1])
    penalty[0, 0] = 0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    dump(output / "coefficients.json", {"mean": float(y.mean()), "ridge": coef.tolist(),
         "feature_mean": center.tolist(), "feature_sd": scale.tolist(), "alpha": 1})
    result = {}
    for split in ("train", "validation", "test"):
        rows = [index[i] for i in index.partitions[f"{split}:assay"]]
        matrix = np.column_stack([np.ones(len(rows)), (features(rows) - center) / scale])
        for method, predictions in (("mean", np.full(len(rows), y.mean())), ("ridge", matrix @ coef)):
            records = []
            for r, prediction in zip(rows, predictions, strict=True):
                target = observation_target(r)
                difference = prediction - target
                if r.qualifier in ("<", "<="):
                    difference = max(difference, 0)
                elif r.qualifier in (">", ">="):
                    difference = min(difference, 0)
                records.append({"observation_id": r.observation_id, "endpoint": r.endpoint,
                    "source": r.source, "group_id": r.group_id, "split": r.split,
                    "peptide": r.peptide, "qualifier": r.qualifier,
                    "mhc_classes": [m.mhc_class for m in r.mhc], "target": target,
                    "prediction": float(prediction), "raw_value": r.value, "raw_unit": r.unit,
                    "loss": float((difference / scales["ic50"].scale) ** 2)})
            path = output / f"{method}-{split}-predictions.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                for r in records:
                    stream.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
            result.setdefault(method, {})[split] = summarize(records)
    dump(FAMILY / "results/baseline-metrics.json", result)
    shutil.copy2(output / "coefficients.json", FAMILY / "results/baseline-coefficients.json")


def fit(device):
    raw = json.loads(MANIFEST.read_text())
    for source in raw["sources"]:
        source["path"] = str((MANIFEST.parent / source["path"]).resolve())
    raw["training"].update(seed=42, batch_size=64, max_epochs=1, min_epochs=1,
        patience=1, learning_rate=3e-4, weight_decay=1e-4, identity_weight=0,
        checkpoint_every_steps=50)
    manifest_path = FAMILY / "reproduce/local-manifest.json"
    dump(manifest_path, raw)
    index = ObservationIndex(manifest_path)
    _, _, scales, _ = preflight(index)
    started = time.monotonic()
    baselines(index, scales)
    baseline_seconds = time.monotonic() - started
    output = RAW / f"one-epoch-{device}"
    summary = fit_index(index, output, device=device)
    sync(device)
    for name in ("summary.json", "coverage.json", "history.json", "admission.json", "resolved-config.json"):
        shutil.copy2(output / name, FAMILY / "results" / name)
    for path in output.glob("*-predictions.jsonl"):
        with path.open("rb") as source, gzip.open(FAMILY / "results" / (path.name + ".gz"), "wb") as target:
            shutil.copyfileobj(source, target)
    dump(FAMILY / "results/execution.json", {
        "device": device, "wall_seconds_with_baselines": time.monotonic() - started,
        "baseline_seconds": baseline_seconds, "memory": memory(device),
        "raw_output": str(output), "summary": summary,
        "artifacts": {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                      for p in output.iterdir() if p.is_file() and not p.name.startswith(".")},
    })
    for path in (RAW / "baselines").glob("*-predictions.jsonl.gz"):
        shutil.copy2(path, FAMILY / "results" / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("benchmark", "fit"))
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process; use approved local GPU access")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ValueError("silent CPU fallback is outside this hardware comparison")
    register(args)
    {"benchmark": benchmark, "fit": fit}[args.action](args.device)


if __name__ == "__main__":
    main()
