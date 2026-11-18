"""Durable state and evaluation receipts for the modular training loop."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

import numpy as np
import torch

from ..data.modular_sources import file_sha256
from .modular import save_checkpoint

RESUME_VERSION = "presto.modular.resume.v1"
SOURCE_FILES = (
    "data/modular.py",
    "data/affinity_matching.py",
    "data/assay_measurement.py",
    "data/measurement_curation.py",
    "data/modular_context.py",
    "data/modular_ordinal.py",
    "data/modular_quantitative.py",
    "data/modular_sources.py",
    "data/modular_lineage.py",
    "data/modular_preparation.py",
    "data/modular_auxiliary.py",
    "data/tcell_assay.py",
    "data/vocab.py",
    "models/modular_dependencies.py",
    "models/modular.py",
    "models/presto.py",
    "models/apm_state.py",
    "models/apm_state_bottleneck.py",
    "models/apc_state.py",
    "models/junction_processing.py",
    "models/assay_measurement.py",
    "models/tcell_assay.py",
    "cli/train.py",
    "training/trainer.py",
    "models/modular_support.py",
    "models/encoders.py",
    "training/modular.py",
    "training/modular_run.py",
    "training/modular_sampling.py",
    "training/modular_controls.py",
    "training/processing_pairs.py",
    "training/modular_resume.py",
    "training/modular_evaluation.py",
    "cli/training_sources.py",
    "scripts/train_iedb.py",
)


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def digest(value):
    result = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(value):
        result.update(chunk.encode())
    return result.hexdigest()


def atomic_json(path, value):
    """Persist large evaluation caches without allocating another full JSON copy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(raw_temp)
    encoder = json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            for chunk in encoder.iterencode(value):
                stream.write(chunk)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def run_lock(output):
    """A process exit releases the lock; stale lock files do not block resume."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+b") as stream:
        if fcntl is not None:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as error:
                raise ValueError("another trainer is active in this run directory") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        else:  # pragma: no cover - Windows
            import msvcrt

            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError("another trainer is active in this run directory") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def execution_contract(index, config, training, scales, device):
    root = Path(__file__).resolve().parent.parent
    backend = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "mha_fastpath": torch.backends.mha.get_fastpath_enabled(),
    }
    return {
        "format": RESUME_VERSION,
        "manifest_sha256": index.manifest_sha256,
        "source_hashes": {str(path): file_sha256(path) for path in index.source_paths},
        "config": json.loads(canonical(config.to_dict())),
        "training": training,
        "scales": {name: asdict(value) for name, value in scales.items()},
        "implementation": {name: file_sha256(root / name) for name in SOURCE_FILES},
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": torch.device(device).type,
        "backend": backend,
        "environment": {
            name: os.environ.get(name)
            for name in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_ENABLE_MPS_FALLBACK")
        },
    }


# Hardware and scheduling facts. A run preempted onto a different host must
# resume (as mhcflurry adapts threads, devices and batch size to the machine),
# so these are recorded for provenance and drift reporting, never enforced.
HOST_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)


def host_profile(device):
    gpu = None
    if torch.device(device).type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "memory": properties.total_memory,
            "capability": [properties.major, properties.minor],
            "visible_count": torch.cuda.device_count(),
        }
    return {
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "torch_build": torch.__config__.show(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(torch.device(device)),
        "gpu": gpu,
        "environment": {name: os.environ.get(name) for name in HOST_ENVIRONMENT},
    }


def host_drift(saved, current):
    """Names of host facts that differ from the snapshot's; empty for a same-host resume."""
    saved = saved or {}
    return sorted(key for key in current if saved.get(key) != current[key])


def optimizer_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        [names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups
    ]


def rng_state(sampler, replay, device):
    numpy = np.random.get_state()
    target = torch.device(device)
    return {
        "sampler": sampler.getstate(),
        "replay": replay.getstate(),
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(target) if target.type == "cuda" else None,
        "mps": torch.mps.get_rng_state() if target.type == "mps" else None,
    }


def restore_rng(state, sampler, replay, device=None):
    sampler.setstate(state["sampler"])
    replay.setstate(state["replay"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if isinstance(cuda_state, torch.Tensor):
        target = torch.device(device) if device is not None else torch.device("cuda")
        if target.type == "cuda":
            torch.cuda.set_rng_state(cuda_state.cpu(), device=target)
    elif isinstance(cuda_state, (list, tuple)) and cuda_state:
        target = torch.device(device) if device is not None else torch.device("cuda")
        if target.type == "cuda":
            idx = (
                target.index
                if target.index is not None and 0 <= target.index < len(cuda_state)
                else 0
            )
            torch.cuda.set_rng_state(cuda_state[idx].cpu(), device=target)
    mps_state = state.get("mps")
    if mps_state is not None:
        target = torch.device(device) if device is not None else torch.device("mps")
        if target.type == "mps":
            torch.mps.set_rng_state(mps_state.cpu())


def save_resume(
    path,
    model,
    scales,
    optimizer,
    state,
    contract,
    sampler,
    replay,
    metadata,
    host=None,
    *,
    device=None,
):
    active_device = device
    if active_device is None:
        parameter = next(model.parameters(), None)
        active_device = (
            parameter.device
            if parameter is not None
            else (host or {}).get("device") or contract["device"]
        )
    save_checkpoint(
        path,
        model,
        scales,
        optimizer=optimizer,
        metadata={
            **metadata,
            "resume": {
                "contract": contract,
                "host": host,
                "optimizer_names": optimizer_names(model, optimizer),
                "rng": rng_state(sampler, replay, active_device),
                "state": state,
            },
        },
    )


def check_contract(payload, contract):
    resume = payload.get("metadata", {}).get("resume")
    if not isinstance(resume, dict) or resume.get("contract") != contract:
        raise ValueError("resume execution contract changed or snapshot is not resumable")
    if resume["state"].get("phase") not in ("training", "evaluation"):
        raise ValueError("invalid resume phase")
    return resume


def cached_evaluation(output, split, binding, evaluator):
    """Never silently repeat an interrupted or completed test consumption."""
    folder = Path(output) / "evaluations"
    path, started = folder / f"{split}.json", folder / f"{split}-started.json"
    if path.exists():
        cache = json.loads(path.read_text())
        content = cache["content"]
        if cache.get("sha256") != digest(content) or content.get("binding") != binding:
            raise ValueError(f"invalid {split} evaluation cache or checkpoint binding")
        return content["metrics"], content["records"]
    if split == "test" and started.exists():
        raise ValueError(
            "test evaluation was interrupted without a complete cache; refusing to rescore"
        )
    atomic_json(started, binding)
    metrics, records = evaluator()
    content = {"binding": binding, "metrics": metrics, "records": records}
    atomic_json(path, {"content": content, "sha256": digest(content)})
    return metrics, records


def completed_result(output, contract):
    output = Path(output)
    path = output / "completed.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt.get("contract_sha256") != digest(contract):
        raise ValueError("completed run contract mismatch")
    for name, expected in receipt["artifacts"].items():
        if file_sha256(output / name) != expected:
            raise ValueError(f"completed run artifact changed: {name}")
    return json.loads((output / "summary.json").read_text())


def complete_run(output, contract):
    output = Path(output)
    names = [
        "summary.json",
        "coverage.json",
        "history.json",
        "initial.pt",
        "last.pt",
        "selected.pt",
        "admission.json",
        "resolved-config.json",
        *(f"{split}-predictions.jsonl" for split in ("train", "validation", "test")),
        *(f"evaluations/{split}.json" for split in ("train", "validation", "test")),
    ]
    atomic_json(
        output / "completed.json",
        {
            "contract_sha256": digest(contract),
            "artifacts": {name: file_sha256(output / name) for name in names},
        },
    )
