"""Run matched reference and comparator-parity fits from the same Stage A parent."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path

import modal

FID = "2026-11-08_1800_codex_unified-parity-retrain"
GPU_FID = "2026-11-07_1915_codex_unified-presentation"
BA_FID = "2026-11-08_1740_codex_netmhc-ba-parity-projection"
GPU = os.environ.get("PRESTO_MODAL_GPU", "H100!")
if GPU != "H100!":
    raise ValueError("this experiment is frozen to PRESTO_MODAL_GPU=H100!")
PARENT = {
    "path": f"{GPU_FID}/stage-a/raw/selected.pt",
    "bytes": 3_910_569,
    "sha256": "e6cc6fbb9950ace32efe418bfa96b119e1b94f0399e6843db0a16b0208937c06",
}
SOURCE_SPECS = {
    "presentation-train.jsonl": {
        "path": f"{GPU_FID}/prepared-v2/presentation-train.jsonl",
        "bytes": 5_977_683_393,
        "sha256": "6609ee375f6bb117f19035560daa6cf50a63b9c229fa46ae24e773afebdda2fc",
    },
    "presentation-val.jsonl": {
        "path": f"{GPU_FID}/prepared-v2/presentation-val.jsonl",
        "bytes": 218_542_980,
        "sha256": "0313b2d6b394320da7dcf6aa9144cbc44b73abe48b16380e03c1b27a9559fc52",
    },
    "concentrations-s42-concentrations-s42-train.jsonl": {
        "path": f"{GPU_FID}/prepared-v2/concentrations-s42-concentrations-s42-train.jsonl",
        "bytes": 836_622_693,
        "sha256": "8593fbd416a489b46397c4fae288d5916ad55d2febec7f7798bc4bb3f788ec1c",
    },
    "concentrations-s42-concentrations-s42-validation.jsonl": {
        "path": f"{GPU_FID}/prepared-v2/concentrations-s42-concentrations-s42-validation.jsonl",
        "bytes": 12_196_679,
        "sha256": "08e41a6c006274b5929142fc6bd4687a289a08df9cc5212edd92399f907f2f8a",
    },
    "netmhcpan42-ba-train.jsonl": {
        "path": f"{BA_FID}/prepared-v2/netmhcpan42-ba-train.jsonl",
        "bytes": 141_870_035,
        "sha256": "2c007e6b46a99835f95ec612aeb498f4e52940aa81c22809ad48b5e922d6d0ca",
    },
    "netmhciipan43-ba-train.jsonl": {
        "path": f"{BA_FID}/prepared-v2/netmhciipan43-ba-train.jsonl",
        "bytes": 112_640_875,
        "sha256": "9fe752274c895cb4cd46e36dfeff43a3db73e0c96fa24ba495e1fcfd7f24c66a",
    },
}
MANIFESTS = {
    "reference": {
        "bundle": "/bundle/reference-manifest.json",
        "bytes": 7_131,
        "sha256": "e2724c0ba873a98fa15f90c6ccc06013f320e65eb8968af0400bf7722e6db16d",
    },
    "parity": {
        "bundle": "/bundle/parity-manifest.json",
        "bytes": 8_221,
        "sha256": "8259d1ec1b59430d7078174a7718d8b7157e351a0261e70384996692605bbd7c",
    },
}
CLOSURE = {
    "path": f"{FID}/preflight-v2-verification-comparison/receipt.json",
    "bytes": 3_901,
    "sha256": "b60d3f9343808b23fa67a6f636df0be01699f59475f7af596701c1ac00121c16",
}
REQUIREMENTS = (
    "torch==2.14.1",
    "numpy==2.5.4",
    "pyyaml==6.0.3",
    "mhcgnomes==3.64.6",
    "mhcseqs==2.6.13",
    "tqdm==4.70.1",
    "psutil==7.2.2",
)
RUNTIME_IDENTITY_KEYS = ("files", "bytes", "sha256")

app = modal.App("presto-unified-parity-fits-20261109")
volume = modal.Volume.from_name("presto-checkpoints")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*REQUIREMENTS)
    .env(
        {
            "PYTHONPATH": "/runtime",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
)
if modal.is_local():
    FAMILY = Path(
        os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[1])
    ).resolve()
    ROOT = FAMILY.parents[1]
    image = (
        image.add_local_file(
            FAMILY / "reproduce/training-v2/launch.json", "/bundle/launch.json"
        )
        .add_local_file(FAMILY / "code/fit_pair.py", "/bundle/fit_pair.py")
        .add_local_file(
            FAMILY / "reproduce/training/reference-manifest.json",
            "/bundle/reference-manifest.json",
        )
        .add_local_file(
            FAMILY / "reproduce/training/parity-manifest.json",
            "/bundle/parity-manifest.json",
        )
        .add_local_file(ROOT / "__init__.py", "/runtime/presto/__init__.py")
        .add_local_dir(ROOT / "data", "/runtime/presto/data")
        .add_local_dir(ROOT / "models", "/runtime/presto/models")
        .add_local_dir(ROOT / "training", "/runtime/presto/training")
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def runtime_tree(root: Path) -> dict:
    paths = [root / "__init__.py"]
    for directory in ("data", "models", "training"):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    rows = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(paths)
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def runtime_identity(contract: dict) -> dict:
    missing = [key for key in RUNTIME_IDENTITY_KEYS if key not in contract]
    if missing:
        raise ValueError(f"runtime tree contract lacks identity fields: {missing}")
    return {key: contract[key] for key in RUNTIME_IDENTITY_KEYS}


def verified_copy(source: Path, destination: Path, spec: dict) -> dict:
    started = time.monotonic()
    if source.stat().st_size != spec["bytes"]:
        raise ValueError(f"source byte count changed: {spec['path']}")
    digest = hashlib.sha256()
    copied = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        for block in iter(lambda: input_stream.read(1 << 24), b""):
            output_stream.write(block)
            digest.update(block)
            copied += len(block)
    observed = digest.hexdigest()
    if copied != spec["bytes"] or observed != spec["sha256"]:
        raise ValueError(f"source hash changed: {spec['path']}")
    return {
        "source_path": str(source.relative_to("/volume")),
        "scratch_path": str(destination),
        "bytes": copied,
        "sha256": observed,
        "elapsed_seconds": time.monotonic() - started,
    }


def gzip_copy(source: Path, destination: Path) -> None:
    with source.open("rb") as input_stream, destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            shutil.copyfileobj(input_stream, compressed)


def artifact(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def compact_receipt(condition: str, receipt: dict) -> dict:
    path = Path("/volume") / FID / "fits" / condition / "receipt.json"
    return {
        "status": receipt["status"],
        "condition": condition,
        "steps": receipt.get("steps"),
        "checkpoint_sha256": receipt.get("checkpoint_sha256"),
        "receipt": {
            "path": str(path.relative_to("/volume")),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        },
    }


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    ephemeral_disk=524288,
    timeout=10800,
    max_containers=2,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def fit(condition: str) -> dict:
    import psutil
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.training import modular_run as runner

    if condition not in MANIFESTS:
        raise ValueError(f"unknown condition: {condition}")
    pins = read_json("/bundle/launch.json")
    if sha256("/bundle/fit_pair.py") != pins["source_sha256"]["code/fit_pair.py"]:
        raise ValueError("fit source differs from frozen contract")
    for name, spec in MANIFESTS.items():
        path = Path(spec["bundle"])
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"{name} manifest differs from frozen contract")
    observed_runtime_tree = runtime_tree(Path("/runtime/presto"))
    expected_runtime_tree = runtime_identity(pins["runtime_tree"])
    if observed_runtime_tree != expected_runtime_tree:
        raise ValueError(
            "runtime source tree differs from frozen contract: "
            + json.dumps(
                {"expected": expected_runtime_tree, "observed": observed_runtime_tree},
                sort_keys=True,
            )
        )

    volume.reload()
    closure_path = Path("/volume") / CLOSURE["path"]
    if (
        closure_path.stat().st_size != CLOSURE["bytes"]
        or sha256(closure_path) != CLOSURE["sha256"]
    ):
        raise ValueError("corrected preflight closure changed")
    closure = read_json(closure_path)
    if (
        closure["status"] != "passed"
        or not closure["scientific_fields_canonical_equal"]
    ):
        raise ValueError("corrected preflight gate did not pass")

    target = Path("/volume") / FID / "fits" / condition
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = target / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["source_commit"] != pins["git"]["source_commit"]:
            raise ValueError("existing fit receipt belongs to another source commit")
        return compact_receipt(condition, receipt)

    execution = {
        "condition": condition,
        "source_commit": pins["git"]["source_commit"],
        "manifest_sha256": MANIFESTS[condition]["sha256"],
        "parent_sha256": PARENT["sha256"],
        "preflight_closure_sha256": CLOSURE["sha256"],
        "runtime_tree": observed_runtime_tree,
        "seed": 42,
        "max_steps": 5000,
        "requested_gpu": GPU,
    }
    execution_path = target / "execution-contract.json"
    if execution_path.exists():
        if read_json(execution_path) != execution:
            raise ValueError("existing fit execution contract changed")
    else:
        write_json(execution_path, execution)
        volume.commit()

    began = time.monotonic()
    attempt = str(time.time_ns())
    events = target / "segments"
    events.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    gpu = torch.cuda.get_device_properties(0)
    if "H100" not in gpu.name:
        raise ValueError(f"requested H100! resolved to unexpected device: {gpu.name}")
    hardware = {
        "requested_gpu": GPU,
        "actual_gpu": gpu.name,
        "gpu_bytes": gpu.total_memory,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "tf32": False,
        "amp": False,
        "compile": False,
        "threads": 4,
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    write_json(events / f"{attempt}-hardware.json", hardware)
    if not (target / "hardware.json").exists():
        write_json(target / "hardware.json", hardware)
    volume.commit()

    manifest_path = Path(MANIFESTS[condition]["bundle"])
    manifest = read_json(manifest_path)
    scratch_root = Path("/tmp/presto-unified-parity-v2-input")
    if scratch_root.exists():
        raise ValueError("frozen input scratch path already exists")
    scratch_root.mkdir(parents=True)
    copies = {}
    for source in manifest["sources"]:
        name = Path(source["path"]).name
        spec = SOURCE_SPECS[name]
        if source["sha256"] != spec["sha256"]:
            raise ValueError(f"manifest/source mismatch: {name}")
        destination = Path(source["path"])
        if destination.parent != scratch_root:
            raise ValueError(f"manifest scratch path changed: {destination}")
        copies[name] = verified_copy(Path("/volume") / spec["path"], destination, spec)
    parent = Path(manifest["pretrained"]["path"])
    if parent != scratch_root / "selected-stage-a.pt":
        raise ValueError("manifest parent scratch path changed")
    copies[parent.name] = verified_copy(
        Path("/volume") / PARENT["path"], parent, PARENT
    )

    raw = target / "raw"
    resume = (raw / "resume.pt").exists()
    initial_steps = 0
    if resume:
        saved = torch.load(raw / "resume.pt", map_location="cpu", weights_only=True)
        initial_steps = saved["metadata"]["resume"]["state"]["steps"]
        del saved
    prior_seconds = sum(
        read_json(path)["elapsed_seconds"] for path in events.glob("*-end.json")
    )
    state = {
        "initial_steps": initial_steps,
        "new_steps": 0,
        "new_examples": 0,
        "peak_rss_bytes": 0,
    }
    process = psutil.Process()
    original_train_step = runner.train_step
    original_save_resume = runner.save_resume
    segment_closed = False

    def observed(model, optimizer, rows, scales, **kwargs):
        if time.monotonic() - began > 9900:
            raise TimeoutError(
                "bounded parity-fit runtime reached; exact continuation required"
            )
        result = original_train_step(model, optimizer, rows, scales, **kwargs)
        state["new_steps"] += 1
        state["new_examples"] += len(rows)
        state["peak_rss_bytes"] = max(
            state["peak_rss_bytes"], process.memory_info().rss
        )
        if state["new_steps"] == 1 or state["new_steps"] % 100 == 0:
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "status": "training",
                        "steps": initial_steps + state["new_steps"],
                        "loss": result["loss"],
                        "elapsed_seconds": time.monotonic() - began,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        return result

    def durable_resume(*args, **kwargs):
        result = original_save_resume(*args, **kwargs)
        volume.commit()
        return result

    runner.train_step = observed
    runner.save_resume = durable_resume
    try:
        summary = runner.fit_index(
            ObservationIndex(manifest_path), raw, device="cuda", resume=resume
        )
        if summary["steps"] != manifest["training"]["max_steps"]:
            raise ValueError("fit did not reach the frozen step budget")
        if summary["coverage_gate"]["status"] != "passed":
            raise ValueError("fit coverage gate failed")
        if summary["evaluation_populations"]["test"]["available_rows"] != 0:
            raise ValueError("TEST rows appeared in the development fit")
        test_predictions = raw / "test-predictions.jsonl"
        if test_predictions.stat().st_size != 0:
            raise ValueError("TEST prediction dump is not empty")

        for name in ("train", "validation", "test"):
            gzip_copy(
                raw / f"{name}-predictions.jsonl",
                target / f"{name}-predictions.jsonl.gz",
            )
        resources = {
            **state,
            "steps": summary["steps"],
            "elapsed_seconds": prior_seconds + time.monotonic() - began,
            "current_segment_seconds": time.monotonic() - began,
            "prior_segment_seconds": prior_seconds,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cpu": 4,
            "memory_mib": 32768,
            "ephemeral_disk_mib": 524288,
            "source_copies": copies,
        }
        resources["peak_rss_bytes"] = max(
            resources["peak_rss_bytes"],
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        write_json(target / "resources.json", resources)
        write_json(
            events / f"{attempt}-end.json",
            {"elapsed_seconds": time.monotonic() - began, **state},
        )
        segment_closed = True
        files = {}
        for path in sorted(target.rglob("*")):
            if path.is_file() and path != receipt_path:
                files[str(path.relative_to(target))] = artifact(path, target)
        receipt = {
            "schema": "presto.unified_parity_fit.v1",
            "status": "completed",
            "experiment": FID,
            "condition": condition,
            "source_commit": pins["git"]["source_commit"],
            "manifest_sha256": MANIFESTS[condition]["sha256"],
            "preflight_closure_sha256": CLOSURE["sha256"],
            "parent_sha256": sha256(parent),
            "runtime_tree": observed_runtime_tree,
            "steps": summary["steps"],
            "checkpoint_sha256": sha256(raw / "selected.pt"),
            "test_accessed": False,
            "locked_confirmation_accessed": False,
            "files": files,
            "hardware": hardware,
            "resources": resources,
        }
        write_json(receipt_path, receipt)
        volume.commit()
        return compact_receipt(condition, receipt)
    except BaseException as error:
        write_json(
            events / f"{attempt}-failure.json",
            {"error_type": type(error).__name__, "error": str(error), **state},
        )
        volume.commit()
        raise
    finally:
        if not segment_closed:
            write_json(
                events / f"{attempt}-end.json",
                {"elapsed_seconds": time.monotonic() - began, **state},
            )
            volume.commit()
        runner.train_step = original_train_step
        runner.save_resume = original_save_resume


def local_contract_test() -> None:
    family = Path(__file__).resolve().parents[1]
    reference = read_json(family / "reproduce/training/reference-manifest.json")
    parity = read_json(family / "reproduce/training/parity-manifest.json")
    reduced = json.loads(json.dumps(parity))
    added = reduced["sources"][-2:]
    reduced["sources"] = reduced["sources"][:-2]
    if reduced != reference:
        raise ValueError("training manifests differ outside the two BA sources")
    if [Path(row["path"]).name for row in added] != [
        "netmhcpan42-ba-train.jsonl",
        "netmhciipan43-ba-train.jsonl",
    ]:
        raise ValueError("parity training manifest lacks the exact BA sources")
    if (
        reference["training"] != parity["training"]
        or reference["model"] != parity["model"]
    ):
        raise ValueError("training/model contract differs between conditions")
    if reference["pretrained"] != parity["pretrained"]:
        raise ValueError("parent contract differs between conditions")
    if any("test" in row["path"] for row in parity["sources"]):
        raise ValueError("TEST source appears in a training manifest")
    example = {"paths": "descriptive scope", "files": 1, "bytes": 2, "sha256": "x"}
    if runtime_identity(example) != {"files": 1, "bytes": 2, "sha256": "x"}:
        raise ValueError("runtime identity includes descriptive metadata")


@app.local_entrypoint()
def main() -> None:
    local_contract_test()
    pins = read_json(FAMILY / "reproduce/training-v2/launch.json")
    if sha256(FAMILY / "code/fit_pair.py") != pins["source_sha256"]["code/fit_pair.py"]:
        raise ValueError("local fit source differs from frozen contract")
    calls_path = FAMILY / "reproduce/training-v2/calls.json"
    calls = read_json(calls_path) if calls_path.exists() else {}
    for condition in ("reference", "parity"):
        if condition in calls:
            continue
        call = fit.spawn(condition)
        calls[condition] = {"call_id": call.object_id, "submitted_unix": time.time()}
        write_json(calls_path, calls)
        print(f"{condition}\t{call.object_id}")


if __name__ == "__main__":
    local_contract_test()
    print(json.dumps({"status": "passed", "conditions": ["reference", "parity"]}))
