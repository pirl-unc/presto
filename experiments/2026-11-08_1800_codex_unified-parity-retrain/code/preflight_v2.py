"""Run the corrected complete canonical loader before parity-data GPU fits."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import modal


FID = "2026-11-08_1800_codex_unified-parity-retrain"
GPU_FID = "2026-11-07_1915_codex_unified-presentation"
BA_FID = "2026-11-08_1740_codex_netmhc-ba-parity-projection"
SCRATCH = Path("/tmp/presto-unified-parity-v2-input")
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
BA_SOURCES = (
    {
        "label_policy": (
            "Archive-native NetMHCpan 4.2 BA; unspecified assay IC50 with literal censoring"
        ),
        "path": "netmhcpan42-ba-train.jsonl",
        "population": "Human full-groove comparator-parity BA TRAIN projection v2",
        "sha256": SOURCE_SPECS["netmhcpan42-ba-train.jsonl"]["sha256"],
        "source": "NetMHCpan-4.2-training/BA",
        "split_policy": (
            "TRAIN only; current exact-pMHC dedup plus affinity core9 and presentation "
            "core8/Hamming1 development embargo"
        ),
    },
    {
        "label_policy": (
            "Archive-native NetMHCIIpan 4.3 BA; unspecified assay IC50 with literal censoring"
        ),
        "path": "netmhciipan43-ba-train.jsonl",
        "population": "Human paired-groove comparator-parity BA TRAIN projection v2",
        "sha256": SOURCE_SPECS["netmhciipan43-ba-train.jsonl"]["sha256"],
        "source": "NetMHCIIpan-4.3-training/BA",
        "split_policy": (
            "TRAIN only; current exact-pMHC dedup plus affinity core9 and presentation "
            "core8/Hamming1 development embargo"
        ),
    },
)
REQUIREMENTS = (
    "torch==2.14.1",
    "numpy==2.5.4",
    "pyyaml==6.0.3",
    "mhcgnomes==3.64.6",
    "mhcseqs==2.6.13",
    "tqdm==4.70.1",
    "psutil==7.2.2",
)

app = modal.App("presto-unified-parity-preflight-v2-20261108")
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
        image.add_local_file(FAMILY / "reproduce/base-manifest.json", "/bundle/base-manifest.json")
        .add_local_file(FAMILY / "reproduce/v2/launch.json", "/bundle/launch.json")
        .add_local_file(FAMILY / "code/preflight_v2.py", "/bundle/preflight_v2.py")
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


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


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


def copy_inputs() -> tuple[dict[str, Path], Path, dict]:
    if SCRATCH.exists():
        raise ValueError(f"scratch path already exists: {SCRATCH}")
    SCRATCH.mkdir(parents=True)
    volume.reload()
    source_paths = {}
    receipts = {}
    for name, spec in SOURCE_SPECS.items():
        destination = SCRATCH / name
        receipts[name] = verified_copy(Path("/volume") / spec["path"], destination, spec)
        source_paths[name] = destination
    parent_path = SCRATCH / "selected-stage-a.pt"
    receipts["selected-stage-a.pt"] = verified_copy(
        Path("/volume") / PARENT["path"], parent_path, PARENT
    )
    return source_paths, parent_path, receipts


def effective_manifest(
    base: dict, condition: str, source_paths: dict[str, Path], parent_path: Path
) -> dict:
    """Resolve one frozen template without changing its scientific contract."""
    if condition not in ("reference", "parity"):
        raise ValueError(f"unknown condition: {condition}")
    result = deepcopy(base)
    for source in result["sources"]:
        name = Path(source["path"]).name
        spec = SOURCE_SPECS[name]
        if source["sha256"] != spec["sha256"]:
            raise ValueError("template/source receipt mismatch")
        source["path"] = str(source_paths[name])
    if condition == "parity":
        for source in BA_SOURCES:
            resolved = deepcopy(source)
            resolved["path"] = str(source_paths[resolved["path"]])
            result["sources"].append(resolved)
    result["pretrained"]["path"] = str(parent_path)
    return result


def assert_matched_contract(reference: dict, parity: dict) -> None:
    """The effective manifests may differ only by the two declared TRAIN sources."""
    reduced = deepcopy(parity)
    added = reduced["sources"][-2:]
    reduced["sources"] = reduced["sources"][:-2]
    if reduced != reference:
        raise ValueError("reference/parity manifests differ outside added BA sources")
    if [Path(source["path"]).name for source in added] != [
        "netmhcpan42-ba-train.jsonl",
        "netmhciipan43-ba-train.jsonl",
    ]:
        raise ValueError("parity manifest lacks the exact declared BA sources")


def condition_summary(index, config, training, scales, admission, elapsed: float) -> dict:
    partitions = {key: len(rows) for key, rows in sorted(index.partitions.items())}
    if any(key.startswith("test:") for key in partitions):
        raise ValueError("TEST rows are forbidden in this development family")
    return {
        "status": "passed",
        "rows": len(index),
        "partitions": partitions,
        "census": dict(sorted(index.census.items())),
        "manifest_sha256": index.manifest_sha256,
        "model": config.to_dict(),
        "training": training,
        "scales": {key: asdict(value) for key, value in scales.items()},
        "admission": admission,
        "elapsed_seconds": elapsed,
    }


def run_canonical_pass(output_name: str, *, independent: bool) -> dict:
    from presto.data.modular_sources import ObservationIndex
    from presto.training.modular_run import preflight

    started = time.monotonic()
    contract = json.loads(Path("/bundle/launch.json").read_text())
    if sha256("/bundle/preflight_v2.py") != contract["source_sha256"]["code/preflight_v2.py"]:
        raise ValueError("preflight v2 source differs from frozen launch contract")
    if (
        sha256("/bundle/base-manifest.json")
        != contract["source_sha256"]["reproduce/base-manifest.json"]
    ):
        raise ValueError("base manifest differs from frozen launch contract")

    output = Path("/volume") / FID / output_name
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    copied = {}
    setup_error = None
    matched_contract = False
    try:
        source_paths, parent_path, copied = copy_inputs()
        base = json.loads(Path("/bundle/base-manifest.json").read_text())
        reference = effective_manifest(base, "reference", source_paths, parent_path)
        parity = effective_manifest(base, "parity", source_paths, parent_path)
        assert_matched_contract(reference, parity)
        matched_contract = True
        for name, manifest in (("reference", reference), ("parity", parity)):
            condition_dir = output / name
            condition_dir.mkdir()
            manifest_path = condition_dir / "manifest.json"
            write_json(manifest_path, manifest)
            began = time.monotonic()
            try:
                index = ObservationIndex(manifest_path)
                config, training, scales, admission = preflight(index)
                result = condition_summary(
                    index, config, training, scales, admission, time.monotonic() - began
                )
                write_json(condition_dir / "admission.json", admission)
            except Exception as error:  # fail closed and preserve the first exact failure
                result = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.monotonic() - began,
                }
            result["effective_manifest_path"] = str(manifest_path.relative_to("/volume"))
            result["effective_manifest_sha256"] = sha256(manifest_path)
            results[name] = result
            if result["status"] != "passed":
                break
    except Exception as error:
        setup_error = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }

    status = (
        "passed"
        if setup_error is None
        and set(results) == {"reference", "parity"}
        and all(value["status"] == "passed" for value in results.values())
        else "failed"
    )
    receipt = {
        "schema": "presto.unified_parity_preflight.v2",
        "status": status,
        "experiment": FID,
        "independent_verification": independent,
        "source_commit": contract["git"]["source_commit"],
        "source_dirty": contract["git"]["dirty"],
        "parent": PARENT,
        "sources": SOURCE_SPECS,
        "source_copies": copied,
        "setup_error": setup_error,
        "conditions": results,
        "matched_contract": matched_contract,
        "test_accessed": False,
        "locked_confirmation_accessed": False,
        "model_predictions_created": False,
        "optimizer_steps": 0,
        "requested_gpu": None,
        "observed_gpu": None,
        "resources": {
            "cpu": 8,
            "memory_mib": 32_768,
            "ephemeral_disk_mib": 524_288,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    write_json(output / "receipt.json", receipt)
    volume.commit()
    return receipt


FUNCTION = {
    "image": image,
    "cpu": 8,
    "memory": 32768,
    "ephemeral_disk": 524288,
    "timeout": 7200,
    "max_containers": 1,
    "retries": 0,
    "single_use_containers": True,
    "block_network": True,
    "volumes": {"/volume": volume},
}


@app.function(**FUNCTION)
def preflight_v2_primary() -> dict:
    return run_canonical_pass("preflight-v2-primary", independent=False)


@app.function(**FUNCTION)
def preflight_v2_verification() -> dict:
    primary_path = Path("/volume") / FID / "preflight-v2-primary/receipt.json"
    primary = json.loads(primary_path.read_text())
    if primary["status"] != "passed":
        raise ValueError("cannot verify a failed primary v2 preflight")
    result = run_canonical_pass("preflight-v2-verification", independent=True)
    if result["status"] != "passed":
        return result
    verification_error = None
    try:
        for name in ("reference", "parity"):
            keys = (
                "rows",
                "partitions",
                "census",
                "manifest_sha256",
                "model",
                "training",
                "scales",
                "admission",
            )
            for key in keys:
                if result["conditions"][name][key] != primary["conditions"][name][key]:
                    raise ValueError(f"independent preflight mismatch: {name}/{key}")
        for name, spec in {**SOURCE_SPECS, "selected-stage-a.pt": PARENT}.items():
            for key in ("bytes", "sha256"):
                if result["source_copies"][name][key] != spec[key]:
                    raise ValueError(f"independent source-copy mismatch: {name}/{key}")
    except Exception as error:
        result["status"] = "failed"
        verification_error = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    result["verification_error"] = verification_error
    result["primary_receipt_sha256"] = sha256(primary_path)
    write_json(Path("/volume") / FID / "preflight-v2-verification/receipt.json", result)
    volume.commit()
    return result


def local_contract_test() -> None:
    family = Path(__file__).resolve().parents[1]
    base = json.loads((family / "reproduce/base-manifest.json").read_text())
    paths = {name: SCRATCH / name for name in SOURCE_SPECS}
    reference = effective_manifest(base, "reference", paths, SCRATCH / "selected-stage-a.pt")
    parity = effective_manifest(base, "parity", paths, SCRATCH / "selected-stage-a.pt")
    assert_matched_contract(reference, parity)
    assert len(reference["sources"]) == 4
    assert len(parity["sources"]) == 6
    assert not any("test" in source["path"] for source in parity["sources"])
    assert all(Path(source["path"]).is_absolute() for source in parity["sources"])
    assert parity["model"] == reference["model"]
    assert parity["training"] == reference["training"]
    assert parity["pretrained"] == reference["pretrained"]


@app.local_entrypoint()
def main(verify: bool = False) -> None:
    local_contract_test()
    contract = json.loads((FAMILY / "reproduce/v2/launch.json").read_text())
    if sha256(FAMILY / "code/preflight_v2.py") != contract["source_sha256"]["code/preflight_v2.py"]:
        raise ValueError("local preflight v2 source differs from frozen contract")
    stem = "preflight-v2-verification" if verify else "preflight-v2"
    call_path = FAMILY / "reproduce/v2" / f"{stem}-call.json"
    if call_path.exists():
        raise ValueError(f"{call_path.name} already exists")
    call = (preflight_v2_verification if verify else preflight_v2_primary).spawn()
    write_json(call_path, {"call_id": call.object_id, "submitted_unix": time.time()})
    print(call.object_id)


if __name__ == "__main__":
    local_contract_test()
    print(json.dumps({"status": "passed", "conditions": ["reference", "parity"]}))
