"""Run the complete canonical loader/preflight before parity-data GPU fits."""

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
        "path": f"{BA_FID}/prepared/netmhcpan42-ba-train.jsonl",
        "bytes": 114_297_607,
        "sha256": "4491918763c1cf1c64825af0228357f43516293d1b6ea73ef8a6a7554d4635df",
    },
    "netmhciipan43-ba-train.jsonl": {
        "path": f"{BA_FID}/prepared/netmhciipan43-ba-train.jsonl",
        "bytes": 91_225_030,
        "sha256": "153a51f8997093c0b8cf724dc8ff649a676ee3ee2e516c52dff810bb18d00ee2",
    },
}
BA_SOURCES = (
    {
        "label_policy": (
            "Archive-native NetMHCpan 4.2 BA; unspecified assay IC50 with literal censoring"
        ),
        "path": "netmhcpan42-ba-train.jsonl",
        "population": "Human full-groove comparator-parity BA TRAIN projection",
        "sha256": SOURCE_SPECS["netmhcpan42-ba-train.jsonl"]["sha256"],
        "source": "NetMHCpan-4.2-training/BA",
        "split_policy": (
            "TRAIN only; current exact-pMHC dedup and exposed-development exact/core9 embargo"
        ),
    },
    {
        "label_policy": (
            "Archive-native NetMHCIIpan 4.3 BA; unspecified assay IC50 with literal censoring"
        ),
        "path": "netmhciipan43-ba-train.jsonl",
        "population": "Human paired-groove comparator-parity BA TRAIN projection",
        "sha256": SOURCE_SPECS["netmhciipan43-ba-train.jsonl"]["sha256"],
        "source": "NetMHCIIpan-4.3-training/BA",
        "split_policy": (
            "TRAIN only; current exact-pMHC dedup and exposed-development exact/core9 embargo"
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

app = modal.App("presto-unified-parity-preflight-20261108")
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
        .add_local_file(FAMILY / "reproduce/launch.json", "/bundle/launch.json")
        .add_local_file(FAMILY / "code/preflight.py", "/bundle/preflight.py")
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


def effective_manifest(base: dict, condition: str, volume_root: str = "/volume") -> dict:
    """Resolve one frozen template without changing its scientific contract."""
    if condition not in ("reference", "parity"):
        raise ValueError(f"unknown condition: {condition}")
    result = deepcopy(base)
    root = Path(volume_root)
    for source in result["sources"]:
        spec = SOURCE_SPECS[Path(source["path"]).name]
        if source["sha256"] != spec["sha256"]:
            raise ValueError("template/source receipt mismatch")
        source["path"] = str(root / spec["path"])
    if condition == "parity":
        for source in BA_SOURCES:
            resolved = deepcopy(source)
            spec = SOURCE_SPECS[resolved["path"]]
            resolved["path"] = str(root / spec["path"])
            result["sources"].append(resolved)
    result["pretrained"]["path"] = str(root / PARENT["path"])
    return result


def assert_matched_contract(reference: dict, parity: dict) -> None:
    """The two effective manifests may differ only by two declared TRAIN sources."""
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
    if sha256("/bundle/preflight.py") != contract["source_sha256"]["code/preflight.py"]:
        raise ValueError("preflight source differs from frozen launch contract")
    if sha256("/bundle/base-manifest.json") != contract["source_sha256"][
        "reproduce/base-manifest.json"
    ]:
        raise ValueError("base manifest differs from frozen launch contract")
    base = json.loads(Path("/bundle/base-manifest.json").read_text())
    reference = effective_manifest(base, "reference")
    parity = effective_manifest(base, "parity")
    assert_matched_contract(reference, parity)
    volume.reload()
    if (Path("/volume") / PARENT["path"]).stat().st_size != PARENT["bytes"]:
        raise ValueError("parent checkpoint byte count changed")
    if sha256(Path("/volume") / PARENT["path"]) != PARENT["sha256"]:
        raise ValueError("parent checkpoint hash changed")
    for spec in SOURCE_SPECS.values():
        if (Path("/volume") / spec["path"]).stat().st_size != spec["bytes"]:
            raise ValueError(f"source byte count changed: {spec['path']}")

    output = Path("/volume") / FID / output_name
    output.mkdir(parents=True, exist_ok=False)
    results = {}
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
        except Exception as error:  # fail closed while preserving the exact first failure
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

    status = "passed" if set(results) == {"reference", "parity"} and all(
        value["status"] == "passed" for value in results.values()
    ) else "failed"
    receipt = {
        "schema": "presto.unified_parity_preflight.v1",
        "status": status,
        "experiment": FID,
        "independent_verification": independent,
        "source_commit": contract["git"]["source_commit"],
        "source_dirty": contract["git"]["dirty"],
        "parent": PARENT,
        "sources": SOURCE_SPECS,
        "conditions": results,
        "matched_contract": True,
        "test_accessed": False,
        "model_predictions_created": False,
        "optimizer_steps": 0,
        "requested_gpu": None,
        "observed_gpu": None,
        "resources": {
            "cpu": 8,
            "memory_mib": 32_768,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    write_json(output / "receipt.json", receipt)
    volume.commit()
    return receipt


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def preflight_primary() -> dict:
    return run_canonical_pass("preflight-primary", independent=False)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def preflight_verification() -> dict:
    primary = json.loads(
        (Path("/volume") / FID / "preflight-primary/receipt.json").read_text()
    )
    if primary["status"] != "passed":
        raise ValueError("cannot verify a failed primary preflight")
    result = run_canonical_pass("preflight-verification", independent=True)
    for name in ("reference", "parity"):
        keys = ("rows", "partitions", "census", "model", "training", "scales", "admission")
        for key in keys:
            if result["conditions"][name][key] != primary["conditions"][name][key]:
                raise ValueError(f"independent preflight mismatch: {name}/{key}")
    result["primary_receipt_sha256"] = sha256(
        Path("/volume") / FID / "preflight-primary/receipt.json"
    )
    write_json(Path("/volume") / FID / "preflight-verification/receipt.json", result)
    volume.commit()
    return result


def local_contract_test() -> None:
    family = Path(__file__).resolve().parents[1]
    base = json.loads((family / "reproduce/base-manifest.json").read_text())
    reference = effective_manifest(base, "reference")
    parity = effective_manifest(base, "parity")
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
    contract = json.loads((FAMILY / "reproduce/launch.json").read_text())
    if sha256(FAMILY / "code/preflight.py") != contract["source_sha256"]["code/preflight.py"]:
        raise ValueError("local preflight source differs from frozen contract")
    stem = "preflight-verification" if verify else "preflight"
    call_path = FAMILY / "reproduce" / f"{stem}-call.json"
    if call_path.exists():
        raise ValueError(f"{call_path.name} already exists")
    call = (preflight_verification if verify else preflight_primary).spawn()
    write_json(call_path, {"call_id": call.object_id, "submitted_unix": time.time()})
    print(call.object_id)


if __name__ == "__main__":
    local_contract_test()
    print(json.dumps({"status": "passed", "conditions": ["reference", "parity"]}))
