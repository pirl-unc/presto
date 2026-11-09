"""Close the corrected preflight by canonical comparison of frozen receipts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import modal


FID = "2026-11-08_1800_codex_unified-parity-retrain"
PRIMARY = {
    "path": f"{FID}/preflight-v2-primary/receipt.json",
    "bytes": 25_681,
    "sha256": "6efe6b3f64cbff080f9b03b9dce0ca1906a008d487fee3bd7d9eb8532ac6aaa2",
}
INDEPENDENT = {
    "path": f"{FID}/preflight-v2-verification-r2/receipt.json",
    "bytes": 26_182,
    "sha256": "fdce01321abedf12f0e3aa961f0bd8ab93bade1896a101bc65530d6015178326",
}
COMPARE_KEYS = (
    "rows",
    "partitions",
    "census",
    "manifest_sha256",
    "effective_manifest_sha256",
    "model",
    "training",
    "scales",
    "admission",
)

app = modal.App("presto-unified-parity-receipt-closure-20261109")
volume = modal.Volume.from_name("presto-checkpoints")
image = modal.Image.debian_slim(python_version="3.12").env({"PYTHONDONTWRITEBYTECODE": "1"})
if modal.is_local():
    FAMILY = Path(
        os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[1])
    ).resolve()
    image = image.add_local_file(
        FAMILY / "reproduce/v5/launch.json", "/bundle/launch.json"
    ).add_local_file(
        FAMILY / "code/compare_preflight_receipts.py",
        "/bundle/compare_preflight_receipts.py",
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def require_equal(left, right, label: str) -> str:
    left_bytes = canonical_bytes(left)
    right_bytes = canonical_bytes(right)
    if left_bytes != right_bytes:
        raise ValueError(f"canonical receipt mismatch: {label}")
    return hashlib.sha256(left_bytes).hexdigest()


def compare_receipts(primary: dict, independent: dict) -> dict:
    if primary["status"] != "passed":
        raise ValueError("primary receipt did not pass")
    if independent["status"] != "failed":
        raise ValueError("independent source receipt lacks the expected raw comparison stop")
    error = independent.get("verification_error") or {}
    if error.get("error") != "independent preflight mismatch: reference/model":
        raise ValueError("independent source receipt has an unexpected failure")
    if primary["independent_verification"] is not False:
        raise ValueError("primary receipt identity changed")
    if independent["independent_verification"] is not True:
        raise ValueError("independent receipt identity changed")

    contract_hashes = {
        "parent": require_equal(primary["parent"], independent["parent"], "parent"),
        "sources": require_equal(primary["sources"], independent["sources"], "sources"),
    }
    for name, receipt in (("primary", primary), ("independent", independent)):
        if receipt["setup_error"] is not None or receipt["matched_contract"] is not True:
            raise ValueError(f"{name} setup/matched-contract gate failed")
        for key in (
            "test_accessed",
            "locked_confirmation_accessed",
            "model_predictions_created",
        ):
            if receipt[key] is not False:
                raise ValueError(f"{name} forbidden-access gate failed: {key}")
        if receipt["optimizer_steps"] != 0:
            raise ValueError(f"{name} optimizer gate failed")
        if receipt["requested_gpu"] is not None or receipt["observed_gpu"] is not None:
            raise ValueError(f"{name} GPU gate failed")

    condition_hashes = {}
    for name in ("reference", "parity"):
        left = primary["conditions"][name]
        right = independent["conditions"][name]
        if left["status"] != "passed" or right["status"] != "passed":
            raise ValueError(f"condition did not pass in both receipts: {name}")
        condition_hashes[name] = {
            key: require_equal(left[key], right[key], f"{name}/{key}") for key in COMPARE_KEYS
        }

    expected = {
        **primary["sources"],
        "selected-stage-a.pt": primary["parent"],
    }
    source_copy_hashes = {}
    for name, spec in expected.items():
        left = primary["source_copies"][name]
        right = independent["source_copies"][name]
        observed = {key: left[key] for key in ("bytes", "sha256")}
        if observed != {key: spec[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"primary source-copy contract changed: {name}")
        if observed != {key: right[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"independent source-copy contract changed: {name}")
        source_copy_hashes[name] = canonical_sha256(observed)

    return {
        "contract_hashes": contract_hashes,
        "condition_hashes": condition_hashes,
        "source_copy_hashes": source_copy_hashes,
    }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=600,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def close_comparison() -> dict:
    started = time.monotonic()
    contract = json.loads(Path("/bundle/launch.json").read_text())
    if (
        sha256("/bundle/compare_preflight_receipts.py")
        != contract["source_sha256"]["code/compare_preflight_receipts.py"]
    ):
        raise ValueError("comparison source differs from frozen contract")
    volume.reload()
    receipts = {}
    for name, spec in (("primary", PRIMARY), ("independent", INDEPENDENT)):
        path = Path("/volume") / spec["path"]
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"frozen {name} receipt changed")
        receipts[name] = json.loads(path.read_text())
    comparisons = compare_receipts(receipts["primary"], receipts["independent"])
    output = Path("/volume") / FID / "preflight-v2-verification-comparison"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "presto.unified_parity_preflight_comparison.v1",
        "status": "passed",
        "experiment": FID,
        "source_commit": contract["git"]["source_commit"],
        "primary_receipt": PRIMARY,
        "independent_receipt": INDEPENDENT,
        "raw_independent_status": "failed_representation_only",
        "scientific_fields_canonical_equal": True,
        "compare_keys": list(COMPARE_KEYS),
        **comparisons,
        "test_accessed": False,
        "locked_confirmation_accessed": False,
        "model_predictions_created": False,
        "optimizer_steps": 0,
        "requested_gpu": None,
        "observed_gpu": None,
        "resources": {
            "cpu": 2,
            "memory_mib": 4096,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    path = output / "receipt.json"
    write_json(path, result)
    volume.commit()
    return {
        "status": result["status"],
        "scientific_fields_canonical_equal": True,
        "receipt": {
            "path": str(path.relative_to("/volume")),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        },
    }


def local_contract_test() -> None:
    family = Path(__file__).resolve().parents[1]
    primary = json.loads((family / "results/v2/preflight-v2-receipt.json").read_text())
    independent = json.loads(
        (family / "results/v4/preflight-v2-verification-r2-receipt.json").read_text()
    )
    result = compare_receipts(primary, independent)
    assert set(result["condition_hashes"]) == {"reference", "parity"}
    perturbed = deepcopy(independent)
    perturbed["conditions"]["parity"]["rows"] += 1
    rejection = None
    try:
        compare_receipts(primary, perturbed)
    except ValueError as error:
        rejection = str(error)
    assert rejection == "canonical receipt mismatch: parity/rows"


@app.local_entrypoint()
def main() -> None:
    local_contract_test()
    contract = json.loads((FAMILY / "reproduce/v5/launch.json").read_text())
    if (
        sha256(FAMILY / "code/compare_preflight_receipts.py")
        != contract["source_sha256"]["code/compare_preflight_receipts.py"]
    ):
        raise ValueError("local comparison source differs from frozen contract")
    call_path = FAMILY / "reproduce/v5/comparison-call.json"
    if call_path.exists():
        raise ValueError(f"{call_path.name} already exists")
    call = close_comparison.spawn()
    write_json(call_path, {"call_id": call.object_id, "submitted_unix": time.time()})
    print(call.object_id)


if __name__ == "__main__":
    local_contract_test()
    print(json.dumps({"status": "passed", "comparison": "canonical-json"}))
