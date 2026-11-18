"""Canonical native MS fits with frozen binding dependencies and fixed budgets."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
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
from presto.models.modular_dependencies import parameter_dependencies
from presto.training.modular import load_checkpoint
from presto.training.modular_evaluation import evaluate
from presto.training.modular_resume import SOURCE_FILES, atomic_json
from presto.training.modular_run import fit_index, preflight

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
RAW = Path("/private/tmp/presto-mhc2-20261026/ms-length-background/native")
ARMS = {"native101": 101, "native102": 102, "permuted101": 101}
LIMITS = {
    "startup_ram": 13 * 1024**3,
    "ram_floor": 10 * 1024**3,
    "rss": 4 * 1024**3,
    "seconds": 7200,
    "disk_floor": 10 * 1024**3,
}


def read(path):
    return json.loads(Path(path).read_text())


def runtime():
    return {name: file_sha256(ROOT / name) for name in SOURCE_FILES}


@contextmanager
def guarded(action):
    path = FAMILY / "reproduce" / f"{action}-{time.time_ns()}.json"
    before = runtime()
    state = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "agent_model": "Codex / GPT-6",
        "argv": sys.argv,
        "runtime_hashes": before,
        "limits": LIMITS,
        "torch": str(torch.__version__),
        "python": sys.version,
        "device": "cpu",
        "requested_modal_gpu": None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "code_sha256": file_sha256(__file__),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTORCH_ENABLE_MPS_FALLBACK",
            )
        },
        "startup_ram": psutil.virtual_memory().available,
    }
    atomic_json(path, state)
    started = time.monotonic()
    stop = threading.Event()
    violation = [None]
    peak = [0]
    process = psutil.Process()

    def fail(_signal, _frame):
        raise RuntimeError("resource guard: " + str(violation[0]))

    def monitor():
        while not stop.wait(0.25):
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
                if peak[0] > LIMITS["rss"]:
                    violation[0] = "rss"
                elif psutil.virtual_memory().available < LIMITS["ram_floor"]:
                    violation[0] = "available_ram"
                elif shutil.disk_usage(FAMILY).free < LIMITS["disk_floor"]:
                    violation[0] = "disk_floor"
                elif time.monotonic() - started > LIMITS["seconds"]:
                    violation[0] = "runtime"
            except Exception as error:
                violation[0] = repr(error)
            if violation[0]:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    handler = signal.signal(signal.SIGUSR1, fail)
    worker = threading.Thread(target=monitor, daemon=True)
    try:
        assert state["startup_ram"] >= LIMITS["startup_ram"], "startup RAM reserve"
        assert shutil.disk_usage(FAMILY).free >= 15 * 1024**3, "startup disk reserve"
        worker.start()
        yield
        assert runtime() == before
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        stop.set()
        if worker.ident is not None:
            worker.join()
        signal.signal(signal.SIGUSR1, handler)
        state.update(
            elapsed_seconds=time.monotonic() - started,
            peak_polled_rss_bytes=peak[0],
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        atomic_json(path, state)


def prepare():
    assert read(FAMILY / "results/independent-cohort-verification.json")["status"] == "passed"
    destination = FAMILY / "reproduce/native-launch.json"
    if destination.exists():
        raise FileExistsError(destination)
    cohort = read(FAMILY / "results/cohort.json")
    assert cohort["native_final_steps"] == 2592
    parent, scales, _ = load_checkpoint(cohort["initialization"]["path"])
    config = asdict(parent.config)
    assert tuple(config["endpoints"]) == ("ic50",)
    config["endpoints"] = ["ic50", "elution_observation"]
    assert not any(
        config[name]
        for name in (
            "cell_types",
            "cytokines",
            "processing_pathways",
            "interventions",
            "components",
        )
    )
    sources = [source for source in cohort["sources"] if source["partition"] != "external"]
    external = [source for source in cohort["sources"] if source["partition"] == "external"]
    null_sources = []
    null_dir = RAW / "null-source"
    null_dir.mkdir(parents=True, exist_ok=False)
    for source in sources:
        if source["partition"] != "train":
            null_sources.append(source)
            continue
        path = null_dir / Path(source["path"]).name
        with Path(source["path"]).open() as stream, path.open("x") as output:
            for line in stream:
                row = json.loads(line)
                positive = (
                    row["peptide"] if row["value"] == 1 else row["provenance"]["parent_peptide"]
                )
                flip = (
                    int(
                        hashlib.sha256(
                            ("presto-ms-label-null-v1|" + positive).encode()
                        ).hexdigest(),
                        16,
                    )
                    % 2
                )
                row["provenance"]["original_label"] = row["value"]
                row["provenance"]["diagnostic_label_policy"] = "paired_hash_coin_training_only"
                if flip:
                    row["value"] = 1.0 - row["value"]
                output.write(json.dumps(row, sort_keys=True) + "\n")
        null_sources.append(
            {
                **source,
                "path": str(path),
                "sha256": file_sha256(path),
                "label_policy": source["label_policy"] + "; training_only_paired_label_permutation",
            }
        )
    training = dict(
        batch_size=64,
        max_epochs=3,
        max_steps=2592,
        min_epochs=3,
        patience=100,
        learning_rate=3e-4,
        weight_decay=1e-4,
        identity_weight=0,
        checkpoint_every_steps=50,
    )
    manifests, admissions = {}, {}
    for arm, seed in ARMS.items():
        manifest = {
            "schema": cohort["schema"],
            "model": config,
            "pretrained": {**cohort["initialization"], "freeze": ["ic50"]},
            "training": {**training, "seed": seed},
            "sources": null_sources if arm == "permuted101" else sources,
        }
        path = FAMILY / "reproduce" / f"{arm}-manifest.json"
        atomic_json(path, manifest)
        index = ObservationIndex(path)
        _, actual_training, inherited_scales, admission = preflight(index)
        assert inherited_scales["ic50"] == scales["ic50"]
        assert actual_training["max_steps"] == 2592
        manifests[arm] = {"path": str(path), "sha256": file_sha256(path)}
        admissions[arm] = admission
    path = FAMILY / "reproduce/external-manifest.json"
    atomic_json(path, {"schema": cohort["schema"], "model": config, "sources": external})
    ObservationIndex(path)
    sources_pin = FAMILY / "reproduce/source"
    for name in ("native.py", "continue_native.py"):
        shutil.copyfile(FAMILY / "code" / name, sources_pin / name)
    state = {
        "status": "native_manifests_verified_not_fitted",
        "utc": datetime.now(timezone.utc).isoformat(),
        "agent_model": "Codex / GPT-6",
        "runtime_hashes": runtime(),
        "cohort_sha256": file_sha256(FAMILY / "results/cohort.json"),
        "cohort_verification_sha256": file_sha256(
            FAMILY / "results/independent-cohort-verification.json"
        ),
        "code_hashes": {
            name: file_sha256(FAMILY / "code" / name)
            for name in ("native.py", "continue_native.py")
        },
        "manifests": manifests,
        "admissions": admissions,
        "initialization": cohort["initialization"],
        "external_manifest": {"path": str(path), "sha256": file_sha256(path)},
        "parent_scales": {name: asdict(value) for name, value in scales.items()},
        "training": training,
        "seeds": ARMS,
    }
    atomic_json(destination, state)
    print(json.dumps({"status": state["status"], "steps_per_arm": 2592}), flush=True)


def index_for(arm):
    pin = read(FAMILY / "reproduce/native-launch.json")
    assert runtime() == pin["runtime_hashes"]
    for name, expected in pin["code_hashes"].items():
        assert file_sha256(FAMILY / "code" / name) == expected
    assert file_sha256(pin["initialization"]["path"]) == pin["initialization"]["sha256"]
    item = pin["manifests"][arm]
    assert file_sha256(item["path"]) == item["sha256"]
    return ObservationIndex(item["path"])


def fit(arm):
    result = fit_index(index_for(arm), RAW / arm, device="cpu")
    last = torch.load(RAW / arm / "last.pt", weights_only=True, map_location="cpu")
    assert last["metadata"]["steps"] == 2592
    assert result["coverage_gate"]["status"] == "passed"
    atomic_json(FAMILY / "results" / f"{arm}-fit.json", result)


def score(arm):
    index = index_for(arm)
    model, scales, _ = load_checkpoint(RAW / arm / "last.pt")
    pin = read(FAMILY / "reproduce/native-launch.json")
    parent, parent_scales, _ = load_checkpoint(pin["initialization"]["path"])
    dependencies = parameter_dependencies(model, "ic50")
    assert scales["ic50"] == parent_scales["ic50"]
    for name in dependencies:
        assert torch.equal(model.state_dict()[name], parent.state_dict()[name]), name
    parent.eval()
    model.eval()
    probes = [
        replace(index[i], endpoint="ic50", value=1.0, unit="nM")
        for i in index.partitions["validation:assay"][
            :: max(1, len(index.partitions["validation:assay"]) // 20)
        ][:20]
    ]
    with torch.no_grad():
        original = [parent_scales["ic50"].decode(float(x.value)) for x in parent(probes)]
        retained = [scales["ic50"].decode(float(x.value)) for x in model(probes)]
    probe_error = max(abs(left - right) for left, right in zip(original, retained, strict=True))
    assert probe_error <= 2e-5, "retained IC50 inference changed"
    ext = pin["external_manifest"]
    assert file_sha256(ext["path"]) == ext["sha256"]
    external = ObservationIndex(ext["path"])
    output_dir = FAMILY / "results" / arm
    output_dir.mkdir(exist_ok=False)
    metrics = {}
    for label, source_index, partition in (
        ("validation", index, "validation"),
        ("test", index, "test"),
        ("external", external, "test"),
    ):
        selected = source_index.partitions[partition + ":assay"]
        result, records = evaluate(model, source_index, selected, scales, batch_size=64)
        metrics[label] = result
        with gzip.open(output_dir / f"{label}-predictions.jsonl.gz", "wt") as stream:
            for record in records:
                value = record["prediction"]
                record["probability"] = (
                    1 / (1 + math.exp(-value))
                    if value >= 0
                    else math.exp(value) / (1 + math.exp(value))
                )
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    atomic_json(
        output_dir / "summary.json",
        {
            "checkpoint": str(RAW / arm / "last.pt"),
            "checkpoint_sha256": file_sha256(RAW / arm / "last.pt"),
            "fixed_steps": 2592,
            "metrics": metrics,
            "parent_ic50_tensors_equal": sorted(dependencies),
            "parent_scaler_equal": True,
            "fresh_ic50_probe_max_abs_log10nm": probe_error,
            "fresh_ic50_probes": [
                {"observation_id": row.observation_id, "parent": left, "retained": right}
                for row, left, right in zip(probes, original, retained, strict=True)
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "fit", "score"])
    parser.add_argument("arm", nargs="?", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with guarded(args.action + ("-" + args.arm if args.arm else "")):
        if args.action == "prepare":
            prepare()
        elif args.action == "fit":
            fit(args.arm)
        else:
            score(args.arm)


if __name__ == "__main__":
    main()
