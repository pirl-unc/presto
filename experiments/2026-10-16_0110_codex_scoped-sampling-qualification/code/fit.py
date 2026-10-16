"""Prospective two-condition sampling comparison; repeated test is exploratory."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/opt/qualification")
CACHE = Path("/data/presto-qualification/scoped-sampling-v1")


def digest(path):
    path = Path(path)
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": value, "size_bytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def preserve_membership(directory):
    from datacache import validate_file

    funnel = json.loads((directory / "data_funnel.json").read_text())
    receipt = funnel["frozen_membership"]
    target = directory / "membership"
    target.mkdir()
    for name, pin in receipt["artifacts"].items():
        source = Path(receipt["directory"]) / name
        validate_file(source, pin["sha256"])
        shutil.copyfile(source, target / name)
    return receipt


def main():
    os.environ["HITLIST_DATA_DIR"] = str(CACHE / "hitlist")
    os.environ["XDG_CACHE_HOME"] = str(CACHE / "cache")
    import datacache
    import torch
    from baselines import compare_baselines
    from presto.scripts.train_iedb import run

    torch.set_num_threads(1)
    metadata = json.loads((ROOT / "invocation.json").read_text())
    declaration = json.loads((ROOT / "launch.json").read_text())
    options = {
        **metadata,
        "declaration": declaration,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_threads": 1,
        "work_limit_seconds": 3300,
        "repeated_test_status": "exploratory",
    }
    key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()

    def builder(inputs, outputs):
        started = datetime.now(timezone.utc).isoformat()
        before = time.monotonic()
        status, results = "success", []
        with tempfile.TemporaryDirectory(prefix="scoped-sampling-") as tmp:
            work = Path(tmp)

            def timeout(signum, frame):
                raise TimeoutError("Sampling comparison exceeded 55 minutes")

            handler = signal.signal(signal.SIGALRM, timeout)
            signal.alarm(3300)
            try:
                preflight = work / "preflight"
                preflight.mkdir()
                preflight_config = {
                    **declaration["common"],
                    "data_preflight_only": True,
                    "supported_output_manifest": str(inputs["supported_outputs.json"]),
                    "run_dir": str(preflight),
                    "checkpoint": str(preflight / "checkpoint.pt"),
                }
                with (preflight / "training.log").open("w") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        run(argparse.Namespace(**preflight_config))
                frozen = preserve_membership(preflight)
                for condition in declaration["conditions"]:
                    directory = work / condition["name"]
                    directory.mkdir()
                    condition_started = time.monotonic()
                    config = {
                        **declaration["common"],
                        "balanced_batches": condition["balanced_batches"],
                        "supported_output_manifest": str(inputs["supported_outputs.json"]),
                        "run_dir": str(directory),
                        "checkpoint": str(directory / "checkpoint.pt"),
                    }
                    with (directory / "training.log").open("w") as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            run(argparse.Namespace(**config))
                    membership = preserve_membership(directory)
                    if membership["artifacts"] != frozen["artifacts"]:
                        raise ValueError("Frozen membership changed across conditions")
                    comparison = compare_baselines(directory)
                    results.append(
                        {
                            **condition,
                            "comparison": comparison,
                            "runtime_seconds": time.monotonic() - condition_started,
                        }
                    )
                # Fixed prospective selection rule; test scores never enter this expression.
                selected = min(
                    results,
                    key=lambda row: next(
                        score["log_loss"]
                        for score in row["comparison"]
                        if score["split"] == "val" and score["predictor"] == "presto"
                    ),
                )["name"]
                write_json(
                    work / "selection.json",
                    {
                        "selected_condition": selected,
                        "criterion": "minimum unweighted validation log loss",
                        "test_status": "exploratory; already inspected in the original study",
                        "training_readiness": "not established by within-search comparisons",
                    },
                )
            except Exception:
                status = "failed"
                (work / "error.txt").write_text(traceback.format_exc())
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, handler)
            runtime = time.monotonic() - before
            for name, path in inputs.items():
                shutil.copyfile(path, work / name)
            shutil.copyfile(ROOT / "consumer-install.json", work / "consumer-install.json")
            versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
            write_json(work / "versions.json", versions)
            check = subprocess.run(
                [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
            )
            (work / "pip-check.txt").write_text(check.stdout + check.stderr)
            if check.returncode:
                status = "failed"
            write_json(work / "comparison.json", results)
            inventory = {
                p.relative_to(work).as_posix(): digest(p)
                for p in sorted(work.rglob("*"))
                if p.is_file()
            }
            write_json(outputs["inventory.json"], inventory)
            with tarfile.open(outputs["run.tar.gz"], "w:gz") as archive:
                for name in inventory:
                    archive.add(work / name, arcname=name, recursive=False)
            write_json(
                outputs["summary.json"],
                {
                    **options,
                    "status": status,
                    "started_utc": started,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "runtime_seconds": runtime,
                    "comparison": results,
                    "versions": versions,
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    "rss_scope": "runner process only, children excluded",
                    "artifacts": {
                        name: digest(outputs[name]) for name in ("run.tar.gz", "inventory.json")
                    },
                },
            )

    inputs = {
        name: {"path": str(ROOT / name), "sha256": digest(ROOT / name)["sha256"]}
        for name in (
            "launch.json",
            "invocation.json",
            "supported_outputs.json",
            "fit.py",
            "baselines.py",
        )
    }
    paths = datacache.materialize(
        CACHE / "runs" / key,
        inputs,
        transform={"version": "scoped-sampling-comparison-v1", "options": options},
        outputs={name: {} for name in ("run.tar.gz", "inventory.json", "summary.json")},
        builder=builder,
    )
    receipt = {
        "summary": json.loads(Path(paths["summary.json"]).read_text()),
        "files": {name: {"path": str(path), **digest(path)} for name, path in paths.items()},
    }
    write_json(ROOT / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
