"""Registered canonical preflight; complete reusable evidence lives in datacache."""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import time
import traceback

import torch
import datacache
import hitlist
from datacache import materialize, validate_file
from hitlist.downloads import data_dir
import presto
from presto.scripts.train_iedb import run


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    family = Path(__file__).resolve().parents[1]
    repository = family.parents[1]
    config_path = family / "reproduce/launch.json"
    config = json.loads(config_path.read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True))
    if dirty:
        raise RuntimeError("Commit the registered launch and code before running this experiment")
    options = {
        "stage": "preflight", "commit": commit, "dirty": dirty,
        "presto_version": presto.__version__, "recipe_sha256": sha(__file__),
        "python": sys.version, "platform": platform.platform(),
        "versions": {
            **{name: importlib.metadata.version(name) for name in ("torch", "pandas", "pyarrow")},
            "hitlist_runtime": hitlist.__version__, "datacache_runtime": datacache.__version__,
        },
        "dependency_imports": {"hitlist": hitlist.__file__, "datacache": datacache.__file__},
        "torch_threads": 1, "gpu": "none", "wall_clock_limit_seconds": 1800,
    }
    key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(1)

    def builder(inputs, outputs):
        with TemporaryDirectory(prefix="presto-scoped-preflight-") as temporary:
            work = Path(temporary)
            started = datetime.now(timezone.utc).isoformat()
            before = time.monotonic()
            status = "success"

            def timeout(signum, frame):
                raise TimeoutError("Canonical preflight exceeded its 30-minute limit")

            handler = signal.signal(signal.SIGALRM, timeout)
            signal.alarm(1800)
            try:
                with (work / "training.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    run(argparse.Namespace(**config, run_dir=str(work), checkpoint=str(work / "checkpoint.pt")))
            except Exception:
                status = "failed"
                (work / "error.txt").write_text(traceback.format_exc())
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, handler)
            elapsed = time.monotonic() - before
            shutil.copyfile(inputs["launch.json"], work / "launch.json")
            membership = {}
            funnel_path = work / "data_funnel.json"
            if funnel_path.exists():
                funnel = json.loads(funnel_path.read_text())
                membership = funnel.get("frozen_membership", {})
                if membership:
                    destination = work / "membership"
                    destination.mkdir()
                    for name, pin in membership["artifacts"].items():
                        source = Path(membership["directory"]) / name
                        validate_file(source, pin["sha256"])
                        shutil.copyfile(source, destination / name)
            inventory = {
                path.relative_to(work).as_posix(): {"sha256": sha(path), "size_bytes": path.stat().st_size}
                for path in sorted(work.rglob("*")) if path.is_file()
            }
            Path(outputs["inventory.json"]).write_text(json.dumps(inventory, indent=2) + "\n")
            with tarfile.open(outputs["run.tar.gz"], "w:gz") as archive:
                for name in inventory:
                    archive.add(work / name, arcname=name, recursive=False)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            summary = {
                **options, "status": status, "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": elapsed,
                "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
                "rss_scope": "runner process only; child processes excluded", "config": config,
                "membership": membership,
                "artifacts": {name: {"sha256": sha(outputs[name]), "size_bytes": Path(outputs[name]).stat().st_size}
                              for name in ("inventory.json", "run.tar.gz")},
            }
            Path(outputs["summary.json"]).write_text(json.dumps(summary, indent=2) + "\n")

    paths = materialize(
        data_dir() / "presto-scoped-qualification" / "preflight" / key,
        {"launch.json": {"path": str(config_path), "sha256": sha(config_path)}},
        transform={"version": "canonical-scoped-preflight-v1", "options": options},
        outputs={name: {} for name in ("run.tar.gz", "inventory.json", "summary.json")},
        builder=builder,
    )
    summary = json.loads(Path(paths["summary.json"]).read_text())
    result = {"directory": str(Path(paths["summary.json"]).parent), "summary_sha256": sha(paths["summary.json"]), **summary}
    print(json.dumps(result, indent=2), flush=True)
    if summary["status"] != "success":
        raise RuntimeError(f"Canonical preflight failed; preserved evidence: {result['directory']}")


if __name__ == "__main__":
    main()
