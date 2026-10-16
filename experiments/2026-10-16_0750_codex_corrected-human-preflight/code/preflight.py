"""One bounded, fully recorded canonical frozen-human-corpus preflight."""

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
SOURCE = Path(
    "/data/hitlist-qualification/runs/a6933c31825caf35475ba38ae243551bb815d65ba03ee93d9ddb768869e4b5a0/human-cohort/c923e5f8f05d1df7156625740b84f5a58f25108f703a01146a2b62336dc54443/artifacts/bundles/2026-10-16T07-29-43Z"
)
CACHE = Path("/data/presto-qualification/corrected-human-v1")
MHC_CACHE = Path("/data/presto-qualification/human-baseline-v1")


def digest(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            sha.update(block)
    return {"sha256": sha.hexdigest(), "size_bytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_mhc_catalog():
    import datacache
    import mhcseqs

    sources = json.loads((ROOT / "mhc-sources.json").read_text())
    files = datacache.install_bundle(
        MHC_CACHE / "mhc-sources",
        {name: {k: row[k] for k in ("url", "sha256", "size")} for name, row in sources.items()},
        download_options={"show_progress": False},
    )
    names = (
        "mhc-seqs-raw.csv",
        "mhc-full-seqs.csv",
        "mhc-merge-report.txt",
        "mhc-validation-report.txt",
    )

    def builder(inputs, outputs):
        # The existing verified FASTAs prevent mhcseqs.build from fetching mutable Latest URLs.
        with tempfile.TemporaryDirectory(prefix="mhc-catalog-") as tmp:
            work = Path(tmp)
            fasta = work / "fasta"
            fasta.mkdir()
            for name, path in inputs.items():
                shutil.copyfile(path, fasta / name)
            mhcseqs.build(output_dir=str(work), data_dir=str(fasta), force_download=False)
            for name in names:
                shutil.copyfile(work / name, outputs[name])

    outputs = datacache.materialize(
        MHC_CACHE / "mhc-catalog",
        {
            name: {"path": str(path), "sha256": sources[name]["sha256"]}
            for name, path in files.items()
        },
        transform={
            "version": "mhcseqs-2.6.13-canonical-build-v1",
            "options": {"mhcseqs": importlib.metadata.version("mhcseqs")},
        },
        outputs={name: {} for name in names},
        builder=builder,
    )
    expected = {
        "mhc-seqs-raw.csv": "4c61d2683c07f8aeee7845fb75038636fbb8d867adf5d55535d371ece16a94eb",
        "mhc-full-seqs.csv": "1d64f5af13f49497ea5c7219f8ec21edd1f973aecd3a6627a065fc554ef022c5",
        "mhc-merge-report.txt": "090f90ca4713cb55b22f43cfefb99e0d2246ebe2cbf2b5ac450393ad701964cb",
        "mhc-validation-report.txt": "349f7aeafdbaa1edfe2de53f8bd674810a1e4fecaa2c7356feeeabc15016ce39",
    }
    for name, path in outputs.items():
        datacache.validate_file(path, expected[name])
    return {
        "shared_catalog_root": str(MHC_CACHE),
        "directory": str(Path(outputs[names[0]]).parent),
        "artifacts": {name: digest(path) for name, path in outputs.items()},
        "sources": sources,
    }


def main():
    os.environ["HITLIST_DATA_DIR"] = str(CACHE / "hitlist")
    os.environ["MHCSEQS_DATA"] = str(CACHE / "mhcseqs-unused-default")
    os.environ["XDG_CACHE_HOME"] = str(CACHE / "xdg")
    import datacache
    import torch
    from presto.scripts import train_iedb

    producer = json.loads((ROOT / "producer.json").read_text())
    for package, version in (
        ("presto", producer["presto_version"]),
        ("hitlist", producer["hitlist_version"]),
        ("mhcseqs", "2.6.13"),
    ):
        if importlib.metadata.version(package) != version:
            raise ValueError("Installed producer differs from frozen contract: " + package)
    torch.set_num_threads(1)
    metadata = json.loads((ROOT / "invocation.json").read_text())
    config = json.loads((ROOT / "launch.json").read_text())
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    config["hitlist_bundle"] = str(SOURCE)
    options = {
        **metadata,
        "config": config,
        "gpu": "none",
        "torch_threads": 1,
        "wall_clock_limit_seconds": 2700,
        "python": sys.version,
        "platform": platform.platform(),
    }

    def builder(inputs, outputs):
        started = datetime.now(timezone.utc).isoformat()
        before = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="human-preflight-") as tmp:
            work = Path(tmp)
            status, catalog = "success", None

            def timeout(signum, frame):
                raise TimeoutError("Human-corpus preflight exceeded 45 minutes")

            def forbidden_model(*args, **kwargs):
                raise AssertionError("A preflight must not construct a model")

            handler = signal.signal(signal.SIGALRM, timeout)
            signal.alarm(2700)
            try:
                with (work / "training.log").open("w") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        catalog = build_mhc_catalog()
                        os.environ["PRESTO_MHCSEQS_SEARCH_DIR"] = catalog["directory"]
                        train_iedb.Presto = forbidden_model
                        train_iedb.run(
                            argparse.Namespace(
                                **config, run_dir=str(work), checkpoint=str(work / "checkpoint.pt")
                            )
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
            shutil.copyfile(SOURCE / "manifest.json", work / "source-manifest.json")
            shutil.copyfile(ROOT / "consumer-install.json", work / "consumer-install.json")
            versions = {
                dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
            }
            write_json(work / "versions.json", versions)
            pip_check = subprocess.run(
                [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
            )
            (work / "pip-check.txt").write_text(pip_check.stdout + pip_check.stderr)
            if pip_check.returncode:
                status = "failed"
            if catalog:
                write_json(work / "mhc-catalog.json", catalog)
                for name, pin in catalog["artifacts"].items():
                    source = Path(catalog["directory"]) / name
                    datacache.validate_file(source, pin["sha256"])
                    destination = work / "mhc-catalog" / name
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source, destination)
            funnel_path = work / "data_funnel.json"
            if funnel_path.exists():
                funnel = json.loads(funnel_path.read_text())
                for kind in ("frozen_assignments", "frozen_membership"):
                    receipt = funnel.get(kind, {})
                    if receipt:
                        folder = work / kind
                        folder.mkdir()
                        for name, pin in receipt["artifacts"].items():
                            source = Path(receipt["directory"]) / name
                            datacache.validate_file(source, pin["sha256"])
                            shutil.copyfile(source, folder / name)
            inventory = {
                path.relative_to(work).as_posix(): digest(path)
                for path in sorted(work.rglob("*"))
                if path.is_file()
            }
            write_json(outputs["inventory.json"], inventory)
            with tarfile.open(outputs["run.tar.gz"], "w:gz") as archive:
                for name in inventory:
                    archive.add(work / name, arcname=name, recursive=False)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            summary = {
                **options,
                "status": status,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": runtime,
                "process_peak_rss_bytes": int(rss * 1024),
                "rss_scope": "runner process only, children excluded",
                "catalog": catalog,
                "versions": versions,
                "artifacts": {
                    name: digest(outputs[name]) for name in ("run.tar.gz", "inventory.json")
                },
            }
            write_json(outputs["summary.json"], summary)

    inputs = {
        name: {"path": str(ROOT / name), "sha256": digest(ROOT / name)["sha256"]}
        for name in (
            "launch.json",
            "invocation.json",
            "mhc-sources.json",
            "preflight.py",
            "producer.json",
            "candidate-support.json",
        )
    }
    paths = datacache.materialize(
        CACHE / "preflight" / key,
        inputs,
        transform={"version": "corrected-human-corpus-preflight-v1", "options": options},
        outputs={name: {} for name in ("run.tar.gz", "inventory.json", "summary.json")},
        builder=builder,
    )
    summary = json.loads(Path(paths["summary.json"]).read_text())
    receipt = {
        "summary": summary,
        "files": {name: {"path": str(path), **digest(path)} for name, path in paths.items()},
    }
    write_json(ROOT / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
