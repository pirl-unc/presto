"""Bounded canonical IC50 fit with the unchanged prospective support gate."""

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

from common import (
    CACHE,
    HITLIST_CACHE,
    ROOT,
    SOURCE,
    build_mhc_catalog,
    digest,
    write_json,
)


def preserve_frozen(work, declaration):
    import datacache

    funnel = json.loads((work / "data_funnel.json").read_text())
    expected = declaration["source_contract"]["frozen_membership"]
    actual = funnel["frozen_membership"]
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError("Frozen membership changed: " + key)
    for kind in ("frozen_assignments", "frozen_membership"):
        receipt = funnel[kind]
        folder = work / kind
        folder.mkdir()
        for name, pin in receipt["artifacts"].items():
            source = Path(receipt["directory"]) / name
            if digest(source) != pin:
                raise ValueError("Frozen artifact changed: " + name)
            datacache.validate_file(source, pin["sha256"])
            shutil.copyfile(source, folder / name)
    return {key: actual[key] for key in expected}


def main():
    os.environ["HITLIST_DATA_DIR"] = str(HITLIST_CACHE)
    os.environ["MHCSEQS_DATA"] = str(CACHE / "mhcseqs-unused-default")
    os.environ["XDG_CACHE_HOME"] = str(CACHE / "xdg")
    import datacache
    import torch
    from baselines import compare, expected_rows
    from diagnostic import install
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
    pins = json.loads((ROOT / "frozen-inputs.json").read_text())
    options = {
        **metadata,
        "config": config,
        "torch_threads": 1,
        "wall_clock_limit_seconds": 5400,
        "python": sys.version,
        "platform": platform.platform(),
        "frozen_input_pins": pins,
    }
    key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()

    def builder(inputs, outputs):
        started, before = datetime.now(timezone.utc).isoformat(), time.monotonic()
        with tempfile.TemporaryDirectory(prefix="corrected-human-fit-") as tmp:
            work = Path(tmp)
            status, catalog, result = "success", None, {}

            def timeout(signum, frame):
                raise TimeoutError("Human IC50 fitting diagnostic exceeded 90 minutes")

            handler = signal.signal(signal.SIGALRM, timeout)
            signal.alarm(5400)
            try:
                with (work / "training.log").open("w") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        frozen = datacache.install_bundle(
                            CACHE / "frozen-inputs", pins, download_options={"show_progress": False}
                        )
                        for name, source in frozen.items():
                            shutil.copyfile(source, work / name)
                        declaration = json.loads(
                            (work / "prospective-supported-outputs.json").read_text()
                        )
                        cohort = expected_rows(
                            work / "membership.tsv", work / "selected-source-contexts.tsv"
                        )
                        train_ids = cohort[cohort.source_partition == "train"].sample_id.tolist()
                        catalog = build_mhc_catalog()
                        os.environ["PRESTO_MHCSEQS_SEARCH_DIR"] = catalog["directory"]
                        actual_config = {
                            **config,
                            "hitlist_bundle": str(SOURCE),
                            "supported_output_manifest": str(
                                work / "prospective-supported-outputs.json"
                            ),
                            "run_dir": str(work),
                            "checkpoint": str(work / "checkpoint.pt"),
                        }
                        write_json(work / "effective-launch.json", actual_config)
                        finish_diagnostic = install(train_iedb, work, train_ids, actual_config)
                        # Canonical support gate runs before the canonical model constructor.
                        # No rebindings, bypasses, or alternate training implementation.
                        train_iedb.run(argparse.Namespace(**actual_config))
                        result["frozen_membership"] = preserve_frozen(work, declaration)
                        result["checkpoint"] = finish_diagnostic()
                        result.update(
                            compare(
                                work, work / "membership.tsv", work / "selected-source-contexts.tsv"
                            )
                        )
                        write_json(work / "qualification-result.json", result)
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
            check = subprocess.run(
                [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
            )
            (work / "pip-check.txt").write_text(check.stdout + check.stderr)
            if check.returncode:
                status = "failed"
            if catalog:
                write_json(work / "mhc-catalog.json", catalog)
                for name, pin in catalog["artifacts"].items():
                    source = Path(catalog["directory"]) / name
                    if digest(source) != pin:
                        raise ValueError("Catalog bytes changed after verification: " + name)
                    destination = work / "mhc-catalog" / name
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source, destination)
            inventory = {
                p.relative_to(work).as_posix(): digest(p)
                for p in sorted(work.rglob("*"))
                if p.is_file()
            }
            write_json(outputs["inventory.json"], inventory)
            with tarfile.open(outputs["run.tar.gz"], "w:gz") as archive:
                for name in inventory:
                    archive.add(work / name, arcname=name, recursive=False)
            summary = {
                **options,
                "status": status,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": runtime,
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
                "rss_scope": "runner process only, children excluded",
                "catalog": catalog,
                "versions": versions,
                "result": result,
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
            "producer.json",
            "frozen-inputs.json",
            "fit.py",
            "baselines.py",
            "diagnostic.py",
            "common.py",
        )
    }
    paths = datacache.materialize(
        CACHE / "fit" / key,
        inputs,
        transform={"version": "corrected-human-ic50-fit-v1", "options": options},
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
