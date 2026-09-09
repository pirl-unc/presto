"""Freeze, upload and execute registered coverage conditions on Modal CPU."""

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
FAMILY = EXPERIMENT.name
RAW = ROOT / "artifacts" / FAMILY


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def archive_paths(tracked):
    """Ship executable source and required package resources, never raw corpora."""
    package_roots = {"cli", "data", "models", "training", "scripts"}
    required = {
        "__init__.py",
        "__main__.py",
        "pyproject.toml",
        "README.md",
        "data/b2m_sequences.csv",
    }
    prefix = f"experiments/{FAMILY}/"
    experiment_files = {"conditions.json", "input_manifest.json", "reproduce/environment.txt"}
    return sorted(
        name
        for name in tracked
        if name in required
        or (Path(name).parts[0] in package_roots and name.endswith(".py"))
        or (
            name.startswith(prefix)
            and (
                name.removeprefix(prefix) in experiment_files
                or name.removeprefix(prefix).startswith("code/")
                and name.endswith(".py")
            )
        )
    )


def prepare():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("Freeze only a clean committed source tree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    target = RAW / "prepared" / commit
    target.mkdir(parents=True, exist_ok=False)
    archive = target / "source.tar"
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True
    ).splitlines()
    paths = archive_paths(tracked)
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive), commit, *paths],
        cwd=ROOT,
        check=True,
    )
    snapshot = target / "presto"
    snapshot.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(snapshot, filter="data")
    files = [
        {"relative_path": str(path.relative_to(snapshot)), "sha256": file_hash(path)}
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    ]
    receipt = {
        "git_commit": commit,
        "dirty": False,
        "files": files,
        "archive_sha256": file_hash(archive),
        "local_snapshot": str(snapshot),
    }
    write_json(snapshot / "source_receipt.json", receipt)
    write_json(target / "receipt.json", receipt)
    print(snapshot, flush=True)


def modal_destination():
    """Verify the pinned client's actual workspace without logging credentials."""
    import modal
    from modal.config import _lookup_workspace, _profile, config

    if modal.__version__ != "1.1.4":
        raise RuntimeError("This frozen launcher requires Modal 1.1.4")
    if _profile != "iskandr":
        raise RuntimeError("Launch with MODAL_PROFILE=iskandr for the registered destination")
    # The pinned client's own profile command uses this read-only lookup.
    workspace = asyncio.run(
        _lookup_workspace(config["server_url"], config["token_id"], config["token_secret"])
    ).username
    if workspace != "iskandr":
        raise RuntimeError(f"Authenticated workspace {workspace!r} differs from 'iskandr'")
    return modal, {"profile": _profile, "workspace": workspace, "environment": "main"}


def upload():
    modal, destination = modal_destination()

    manifest = json.loads((EXPERIMENT / "input_manifest.json").read_text())
    for entry in manifest:
        if file_hash(entry["local_path"]) != entry["sha256"]:
            raise RuntimeError(f"Frozen local input changed: {entry['local_path']}")
    volume = modal.Volume.from_name("presto-data", environment_name=destination["environment"])
    with volume.batch_upload() as batch:
        for entry in manifest:
            destination = str(Path(entry["remote_path"]).relative_to("/inputs"))
            batch.put_file(entry["local_path"], destination)
    for entry in manifest:
        if file_hash(entry["local_path"]) != entry["sha256"]:
            raise RuntimeError(f"Input changed during upload: {entry['local_path']}")
    write_json(
        RAW / "upload.json",
        {
            "volume": "presto-data",
            "files": manifest,
            "modal_version": modal.__version__,
            "destination": destination,
        },
    )
    print("Uploaded frozen inputs without overwriting existing objects", flush=True)


def remote_census(condition):
    """Global function so Modal can serialize its implementation and durable call."""
    import importlib.util
    import modal
    from contextlib import redirect_stderr, redirect_stdout

    family = "2026-09-09_1541_codex_canonical-coverage"
    worker = Path("/opt/presto/experiments") / family / "code/census.py"
    logs = Path("/results") / family / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output_volume = modal.Volume.from_name("presto-checkpoints", environment_name="main")
    try:
        with (logs / f"{condition}.log").open("x", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                spec = importlib.util.spec_from_file_location("canonical_census_worker", worker)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.run(condition)
    finally:
        output_volume.commit()
    return {
        "condition": condition,
        "output_volume": "presto-checkpoints",
        "output_path": f"{family}/{condition}",
    }


def execute(condition, snapshot):
    snapshot = snapshot.resolve()
    receipt = json.loads((snapshot / "source_receipt.json").read_text())
    for entry in receipt["files"]:
        if file_hash(snapshot / entry["relative_path"]) != entry["sha256"]:
            raise RuntimeError(f"Snapshot changed: {entry['relative_path']}")
    launcher_path = f"experiments/{FAMILY}/code/launch.py"
    launcher_entries = [
        entry for entry in receipt["files"] if entry["relative_path"] == launcher_path
    ]
    if len(launcher_entries) != 1:
        raise RuntimeError("Frozen receipt requires exactly one executing launcher entry")
    if file_hash(Path(__file__).resolve()) != launcher_entries[0]["sha256"]:
        raise RuntimeError("Executing launcher differs from frozen source; prepare a new snapshot")
    modal, destination = modal_destination()
    frozen_experiment = snapshot / "experiments" / FAMILY
    configs = json.loads((frozen_experiment / "conditions.json").read_text())
    if condition not in configs:
        raise ValueError(f"Unregistered condition {condition}")
    local_result = EXPERIMENT / "results" / condition
    local_result.mkdir(parents=True, exist_ok=False)
    bundle = local_result / "reproduce"
    bundle.mkdir()
    for name in ("launch.py", "census.py"):
        shutil.copy2(frozen_experiment / "code" / name, bundle / name)
    invocation = {
        "argv": sys.argv,
        "cwd": str(ROOT),
        "condition": condition,
        "source_receipt": receipt,
        "args": configs[condition],
        "modal_version": modal.__version__,
        "destination": destination,
        "status": "preparing_image",
    }
    write_json(bundle / "launch.json", invocation)
    (bundle / "launch.sh").write_text(
        "#!/bin/sh\nset -eu\ncd "
        + shlex.quote(str(ROOT))
        + "\nexec "
        + shlex.join([sys.executable, *sys.argv])
        + "\n"
    )
    prefix = f"/inputs/{FAMILY}"
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install_from_requirements(str(frozen_experiment / "reproduce/environment.txt"))
        .pip_install("modal==1.1.4")
        .add_local_dir(str(snapshot), "/opt/presto", copy=True)
        .run_commands("python -m pip install --no-deps -e /opt/presto")
        .env(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
                "TQDM_DISABLE": "1",
                "PRESTO_MHCSEQS_SEARCH_DIR": prefix + "/mhcseqs",
                "HITLIST_DATA_DIR": prefix + "/hitlist",
            }
        )
    )
    app = modal.App("presto-codex-canonical-coverage")
    remote = app.function(
        image=image,
        cpu=(4, 8),
        memory=(65536, 196608),
        timeout=14400,
        volumes={
            "/inputs": modal.Volume.from_name(
                "presto-data", environment_name=destination["environment"]
            ),
            "/results": modal.Volume.from_name(
                "presto-checkpoints", environment_name=destination["environment"]
            ),
        },
    )(remote_census)
    try:
        with (
            modal.enable_output(),
            app.run(detach=True, environment_name=destination["environment"]),
        ):
            call = remote.spawn(condition)
            invocation.update(status="running", app_id=app.app_id, call_id=call.object_id)
            write_json(local_result / "handle.json", invocation)
            print(json.dumps({"app_id": app.app_id, "call_id": call.object_id}), flush=True)
            result = call.get()
            invocation.update(status="remote_complete", result=result)
    except BaseException as exc:
        invocation.update(status="launch_or_remote_failed", error=repr(exc))
        raise
    finally:
        write_json(local_result / "handle.json", invocation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "upload", "execute"))
    parser.add_argument("--condition", default="merged_measured")
    parser.add_argument("--snapshot", type=Path)
    options = parser.parse_args()
    os.chdir(ROOT)
    if options.action == "prepare":
        prepare()
    elif options.action == "upload":
        upload()
    elif options.snapshot is None:
        parser.error("execute requires --snapshot from a clean prepare receipt")
    else:
        execute(options.condition, options.snapshot)
