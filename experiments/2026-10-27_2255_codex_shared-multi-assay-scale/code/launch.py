"""H100 comparison: frozen source, common trainer, live metrics and complete artifact collection."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

FAMILY_ID = "2026-10-27_2255_codex_shared-multi-assay-scale"
if modal.is_local():
    FAMILY = Path(__file__).resolve().parents[1]
    ROOT = FAMILY.parents[1]
    UPLOAD = ROOT / "artifacts" / FAMILY_ID / "upload"
else:
    FAMILY = Path("/bundle")
    ROOT = Path("/runs")
    UPLOAD = Path("/bundle")
GPU = os.environ.get("PRESTO_MODAL_GPU", "H100!")
if GPU != "H100!":
    raise ValueError("this benchmark requires H100!")
VARIANTS = ("ic50-exact", "ic50-bounds", "multi-assay", "multi-assay-wide")
CONDITIONS = [f"{variant}-s{seed}" for seed in (42, 43) for variant in VARIANTS]
REQUIREMENTS = [
    "torch==2.14.1",
    "numpy==2.5.4",
    "pyyaml==6.0.3",
    "mhcgnomes==3.64.6",
    "mhcseqs==2.6.13",
    "tqdm==4.70.1",
    "psutil==7.2.2",
]
if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(*REQUIREMENTS)
        .env(
            {
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        .add_local_dir(UPLOAD / "runtimes", remote_path="/runtimes")
        .add_local_dir(UPLOAD / "inputs", remote_path="/input")
        .add_local_file(
            UPLOAD / "source-pins.json", remote_path="/bundle/source-pins.json"
        )
        .add_local_file(FAMILY / "code/worker.py", remote_path="/bundle/worker.py")
    )
else:
    image = modal.Image.debian_slim(python_version="3.12")

app = modal.App("presto-shared-multi-assay-scale-20261027")
volume = modal.Volume.from_name("presto-checkpoints", create_if_missing=True)
live = modal.Dict.from_name("presto-cross-class-live", create_if_missing=True)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot(path, condition):
    report = read(path / "report.json", {})
    return dict(
        condition=condition,
        status=read(path / "status.json", {"status": "starting"}),
        progress=read(path / "progress.json", {}),
        history=read(path / "raw/history.json", []),
        hardware=read(path / "hardware.json", {}),
        metrics={
            "validation": report.get("summary", {})
            .get("metrics", {})
            .get("validation", {})
        },
        transfer={
            key: {**value, "metrics": {"validation": value["metrics"]["validation"]}}
            for key, value in report.get("transfer", {}).items()
        },
        selected_epoch=report.get("summary", {}).get("selected_epoch"),
        updated_unix=time.time(),
    )


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    timeout=14400,
    startup_timeout=1200,
    retries=0,
    max_containers=2,
    volumes={"/runs": volume},
)
def run_condition(condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown frozen condition")
    destination = Path("/runs") / FAMILY_ID / condition
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "status.json").exists():
        raise ValueError("already attempted; no automatic retry")
    key = f"{FAMILY_ID}:{condition}"
    live.put(
        key,
        dict(
            condition=condition, status={"status": "starting"}, updated_unix=time.time()
        ),
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "/bundle/worker.py", condition, str(destination)],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    while process.poll() is None:
        live.put(key, snapshot(destination, condition))
        time.sleep(5)
    final = snapshot(destination, condition)
    if process.returncode:
        final["status"] = {
            "status": "failed",
            "error": f"worker exit {process.returncode}",
        }
        live.put(key, final)
        volume.commit()
        raise RuntimeError(f"{condition} failed with exit {process.returncode}")
    final["status"] = {"status": "packaging"}
    live.put(key, final)
    archive = destination.parent / f"{condition}.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(destination, arcname=condition)
    receipt = dict(
        condition=condition,
        archive=str(archive.relative_to("/runs")),
        sha256=sha(archive),
        bytes=archive.stat().st_size,
    )
    volume.commit()
    final.update(status={"status": "completed"}, artifact=receipt)
    live.put(key, final)
    return receipt


def collect(receipt):
    archive = (
        ROOT / "artifacts" / FAMILY_ID / "downloads" / Path(receipt["archive"]).name
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".partial")
    with temporary.open("wb") as stream:
        volume.read_file_into_fileobj(receipt["archive"], stream)
    assert sha(temporary) == receipt["sha256"]
    temporary.replace(archive)
    raw = ROOT / "artifacts" / FAMILY_ID / "runs"
    raw.mkdir(exist_ok=True)
    with tarfile.open(archive) as stream:
        stream.extractall(raw, filter="data")
    condition = receipt["condition"]
    source = raw / condition
    dest = FAMILY / "results" / condition
    dest.mkdir(exist_ok=True)
    for p in source.glob("*.json*"):
        if p.is_file():
            shutil.copy2(p, dest / p.name)
    for name in (
        "summary.json",
        "history.json",
        "resolved-config.json",
        "coverage.json",
        "admission.json",
    ):
        shutil.copy2(source / "raw" / name, dest / name)
    report = read(dest / "report.json")
    report["remote_checkpoint"] = report["checkpoint"]
    report["checkpoint"] = str(source / "raw/selected.pt")
    assert sha(report["checkpoint"]) == report["checkpoint_sha256"]
    report["archive"] = dict(
        **receipt, local_archive=str(archive), raw_directory=str(source)
    )
    atomic(dest / "report.json", report)
    return report


@app.local_entrypoint()
def main():
    invocation = FAMILY / "reproduce/launch.json"
    if invocation.exists():
        raise ValueError("already launched; use saved call IDs for collection")
    pins = {p.name: sha(p) for p in (FAMILY / "code").glob("*.py")}
    for p in (FAMILY / "code").glob("*.py"):
        shutil.copy2(p, FAMILY / "reproduce/source" / p.name)
    atomic(
        invocation,
        dict(
            agent_model="Codex / GPT-6",
            argv=["modal", "run", "--detach", str(Path(__file__).resolve())],
            cwd=str(ROOT),
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            git_dirty=bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            source_pins=pins,
            transport_metadata_sha256=sha(UPLOAD / "source-pins.json"),
            conditions=CONDITIONS,
            requirements=REQUIREMENTS,
            requested_gpu=GPU,
            cpu=4,
            memory_mib=32768,
            max_containers=2,
            timeout_seconds=14400,
            environment={"PRESTO_MODAL_GPU": GPU},
            started_unix=time.time(),
        ),
    )
    (FAMILY / "reproduce/launch.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        + f'cd "{ROOT}"\n'
        + "PRESTO_MODAL_GPU=H100! /Users/iskander/code/shared-virtual-env/bin/modal "
        + f'run --detach "{Path(__file__).resolve()}"\n'
    )
    calls = {condition: run_condition.spawn(condition) for condition in CONDITIONS}
    atomic(
        FAMILY / "reproduce/call-ids.json",
        {key: call.object_id for key, call in calls.items()},
    )
    pending = set(calls)
    states = {}
    done = []
    failed = []
    while pending:
        for condition in CONDITIONS:
            if condition not in pending:
                continue
            state = live.get(f"{FAMILY_ID}:{condition}", None)
            if state:
                states[condition] = state
            try:
                receipt = calls[condition].get(timeout=0.1)
            except TimeoutError:
                continue
            except Exception as exc:
                failed.append(dict(condition=condition, error=repr(exc)))
                states[condition] = {
                    "condition": condition,
                    "status": {"status": "failed", "error": repr(exc)},
                }
                pending.remove(condition)
                continue
            report = collect(receipt)
            done.append(condition)
            pending.remove(condition)
            print(
                json.dumps(
                    dict(
                        condition=condition,
                        status="downloaded",
                        validation=report["summary"]["metrics"]["validation"],
                    )
                ),
                flush=True,
            )
        atomic(
            FAMILY / "results/live-status.json",
            dict(
                conditions=states,
                completed=done,
                failed=failed,
                updated_unix=time.time(),
            ),
        )
        atomic(
            FAMILY / "reproduce/controller.json",
            dict(
                status="running" if pending else "failed" if failed else "completed",
                completed=done,
                failed=failed,
                pending=sorted(pending),
            ),
        )
        if pending:
            time.sleep(5)
    if failed:
        raise RuntimeError("some conditions failed; see preserved receipts")
