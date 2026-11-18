"""Recover interrupted runs with their original canonical runtime and manifests."""

import json
import shutil
import time
from pathlib import Path

import modal
from launch import E, GPU, NAME, base_image, volume

original_image = base_image
pathway_image = base_image
if modal.is_local():
    original_image = base_image.add_local_dir(
        E / "reproduce/runtime", "/runtime/presto"
    ).add_local_dir(E / "code", "/bundle")
    pathway_image = base_image.add_local_dir(
        E / "reproduce/runtime-pathway", "/runtime/presto"
    ).add_local_dir(E / "code", "/bundle")
app = modal.App("presto-shared-checkpoint-recovery-20261117")


def inspect_one(root, run):
    import torch
    from prepare import sha

    out = root / "fits" / run
    report = {"run": run, "completed": (out / "receipt.json").exists()}
    if report["completed"]:
        report["receipt"] = json.loads((out / "receipt.json").read_text())
        return report
    snapshot = out / "raw/resume.pt"
    payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    state = payload["metadata"]["resume"]["state"]
    staging = json.loads((out / "source-staging.json").read_text())
    assert sha(out / "manifest.json") == staging["original_manifest_sha256"]
    assert sha(out / "execution-manifest.json") == staging["execution_manifest_sha256"]
    report.update(
        snapshot_sha256=sha(snapshot),
        snapshot_bytes=snapshot.stat().st_size,
        manifest_sha256=staging["original_manifest_sha256"],
        execution_manifest_sha256=staging["execution_manifest_sha256"],
        state={
            k: state[k]
            for k in (
                "steps",
                "position",
                "epoch",
                "phase",
                "last_epoch",
                "best",
                "best_epoch",
                "elapsed_seconds",
                "validation_pending",
            )
            if k in state
        },
    )
    return report


@app.function(image=pathway_image, cpu=2, memory=16384, timeout=3600, volumes={"/volume": volume})
def inspect_runs(runs):
    root = Path("/volume") / NAME
    results = [inspect_one(root, run) for run in runs]
    out = root / "recovery"
    out.mkdir(exist_ok=True)
    path = out / f"inspection-{time.time_ns()}.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    volume.commit()
    return {"path": str(path), "runs": results}


def recover(run, runtime, expected_snapshot_sha):
    import torch
    from fit import fit

    root = Path("/volume") / NAME
    out = root / "fits" / run
    before = inspect_one(root, run)
    if before["completed"]:
        return before["receipt"]
    assert before["snapshot_sha256"] == expected_snapshot_sha
    arm, rest = run.rsplit("-s", 1)
    seed, steps = map(int, rest.split("-n"))
    assert ("-sqrt" in arm) == (runtime == "5f9bfe37")
    attempt = out / "recovery" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    for name in ("resume.pt", "selected.pt", "history.json"):
        if (out / "raw" / name).exists():
            shutil.copyfile(out / "raw" / name, attempt / name)
    report = {
        "before": before,
        "runtime": runtime,
        "attempt": str(attempt),
        "status": "resuming",
        "hardware": {
            "requested_gpu": GPU,
            "name": torch.cuda.get_device_name(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
        },
    }
    (attempt / "before.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    try:
        result = fit(root, out, arm, seed, steps, commit=volume.commit, resume_run=True)
        report.update(status="completed", completed_receipt=result)
        return result
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (attempt / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()


@app.function(
    image=original_image,
    gpu=GPU,
    cpu=4,
    memory=65536,
    timeout=43200,
    volumes={"/volume": volume},
    retries=0,
    single_use_containers=True,
)
def resume_original(run, expected_snapshot_sha):
    return recover(run, "2da11504", expected_snapshot_sha)


@app.function(
    image=pathway_image,
    gpu=GPU,
    cpu=4,
    memory=65536,
    timeout=43200,
    volumes={"/volume": volume},
    retries=0,
    single_use_containers=True,
)
def resume_pathway(run, expected_snapshot_sha):
    return recover(run, "5f9bfe37", expected_snapshot_sha)


@app.function(
    image=pathway_image, cpu=1, memory=2048, timeout=43200,
    volumes={"/volume": volume}, retries=0,
)
def coordinate(names, inspected):
    """Keep ownership of every worker remote when the local client disconnects."""
    path = Path("/volume") / NAME / "recovery" / f"coordinator-{time.time_ns()}.json"
    report = {"calls": [], "results": {}, "errors": {}, "status": "running"}
    calls = [
        (
            run,
            (resume_pathway if "-sqrt-" in run else resume_original).spawn(
                run, inspected[run]["snapshot_sha256"]
            ),
        )
        for run in names
        if not inspected[run]["completed"]
    ]
    report["calls"] = [{"run": run, "call": call.object_id} for run, call in calls]
    path.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print(json.dumps({"coordinator": str(path), "calls": report["calls"]}), flush=True)
    for run, call in calls:
        try:
            report["results"][run] = call.get()
        except Exception as error:
            report["errors"][run] = repr(error)
        path.write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
    report["status"] = "failed" if report["errors"] else "completed"
    path.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    if report["errors"]:
        raise RuntimeError(json.dumps(report["errors"]))
    return {"path": str(path), **report}


@app.local_entrypoint()
def main(runs: str, action: str = "inspect"):
    names = runs.split(",")
    assert names and len(names) == len(set(names))
    if action == "inspect":
        result = inspect_runs.remote(names)
        (E / "results/recovery-inspection.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        assert action == "resume"
        inspected = {
            r["run"]: r
            for r in json.loads((E / "results/recovery-inspection.json").read_text())["runs"]
        }
        call = coordinate.spawn(names, inspected)
        (E / "reproduce" / f"recovery-coordinator-call-{time.time_ns()}.json").write_text(
            json.dumps({"call": call.object_id, "runs": names}, indent=2) + "\n"
        )
        result = call.get()
        (E / "results/recovery-outcomes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
