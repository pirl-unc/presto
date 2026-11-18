"""Disconnect-resilient evaluation of the six recorded recovery calls."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
from launch_pathway import E, GPU, NAME, image, volume

app = modal.App("presto-shared-recovery-closeout-20261117")


@app.function(
    image=image, gpu=GPU, cpu=4, memory=65536, timeout=14400,
    volumes={"/volume": volume}, single_use_containers=True, retries=0,
)
def evaluate_completed(run):
    from evaluate_pathway import evaluate_run as selected
    from evaluate_terminal import evaluate_run as terminal
    from prepare import sha

    root = Path("/volume") / NAME
    fit = json.loads((root / "fits" / run / "receipt.json").read_text())
    assert fit["steps"] == 6000 and fit["coverage"] == "passed"
    results = {}
    for model, checkpoint, evaluator in (
        (run, "selected.pt", selected),
        (run + "-terminal", "last.pt", terminal),
    ):
        out = root / "evaluation" / model
        if out.exists():
            receipt = json.loads((out / "receipt.json").read_text())
            assert receipt["checkpoint_sha256"] == sha(root / "fits" / run / "raw" / checkpoint)
            assert not receipt["test_accessed"]
            results[model] = {"reused_completed": True, "checkpoint_sha256": receipt["checkpoint_sha256"]}
        else:
            results[model] = evaluator.local(run)
        print(json.dumps({"evaluation": model, "result": results[model]}), flush=True)
    return results


@app.function(
    image=image, cpu=1, memory=2048, timeout=43200,
    volumes={"/volume": volume}, retries=0,
)
def coordinate(calls):
    out = Path("/volume") / NAME / "recovery" / f"closeout-{time.time_ns()}.json"
    report = {"calls": calls, "outcomes": {}, "errors": {}, "status": "running"}
    out.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    print(json.dumps({"closeout": str(out), "runs": [r["run"] for r in calls]}), flush=True)

    def finish(item):
        result = modal.FunctionCall.from_id(item["call"]).get()
        assert result["coverage"] == "passed" and result["steps"] == 6000
        call = evaluate_completed.spawn(item["run"])
        return {"evaluation_call": call.object_id, "results": call.get()}

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        pending = {pool.submit(finish, item): item["run"] for item in calls}
        for future in as_completed(pending):
            run = pending[future]
            try:
                report["outcomes"][run] = future.result()
            except Exception as error:
                report["errors"][run] = repr(error)
            out.write_text(json.dumps(report, indent=2) + "\n")
            volume.commit()
    report["status"] = "failed" if report["errors"] else "completed"
    out.write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    if report["errors"]:
        raise RuntimeError(json.dumps(report["errors"]))
    return {"path": str(out), **report}


@app.local_entrypoint()
def main(calls_file: str):
    calls = json.loads(Path(calls_file).read_text())
    assert calls and len(calls) == len({r["run"] for r in calls})
    call = coordinate.spawn(calls)
    (E / "reproduce" / f"closeout-coordinator-call-{time.time_ns()}.json").write_text(
        json.dumps({"call": call.object_id, "inputs": calls}, indent=2) + "\n"
    )
    result = call.get()
    (E / "results/recovery-closeout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
