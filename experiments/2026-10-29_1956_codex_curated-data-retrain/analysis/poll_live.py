"""Publish real worker progress alongside the immutable parent selected checkpoints."""

from pathlib import Path
import asyncio
import functools
import http.server
import json
import threading
import time
import modal

F = Path(__file__).resolve().parents[1]
P = F.parent / "2026-10-28_1540_codex_temporal-multi-output"
D = modal.Dict.from_name("presto-cross-class-live")
server = http.server.ThreadingHTTPServer(
    ("127.0.0.1", 54503),
    functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(F)),
)
threading.Thread(target=server.serve_forever, daemon=True).start()


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


async def snapshots():
    return await asyncio.wait_for(asyncio.gather(*(
        D.get.aio(F.name + ":curated-multi-s" + str(seed), None) for seed in (42, 43)
    )), timeout=30)


prior = []
for seed in (42, 43):
    report = read(P / "results" / ("wide-multi-s" + str(seed)) / "report.json")
    prior.append(dict(condition="parent-s" + str(seed), status=dict(status="selected checkpoint"),
        progress=dict(steps=report["selected_steps"], max_steps=report["summary"]["steps"]),
        metrics=dict(validation=report["summary"]["metrics"]["validation"]),
        ligand_validation=report["ligands"]["validation"]))
last = [None, None]
runner = asyncio.Runner()
while True:
    error = None
    try:
        last = runner.run(snapshots())
    except Exception as exc:
        error = repr(exc)
    prepared = read(F / "results/prepared.json", {})
    state = dict(family=F.name, refreshed_unix=time.time(), poll_error=error,
        runs=prior + [r for r in last if r],
        preparation=read(F / "results/preparation-status.json", {"stage": "CPU source/partition verification"}),
        train_rows=sum(prepared.get("train_strata", {}).values()),
        planned_steps=prepared.get("steps"), source_review=prepared.get("source_review", {}))
    if prepared.get("status") == "prepared":
        state["preparation"] = dict(stage="Cohort verified; worker status below is authoritative")
    target = F / "results/live.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, allow_nan=False) + "\n")
    temporary.replace(target)
    time.sleep(20)
