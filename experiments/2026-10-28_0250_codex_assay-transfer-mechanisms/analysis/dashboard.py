"""Local live view of Modal receipts, with the CPU reference kept separate."""

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
ORDER = [
    f"{v}-s{s}"
    for s in (42, 43)
    for v in (
        "small-ic50",
        "wide-ic50",
        "wide-six-natural",
        "wide-six-class-balanced",
        "wide-six-primary-matched",
        "wide-six-primary-shuffled",
    )
]
ENDPOINTS = ("ic50", "kd", "ec50", "kd_via_ic50", "kd_via_ec50", "half_life")


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def cells(spec, state, endpoint="ic50"):
    result = {}
    history = state.get("history", [])
    current = history[-1].get("validation", {}) if history else {}
    for kind in ("I", "II"):
        cell = {"scope": "in-class" if kind in spec["classes"] else "zero-shot"}
        for split in ("validation", "test"):
            selected = (
                state.get("metrics", {})
                .get(split, {})
                .get(endpoint, {})
                .get("by_class", {})
                .get(kind)
            )
            if selected is None:
                for transfer in state.get("transfer", {}).values():
                    if transfer["target_class"] == kind:
                        selected = (
                            transfer["metrics"]
                            .get(split, {})
                            .get(endpoint, {})
                            .get("by_class", {})
                            .get(kind)
                        )
            if selected is None and split == "validation":
                selected = current.get(endpoint, {}).get("by_class", {}).get(kind)
            cell[split] = selected
        result[kind] = cell
    return result


def snapshot():
    meta = read(FAMILY / "reproduce/source-pins.json")
    live = read(FAMILY / "results/live-status.json", {})
    workers = read(FAMILY / "results/live-workers.json", {})
    states = {**live.get("conditions", {}), **workers.get("conditions", {})}
    rows = []
    decision = read(FAMILY / "results/validation-decision.json", {})
    verified = 0
    for name in ORDER:
        spec = meta["conditions"][name]
        state = states.get(name, {})
        report = read(FAMILY / "results" / name / "report.json", {})
        verification = read(
            FAMILY / "results" / f"{name}-independent-verification.json", {}
        )
        if (
            decision.get("test_used_for_decision") is False
            and verification.get("status") == "passed"
        ):
            state = dict(state)
            state["metrics"] = report["summary"]["metrics"]
            state["transfer"] = report.get("transfer", {})
            state["selected_epoch"] = report["summary"]["selected_epoch"]
            verified += 1
        rows.append(
            dict(
                name=name,
                seed=spec["seed"],
                max_steps=spec["max_steps"],
                classes=spec["classes"],
                runtime=spec["runtime"],
                status=state.get("status", {"status": "queued"}),
                progress=state.get("progress", {}),
                history=state.get("history", []),
                cells=cells(spec, state),
                endpoint_cells={e: cells(spec, state, e) for e in ENDPOINTS},
                selected_epoch=state.get("selected_epoch"),
                hardware=state.get("hardware", {}),
                resources=report.get("resources", {}),
            )
        )
    return dict(
        updated_unix=time.time(),
        worker_polled_unix=workers.get("updated_unix", live.get("updated_unix")),
        worker_poll_errors=workers.get("errors", {}),
        remotely_finished=sum(
            s.get("status", {}).get("status") == "completed" for s in states.values()
        ),
        verified=verified,
        preferred_variant=decision.get("preferred_variant"),
        rows=rows,
        controller=read(
            FAMILY / "reproduce/controller.json",
            {
                "status": "preparing: source and sampler checks; no training started",
                "completed": [],
            },
        ),
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            data = (FAMILY / "analysis/dashboard.html").read_bytes()
            content = "text/html; charset=utf-8"
        elif self.path == "/state.json":
            data = json.dumps(snapshot(), allow_nan=False).encode()
            content = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 54501), Handler)
    print("http://127.0.0.1:54501/", flush=True)
    server.serve_forever()
