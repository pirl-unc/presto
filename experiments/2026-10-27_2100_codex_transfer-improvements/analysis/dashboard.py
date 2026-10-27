"""Local live view of Modal receipts, with the CPU reference kept separate."""

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
CPU = ROOT / "experiments/2026-10-27_1835_codex_cross-class-flank-benchmark"
CPU_RAW = ROOT / "artifacts" / CPU.name
ORDER = [
    f"{v}-s{s}"
    for s in (42, 43)
    for v in (
        "pooled-natural-joint",
        "pooled-balanced-joint",
        "interaction-natural-joint",
        "interaction-balanced-joint",
        "interaction-I",
        "interaction-II",
    )
]


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def cells(spec, state):
    result = {}
    history = state.get("history", [])
    current = history[-1].get("validation", {}) if history else {}
    for kind in ("I", "II"):
        cell = {"scope": "in-class" if kind in spec["classes"] else "zero-shot"}
        for split in ("validation", "test"):
            selected = (
                state.get("metrics", {})
                .get(split, {})
                .get("ic50", {})
                .get("by_class", {})
                .get(kind)
            )
            if selected is None:
                for transfer in state.get("transfer", {}).values():
                    if transfer["target_class"] == kind:
                        selected = (
                            transfer["metrics"]
                            .get(split, {})
                            .get("ic50", {})
                            .get("by_class", {})
                            .get(kind)
                        )
            if selected is None and split == "validation":
                selected = current.get("ic50", {}).get("by_class", {}).get(kind)
            cell[split] = selected
        result[kind] = cell
    return result


def snapshot():
    meta = read(FAMILY / "reproduce/source-pins.json")
    live = read(FAMILY / "results/live-status.json", {})
    rows = []
    for name in ORDER:
        spec = meta["conditions"][name]
        state = live.get("conditions", {}).get(name, {})
        report = read(FAMILY / "results" / name / "report.json", {})
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
                selected_epoch=state.get("selected_epoch"),
                hardware=state.get("hardware", {}),
                resources=report.get("resources", {}),
            )
        )
    name = "shared-I-s42"
    report = read(CPU / "results" / name / "report.json", {})
    state = dict(
        history=read(CPU_RAW / name / "history.json", []),
        metrics=report.get("summary", {}).get("metrics", {}),
        transfer=report.get("transfer", {}),
    )
    cpu = dict(
        progress=read(CPU / "results" / f"{name}-progress.json", {}),
        status=read(CPU / "results" / f"{name}-status.json", {"status": "unknown"}),
        cells=cells({"classes": ["I"]}, state),
    )
    return dict(
        updated_unix=time.time(),
        rows=rows,
        cpu_reference=cpu,
        controller=read(
            FAMILY / "reproduce/controller.json", {"status": "starting", "completed": []}
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
    server = ThreadingHTTPServer(("127.0.0.1", 54496), Handler)
    print("http://127.0.0.1:54496/", flush=True)
    server.serve_forever()
