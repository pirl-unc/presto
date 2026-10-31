"""Read-only loopback display. Serves only generated status and the dashboard UI."""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from support import CONDITIONS, FAMILY, RAW, read, atomic


def snapshot():
    rows = []
    for name, spec in CONDITIONS.items():
        directory = FAMILY / "results" / name
        report = read(directory / "report.json", {})
        history = read(RAW / name / "history.json", [])
        metrics = report.get("summary", {}).get("metrics", {})
        current = history[-1].get("validation", {}) if history else {}
        cells = {}
        for kind in ("I", "II"):
            cell = {"scope": "in-class" if kind in spec["classes"] else "zero-shot"}
            for split in ("validation", "test"):
                selected = metrics.get(split, {}).get("ic50", {}).get("by_class", {}).get(kind)
                if selected is None:
                    for transfer in report.get("transfer", {}).values():
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
            cells[kind] = cell
        rows.append(
            dict(
                name=name,
                **spec,
                status=read(FAMILY / "results" / f"{name}-status.json", {"status": "pending"}),
                progress=read(FAMILY / "results" / f"{name}-progress.json", {}),
                history=history,
                cells=cells,
                selected_epoch=report.get("summary", {}).get("selected_epoch"),
            )
        )
    return dict(
        updated_unix=time.time(),
        rows=rows,
        controller=read(FAMILY / "reproduce" / "controller.json", {"status": "preparing"}),
        profiles={
            label: read(FAMILY / "results" / f"profile-{label}.json")
            for label in ("three_aa", "full", "shared")
        },
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            data = (FAMILY / "code" / "dashboard.html").read_bytes()
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


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    atomic(
        FAMILY / "results" / "dashboard.json",
        dict(url=url, started_unix=time.time(), access="loopback only"),
    )
    print(url, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
