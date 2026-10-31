"""Bounded read-only recovery probe for one already-finished artifact."""

from pathlib import Path
import hashlib
import json
import signal
import time

import modal

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
NAME = "wide-six-natural-s42"


def timeout(_signal, _frame):
    raise TimeoutError("120-second transfer diagnostic budget")


def main():
    state = json.loads((F / "results/live-workers.json").read_text())["conditions"][
        NAME
    ]
    assert state["status"]["status"] == "completed"
    expected = state["artifact"]
    destination = (
        ROOT / "artifacts" / F.name / "alternate-downloads" / (NAME + ".tar.partial")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    started = time.monotonic()
    result = dict(
        condition=NAME,
        expected=expected,
        path=str(destination),
        status="started",
        method="modal.Volume.read_file ordered iterator",
        timeout_seconds=120,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    digest = hashlib.sha256()
    size = 0
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        volume = modal.Volume.from_name("presto-checkpoints")
        with destination.open("xb") as stream:
            for chunk in volume.read_file(expected["archive"]):
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        assert size == expected["bytes"]
        assert digest.hexdigest() == expected["sha256"]
        final = destination.with_suffix(".gz")
        destination.replace(final)
        result.update(status="passed", path=str(final))
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
    finally:
        signal.alarm(0)
        result.update(
            seconds=time.monotonic() - started,
            bytes_received=size,
            sha256=digest.hexdigest(),
        )
        (F / "results/stream-archive-probe.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
