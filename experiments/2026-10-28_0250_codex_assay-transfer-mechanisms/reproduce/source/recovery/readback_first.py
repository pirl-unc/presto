"""CPU-only transport of existing immutable archives; no model/data evaluation."""

from pathlib import Path
import hashlib
import json
import time

import modal

FAMILY_ID = "2026-10-28_0250_codex_assay-transfer-mechanisms"
VARIANTS = (
    "small-ic50",
    "wide-ic50",
    "wide-six-natural",
    "wide-six-class-balanced",
    "wide-six-primary-matched",
    "wide-six-primary-shuffled",
)
NAMES = tuple(f"{v}-s{s}" for s in (42, 43) for v in VARIANTS)
app = modal.App("presto-assay-transfer-artifact-readback")
volume = modal.Volume.from_name("presto-checkpoints")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=1,
    memory=1024,
    timeout=900,
    max_containers=1,
    volumes={"/runs": volume},
)
def stream_archive(condition: str, expected_sha: str, expected_bytes: int):
    if condition not in NAMES:
        raise ValueError("condition outside the frozen family")
    path = Path("/runs") / FAMILY_ID / f"{condition}.tar.gz"
    assert path.stat().st_size == expected_bytes
    assert sha(path) == expected_sha
    offset = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(256 << 10), b""):
            yield dict(offset=offset, data=chunk)
            offset += len(chunk)
    assert offset == expected_bytes


@app.local_entrypoint()
def main(condition: str = "wide-six-natural-s42"):
    family = Path(__file__).resolve().parents[1]
    root = family.parents[1]
    state = json.loads((family / "results/live-workers.json").read_text())[
        "conditions"
    ][condition]
    assert state["status"]["status"] == "completed"
    expected = state["artifact"]
    destination = (
        root
        / "artifacts"
        / family.name
        / "readback-downloads"
        / f"{condition}.tar.partial"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    started = time.monotonic()
    result = dict(
        condition=condition,
        expected=expected,
        method="CPU-only mounted filesystem and 256KiB function-result stream",
        requested_gpu=None,
        cpu=1,
        memory_mib=1024,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status="started",
    )
    size = 0
    try:
        with destination.open("xb") as output:
            for item in stream_archive.remote_gen(
                condition, expected["sha256"], expected["bytes"]
            ):
                assert item["offset"] == size
                output.write(item["data"])
                size += len(item["data"])
        assert size == expected["bytes"] and sha(destination) == expected["sha256"]
        final = destination.with_suffix(".gz")
        destination.replace(final)
        result.update(status="passed", path=str(final))
    except Exception as exc:
        result.update(status="failed", error=repr(exc), path=str(destination))
        raise
    finally:
        result.update(bytes_received=size, seconds=time.monotonic() - started)
        (family / "results" / f"{condition}-readback.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)
