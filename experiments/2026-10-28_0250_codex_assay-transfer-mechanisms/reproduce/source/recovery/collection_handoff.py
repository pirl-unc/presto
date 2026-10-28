"""Guarded collector handoff after every frozen fit has finished; never train."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require_extraction_space(archive):
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
    expanded = sum(m.size for m in members if m.isfile())
    # The frozen collector also copies compressed predictions and JSON summaries.
    copies = sum(
        m.size
        for m in members
        if m.isfile() and m.name.count("/") == 1 and ".json" in m.name
    )
    required = expanded + copies + archive.stat().st_size + (2 << 30)
    available = shutil.disk_usage(ROOT).free
    if available < required:
        raise RuntimeError(
            f"Need {required} free bytes before extraction; have {available}"
        )
    return dict(expanded_bytes=expanded, free_before=available, required=required)


def compact_verified_expansions(receipt):
    name = receipt["condition"]
    archive = ROOT / "artifacts" / F.name / "downloads" / f"{name}.tar.gz"
    assert sha(archive) == receipt["sha256"]
    raw = ROOT / "artifacts" / F.name / "runs"
    allow = [
        f"{name}/raw/{suffix}"
        for split in ("train", "validation", "test")
        for suffix in (f"evaluations/{split}.json", f"{split}-predictions.jsonl")
    ]
    entries = []
    record = F / "results" / f"{name}-storage-compaction.json"
    assert not record.exists()
    with tarfile.open(archive) as stream:
        for member_name in allow:
            member = stream.getmember(member_name)
            path = raw / member_name
            assert member.isfile() and path.is_file()
            assert path.stat().st_size == member.size
            h = hashlib.sha256()
            with stream.extractfile(member) as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    h.update(chunk)
            digest = sha(path)
            assert digest == h.hexdigest()
            entries.append(dict(member=member_name, bytes=member.size, sha256=digest))
    # Preserve verified recovery coordinates before deleting any redundant bytes.
    result = dict(
        status="verified_before_compaction",
        archive=str(archive),
        archive_sha256=receipt["sha256"],
        entries=entries,
        removed_bytes=0,
        policy="Remove only byte-identical expanded exports; retain original archive, all checkpoints and gzip predictions. No metric parsing.",
    )
    atomic(record, result)
    for entry in entries:
        (raw / entry["member"]).unlink()
        result["removed_bytes"] += entry["bytes"]
        atomic(record, result)
    result.update(status="passed", free_after=shutil.disk_usage(ROOT).free)
    atomic(record, result)
    return result


def prepare():
    import modal

    names = tuple(read(F / "reproduce/source-pins.json")["conditions"])
    workers = read(F / "results/live-workers.json")
    cache = read(F / "results/artifact-readback-state.json")
    assert workers["all_terminal"] and cache["status"] == "completed"
    assert set(cache["ready"]) == set(names) and not cache["pending"]
    calls = read(F / "reproduce/call-ids.json")
    assert set(calls) == set(names)
    receipts = {}
    paths = {}
    for name in names:
        assert workers["conditions"][name]["status"]["status"] == "completed"
        receipt = modal.FunctionCall.from_id(calls[name]).get(timeout=10)
        assert (
            receipt
            == workers["conditions"][name]["artifact"]
            == cache["ready"][name]["expected"]
        )
        path = Path(cache["ready"][name]["path"])
        assert (
            path.stat().st_size == receipt["bytes"] and sha(path) == receipt["sha256"]
        )
        receipts[name] = receipt
        paths[receipt["archive"]] = str(path)
    destination = F / "reproduce/collection-handoff-ready.json"
    assert not destination.exists()
    atomic(
        destination,
        dict(
            status="ready",
            all_original_calls_terminal=True,
            no_fit_restarts=True,
            prepared_unix=time.time(),
            call_ids=calls,
            receipts=receipts,
            verified_local_archives=paths,
            original_controller=read(F / "reproduce/controller.json"),
        ),
    )
    print(
        "All twelve original calls returned and all original archive hashes pass. Stop only the verified original collector, then run --finish with its former PID.",
        flush=True,
    )


def finish(former_controller_pid):
    assert former_controller_pid is not None
    # Never stop a process here. The operator must first verify/stop the owned
    # original collector after prepare proves no fit calls remain pending.
    probe = subprocess.run(
        ["ps", "-p", str(former_controller_pid), "-o", "pid="],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0 or probe.stdout.strip():
        raise ValueError("former collector PID still exists; do not race two writers")
    assert probe.returncode == 1 and not probe.stderr.strip(), probe.stderr
    ready = read(F / "reproduce/collection-handoff-ready.json")
    assert ready["status"] == "ready" and ready["all_original_calls_terminal"]
    assert len(ready["receipts"]) == 12
    source = F / "code/launch.py"
    assert sha(source) == read(F / "reproduce/launch.json")["source_pins"]["launch.py"]
    sys.path.insert(0, str(F / "code"))
    import launch

    class VerifiedLocalTransport:
        def read_file_into_fileobj(self, path, output):
            source = Path(ready["verified_local_archives"][path])
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=1 << 20)

    launch.volume = VerifiedLocalTransport()
    original = F / "reproduce/controller-original-before-handoff.json"
    assert not original.exists()
    atomic(original, read(F / "reproduce/controller.json"))
    downloads = ROOT / "artifacts" / F.name / "downloads"
    failed = downloads / "failed-volume-api"
    failed.mkdir(exist_ok=True)
    preserved = []
    for path in downloads.glob("*.partial"):
        destination = failed / path.name
        assert not destination.exists()
        digest = sha(path)
        path.replace(destination)
        preserved.append(
            dict(path=str(destination), sha256=digest, bytes=destination.stat().st_size)
        )
    completed = []
    storage = {}
    for name, receipt in ready["receipts"].items():
        # Original frozen collector performs extraction, all copies, checkpoint
        # checksum validation and local report rewriting. Only transport differs.
        archive = Path(ready["verified_local_archives"][receipt["archive"]])
        storage[name] = require_extraction_space(archive)
        launch.collect(receipt)
        storage[name]["compaction"] = compact_verified_expansions(receipt)
        completed.append(name)
        atomic(
            F / "reproduce/controller.json",
            dict(
                status="completed" if len(completed) == 12 else "collecting_recovered",
                completed=completed,
                pending=[n for n in ready["receipts"] if n not in completed],
                failed=[],
                collection_transport="verified CPU readback through original frozen collector",
            ),
        )
        print(name, "collected", flush=True)
    atomic(
        F / "results/collection-handoff.json",
        dict(
            status="passed",
            completed=completed,
            former_controller_pid=former_controller_pid,
            preserved_failed_downloads=preserved,
            original_collector_sha256=sha(source),
            no_fit_restarts=True,
            original_archive_hashes_unchanged=True,
            storage=storage,
            finished_unix=time.time(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--finish", action="store_true")
    parser.add_argument("--former-controller-pid", type=int)
    args = parser.parse_args()
    prepare() if args.prepare else finish(args.former_controller_pid)
