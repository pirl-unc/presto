"""Recover a new immutable source batch and verify the original batch replay."""

from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time

import modal

FID = "2026-10-29_1653_codex_measurement-curation-recovery"
app = modal.App("presto-source-recovery-late-20261029")
volume = modal.Volume.from_name("presto-checkpoints")
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    manifest = json.loads((F / "reproduce/recovery-late-manifest.json").read_text())
    image = image.add_local_file(F / "code/recover_strazar.py", "/bundle/recover_strazar.py")
    image = image.add_local_file(
        F / "reproduce/recovery-late-manifest.json", "/bundle/manifest.json"
    )
    image = image.add_local_file(manifest["accepted_path"], "/bundle/accepted.csv")
    image = image.add_local_file(F / "results/strazar-recovery.json", "/bundle/original.json")
    for source in manifest["sources"]:
        image = image.add_local_file(source["path"], "/inputs/" + source["id"] + ".ssv")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@app.function(
    image=image, cpu=4, memory=16384, timeout=3600, max_containers=1, volumes={"/volume": volume}
)
def run():
    sys.path.insert(0, "/bundle")
    from recover_strazar import recover

    started = time.monotonic()
    manifest, original = read("/bundle/manifest.json"), read("/bundle/original.json")
    assert sha("/bundle/accepted.csv") == original["accepted_ligand_set_sha256"]
    sources = []
    # Keep original filenames for the verified filename-to-restriction mapping.
    Path("/sources").mkdir()
    for source in manifest["sources"]:
        path = Path("/sources") / Path(source["path"]).name
        assert not path.exists()
        path.symlink_to("/inputs/" + source["id"] + ".ssv")
        assert path.stat().st_size == source["bytes"] and sha(path) == source["sha256"]
        sources.append(dict(source, path=str(path), original_local_path=source["path"]))
    original_keys = {(Path(s["path"]).name, s["sha256"]) for s in original["sources"]}
    prior = [s for s in sources if (Path(s["path"]).name, s["sha256"]) in original_keys]
    assert len(prior) == len(original["sources"]) == 11
    replay = recover(Path("/bundle/accepted.csv"), prior, Path("/tmp/replay"))
    assert replay["output"]["sha256"] == original["output"]["sha256"]
    assert replay["counts"] == original["counts"]
    destination = Path("/volume") / FID / "recovery-batches/late"
    assert not destination.exists(), "Never overwrite a source recovery batch"
    result = recover(Path("/bundle/accepted.csv"), sources, Path("/tmp/recovered-late"))

    def identities(path):
        with open(path) as stream:
            return {(r["peptide"], r["mhc_restriction"]) for r in csv.DictReader(stream)}

    before, after = identities(replay["output"]["path"]), identities(result["output"]["path"])
    assert before <= after
    result.update(
        batch="late",
        status="source_verified_not_model_admitted",
        verification=dict(
            status="passed",
            original_csv_byte_replay=True,
            original_counts_equal=True,
            all_previous_positive_identities_retained=True,
        ),
        previous_positive_pmhcs=len(before),
        positive_pmhcs=len(after),
        additional_source_verified_positive_pmhcs=len(after - before),
        available_source_files=20,
        remaining_source_files=112,
        requested_gpu=None,
        cpu=4,
        memory_mib=16384,
        total_seconds=time.monotonic() - started,
    )
    destination.mkdir(parents=True)
    shutil.copytree("/tmp/recovered-late", destination / "recovered")
    result["output"]["path"] = str(destination / "recovered" / Path(result["output"]["path"]).name)
    result["evidence"]["path"] = str(destination / "recovered/source-evidence.jsonl")
    (destination / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archive = destination / "recovered.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(destination / "recovered", arcname="recovered-late")
        tar.add(destination / "summary.json", arcname="summary.json")
    result["archive"] = dict(
        path=str(archive.relative_to("/volume")), sha256=sha(archive), bytes=archive.stat().st_size
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    assert not (F / "reproduce/recovery-late-call.json").exists()
    for name in ("recover_late.py", "recover_strazar.py"):
        shutil.copyfile(F / "code" / name, F / "reproduce/source" / ("late-" + name))
    receipt = dict(
        argv=[sys.executable, "-m", "modal", "run", "--detach", __file__],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=subprocess.check_output(["git", "status", "--short"], text=True),
        source_hashes={
            name: sha(F / "code" / name) for name in ("recover_late.py", "recover_strazar.py")
        },
        manifest_sha256=sha(F / "reproduce/recovery-late-manifest.json"),
        started_unix=time.time(),
        requested_gpu=None,
        cpu=4,
        memory_mib=16384,
    )
    (F / "reproduce/recovery-late-launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    call = run.spawn()
    (F / "reproduce/recovery-late-call.json").write_text(
        json.dumps(dict(call_id=call.object_id)) + "\n"
    )
    result = call.get()
    (F / "results/recovery-late.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    target = Path("/Users/iskander/code/presto/artifacts") / FID / "recovered-late.tar.gz"
    assert not target.exists()
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as stream:
        for block in volume.read_file(result["archive"]["path"]):
            stream.write(block)
    assert temporary.stat().st_size == result["archive"]["bytes"]
    assert sha(temporary) == result["archive"]["sha256"]
    temporary.replace(target)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "positive_pmhcs",
                    "additional_source_verified_positive_pmhcs",
                    "total_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    pass  # Invoke with modal run; local execution never starts remote work implicitly.
