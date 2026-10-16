"""Launch the registered CPU-only preflight and recover verified artifacts."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("presto-human-corpus-preflight")
volume = modal.Volume.from_name("presto-data", create_if_missing=False)
if modal.is_local():
    FAMILY = Path(__file__).resolve().parents[1]
    REPOSITORY = FAMILY.parents[1]
    WHEEL = Path(os.environ["PRESTO_QUALIFICATION_WHEEL"]).resolve(strict=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install_from_requirements(
            str(FAMILY / "reproduce/hitlist-requirements.txt"), extra_options="--require-hashes"
        )
        .add_local_file(
            str(FAMILY / "reproduce/consumer-constraints.txt"),
            "/opt/qualification/constraints.txt",
            copy=True,
        )
        .run_commands(
            "python -m pip install torch==2.14.1+cpu mhcseqs==2.6.13 "
            "--extra-index-url https://download.pytorch.org/whl/cpu "
            "--constraint /opt/qualification/constraints.txt "
            "--report /opt/qualification/consumer-install.json"
        )
        .add_local_file(str(WHEEL), "/opt/qualification/" + WHEEL.name, copy=True)
        .run_commands(
            "python -m pip install --no-deps /opt/qualification/" + WHEEL.name,
            "python -m pip check",
        )
        .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        .add_local_file(str(FAMILY / "code/preflight.py"), "/opt/qualification/preflight.py")
        .add_local_file(str(FAMILY / "reproduce/launch.json"), "/opt/qualification/launch.json")
        .add_local_file(
            str(FAMILY / "reproduce/mhc-sources.json"), "/opt/qualification/mhc-sources.json"
        )
    )
else:
    image = modal.Image.debian_slim(python_version="3.12")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(8, 8),
    memory=(65536, 65536),
    ephemeral_disk=524288,
    timeout=3000,
    max_containers=1,
    retries=0,
)
def execute(metadata):
    import sys

    root = Path("/opt/qualification")
    (root / "invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        subprocess.run([sys.executable, str(root / "preflight.py")], check=True)
        return json.loads((root / "receipt.json").read_text())
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    from datacache import materialize

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=REPOSITORY, text=True)
    )
    if dirty:
        raise RuntimeError("Commit the exact experiment recipe before launching")
    metadata = {
        "commit": commit,
        "dirty": dirty,
        "wheel": {"name": WHEEL.name, "sha256": digest(WHEEL)},
        "recipe_sha256": digest(FAMILY / "code/preflight.py"),
        "launcher_sha256": digest(__file__),
        "configuration_sha256": digest(FAMILY / "reproduce/launch.json"),
        "source_pins_sha256": digest(FAMILY / "reproduce/mhc-sources.json"),
        "requirements_sha256": digest(FAMILY / "reproduce/hitlist-requirements.txt"),
        "constraints_sha256": digest(FAMILY / "reproduce/consumer-constraints.txt"),
        "cpu_limit": 8,
        "memory_limit_mib": 65536,
        "gpu": "none",
        "timeout_seconds": 3000,
    }
    receipt = execute.remote(metadata)
    with tempfile.TemporaryDirectory(prefix="human-preflight-download-") as tmp:
        sources = {}
        for name, pin in receipt["files"].items():
            path = Path(tmp) / name
            with path.open("wb") as stream:
                for chunk in volume.read_file(pin["path"].removeprefix("/data/")):
                    stream.write(chunk)
            if path.stat().st_size != pin["size_bytes"] or digest(path) != pin["sha256"]:
                raise ValueError("Remote result checksum mismatch: " + name)
            sources[name] = {"path": str(path), "sha256": pin["sha256"]}

        def builder(inputs, outputs):
            for name, source in inputs.items():
                shutil.copyfile(source, outputs[name])

        paths = materialize(
            Path.home() / ".hitlist/presto-human-qualification/preflight",
            sources,
            transform={"version": "remote-preflight-recovery-v1", "options": metadata},
            outputs={name: {} for name in sources},
            builder=builder,
        )
    receipt["local_paths"] = {name: str(path) for name, path in paths.items()}
    (FAMILY / "results/preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if receipt["summary"]["status"] != "success":
        raise RuntimeError("Preflight failed; original logs and outputs have been preserved")


if __name__ == "__main__":
    raise SystemExit("Use modal run with the registered reproduce/launch.sh")
