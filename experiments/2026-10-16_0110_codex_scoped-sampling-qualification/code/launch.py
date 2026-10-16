"""Launch the registered CPU-only sampling and recover verified artifacts."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("presto-scoped-sampling-qualification")
volume = modal.Volume.from_name("presto-data", create_if_missing=False)
if modal.is_local():
    FAMILY = Path(__file__).resolve().parents[1]
    REPOSITORY = FAMILY.parents[1]
    WHEEL = Path(os.environ["PRESTO_QUALIFICATION_WHEEL"]).resolve(strict=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .add_local_file(
            str(FAMILY / "reproduce/requirements.txt"),
            "/opt/qualification/requirements.txt",
            copy=True,
        )
        .run_commands(
            "python -m pip install torch==2.14.1+cpu "
            "--extra-index-url https://download.pytorch.org/whl/cpu "
            "-r /opt/qualification/requirements.txt "
            "--report /opt/qualification/consumer-install.json"
        )
        .add_local_file(str(WHEEL), "/opt/qualification/" + WHEEL.name, copy=True)
        .run_commands(
            "python -m pip install --no-deps /opt/qualification/" + WHEEL.name,
            "python -m pip check",
        )
        .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        .add_local_file(str(FAMILY / "code/fit.py"), "/opt/qualification/fit.py")
        .add_local_file(str(FAMILY / "code/baselines.py"), "/opt/qualification/baselines.py")
        .add_local_file(str(FAMILY / "reproduce/launch.json"), "/opt/qualification/launch.json")
        .add_local_file(
            str(FAMILY / "reproduce/supported_outputs.json"),
            "/opt/qualification/supported_outputs.json",
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
    memory=(32768, 32768),
    ephemeral_disk=524288,
    timeout=3600,
    max_containers=1,
    retries=0,
)
def execute(metadata):
    import sys

    root = Path("/opt/qualification")
    (root / "invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        subprocess.run([sys.executable, str(root / "fit.py")], check=True)
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
        "recipe_sha256": digest(FAMILY / "code/fit.py"),
        "launcher_sha256": digest(__file__),
        "configuration_sha256": digest(FAMILY / "reproduce/launch.json"),
        "supported_outputs_sha256": digest(FAMILY / "reproduce/supported_outputs.json"),
        "baselines_sha256": digest(FAMILY / "code/baselines.py"),
        "requirements_sha256": digest(FAMILY / "reproduce/requirements.txt"),
        "cpu_limit": 8,
        "memory_limit_mib": 32768,
        "gpu": "none",
        "timeout_seconds": 3600,
    }
    receipt = execute.remote(metadata)
    with tempfile.TemporaryDirectory(prefix="scoped-sampling-download-") as tmp:
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
            Path.home() / ".hitlist/presto-scoped-sampling-qualification/sampling",
            sources,
            transform={"version": "remote-sampling-recovery-v1", "options": metadata},
            outputs={name: {} for name in sources},
            builder=builder,
        )
    receipt["local_paths"] = {name: str(path) for name, path in paths.items()}
    (FAMILY / "results/sampling.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if receipt["summary"]["status"] != "success":
        raise RuntimeError(
            "Sampling comparison failed; original logs and outputs have been preserved"
        )


if __name__ == "__main__":
    raise SystemExit("Use modal run with the registered reproduce/launch.sh")
