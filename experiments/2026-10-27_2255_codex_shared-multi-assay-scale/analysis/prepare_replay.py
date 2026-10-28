"""Export the complete pinned package for verification, without changing a fit."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import tomllib

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
COMMIT = "0c4c984b2628fecf54ef29e756fdfc0a74963dc4"


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def main():
    destination = F / "reproduce/source/serving-runtime/presto"
    assert not destination.exists()
    config = tomllib.loads(git_bytes("pyproject.toml").decode())
    directories = {
        Path(p.removeprefix("presto").lstrip(".").replace(".", "/"))
        for p in config["tool"]["setuptools"]["packages"]
    }
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", COMMIT], cwd=ROOT, text=True
    ).splitlines()
    paths = [p for p in tracked if p.endswith(".py") and Path(p).parent in directories]
    paths.append("data/b2m_sequences.csv")
    hashes = {}
    for relative in sorted(paths):
        data = git_bytes(relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    pins = json.loads((F / "reproduce/source-pins.json").read_text())
    frozen = pins["runtimes"]["shared"]["files"]
    assert all(hashes.get(path) == digest for path, digest in frozen.items())
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from presto.inference.modular import ModularPredictor; "
            "import presto; assert presto.__file__.startswith(sys.argv[1]); "
            "print('Pinned public predictor import passed')",
            str(destination.parent),
        ],
        cwd="/private/tmp",
        check=True,
    )
    receipt = dict(
        source_commit=COMMIT,
        created_unix=time.time(),
        reason="Training-only bundle omitted inference; original replay failed before inference. Export the full package from the identical pinned commit for verification only.",
        files=hashes,
        original_training_files_verified=len(frozen),
        public_predictor_import="passed",
        fitting_or_heldout_inference=False,
        original_failed_log="launch_logs/gpu-training-replay.log",
    )
    (F / "reproduce/serving-runtime.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(f"Exported {len(hashes)} files; all {len(frozen)} frozen files identical")


if __name__ == "__main__":
    main()
