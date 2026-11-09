"""Freeze two source runtimes and previously authorized benchmark inputs."""

from pathlib import Path
import hashlib
import json
import shutil
import subprocess

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
PRIOR = ROOT / "experiments/2026-10-27_1930_codex_modal-cross-class"
UPLOAD = ROOT / "artifacts" / FAMILY.name / "upload"
INPUTS = Path(
    "/Users/iskander/code/presto/artifacts/2026-10-27_1930_codex_modal-cross-class/upload/inputs"
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pins = json.loads((PRIOR / "reproduce/source-pins.json").read_text())
    base_files = list(pins["runtimes"]["shared"]["files"])
    shared_new = ["training/modular_sampling.py"]
    # Pooled architecture comes from immutable git parent, with common sampler added.
    base = "4615573598f24dc1e1f05ec1f8b130d0a90c64a9"
    for runtime in ("pooled", "interaction"):
        files = base_files + shared_new
        if runtime == "interaction":
            files += ["legacy/shared_v4.py", "legacy/shared_v4_support.py"]
        destination = UPLOAD / "runtimes" / runtime / "presto"
        for relative in files:
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if runtime == "pooled" and relative not in (
                "training/modular_sampling.py",
                "training/modular_run.py",
            ):
                path.write_bytes(
                    subprocess.check_output(["git", "show", f"{base}:{relative}"], cwd=ROOT)
                )
            else:
                shutil.copy2(ROOT / relative, path)
        pins.setdefault("new_runtimes", {})[runtime] = dict(
            local_runtime=str(destination),
            files={rel: sha(destination / rel) for rel in files},
            base_commit=base,
            common_trainer_sha256=sha(destination / "training/modular_run.py"),
            architecture="pooled shared-v4"
            if runtime == "pooled"
            else "shared-v5 residue cross-attention",
        )
    pins["runtimes"] = pins.pop("new_runtimes")
    pins["conditions"] = {}
    for seed in (42, 43):
        for variant in (
            "pooled-natural-joint",
            "pooled-balanced-joint",
            "interaction-natural-joint",
            "interaction-balanced-joint",
            "interaction-I",
            "interaction-II",
        ):
            classes = ["I", "II"] if variant.endswith("joint") else [variant.rsplit("-", 1)[1]]
            pins["conditions"][f"{variant}-s{seed}"] = dict(
                runtime=variant.split("-")[0],
                classes=classes,
                seed=seed,
                max_steps=2000 if len(classes) == 2 else 1000,
                balance_by=["mhc_class"] if "balanced" in variant else [],
            )
    (UPLOAD / "inputs").mkdir(parents=True, exist_ok=True)
    for split, expected in pins["joint_input_hashes"].items():
        source = INPUTS / f"joint-{split}.jsonl"
        assert sha(source) == expected
        shutil.copy2(source, UPLOAD / "inputs" / source.name)
    pins["plan"] = "experiments/agents/codex/plans/2026-10-27_transfer-improvements.md"
    pins["source_git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    pins["source_git_dirty"] = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=ROOT, text=True
        ).strip()
    )
    for path in (UPLOAD / "source-pins.json", FAMILY / "reproduce/source-pins.json"):
        path.write_text(json.dumps(pins, indent=2) + "\n")
    for runtime in pins["runtimes"]:
        # Small code snapshots are part of the canonical reproducibility bundle.
        shutil.copytree(
            UPLOAD / "runtimes" / runtime,
            FAMILY / "reproduce/source/runtimes" / runtime,
            dirs_exist_ok=True,
        )
    print(
        json.dumps(
            {
                "conditions": list(pins["conditions"]),
                "upload_bytes": sum(p.stat().st_size for p in UPLOAD.rglob("*") if p.is_file()),
            }
        )
    )


if __name__ == "__main__":
    main()
