"""Seal inference code and archived checkpoints before outer outcome access."""

import datetime
import os
import platform
import shutil
import subprocess
import sys

from core import ROOT, read, sha, write

REPO = ROOT.parents[1]
recipe = REPO / "data/hap1_response_recipes/2026-10-24_0730_v1"
baseline = REPO / "experiments/2026-10-24_0730_codex_hap1-response-baselines"
old = REPO / "experiments/2026-10-24_0810_codex_erap1-trimming"
new = REPO / "experiments/2026-10-24_1007_codex_erap1-length-response"
reproduce = ROOT / "reproduce"
reproduce.mkdir(exist_ok=False)
(reproduce / "source").mkdir()
(ROOT / "launch_logs").mkdir()
(ROOT / "results").mkdir()
pins = {}
assert (
    sha(recipe / "manifest.json")
    == "a1eec392c06485966e131663f85e7599dcd9e5ff621841eb0d44eeaafe779309"
)
for relative, record in read(recipe / "manifest.json")["files"].items():
    path = recipe / relative
    assert sha(path) == record["sha256"]
    pins[str(path)] = record["sha256"]
pins[str(recipe / "manifest.json")] = sha(recipe / "manifest.json")
for directory in [baseline, old, new]:
    closure = directory / "results/closure.json"
    pins[str(closure)] = sha(closure)
    for relative, expected in read(closure)["artifact_sha256"].items():
        if relative.startswith("results/"):
            assert sha(directory / relative) == expected, relative
            if (
                relative.endswith("fit.json")
                and "/full/" in relative
                or relative.startswith("results/independent-")
                or relative in ["results/fits/checkpoints.json", "results/fits/cohort.parquet"]
                or relative.endswith("-train.parquet")
                or relative.endswith("/train.parquet")
            ):
                pins[str(directory / relative)] = expected
checkpoints = {
    "direct": {
        name: value["full"]
        for name, value in read(baseline / "results/fits/checkpoints.json").items()
    },
    "physical": {},
}
for prefix, directory, arms in [
    ("0810", old, ["uniform_exact", "residue_exact", "augmented_exact", "augmented_sampled"]),
    ("1007", new, ["length_exact", "residue_length_exact", "augmented_exact", "augmented_sampled"]),
]:
    for arm in arms:
        checkpoints["physical"][prefix + "_" + arm] = {
            "directory": str(directory),
            "checkpoint": read(directory / "results/fits" / arm / "full/fit.json"),
        }
write(reproduce / "checkpoints.json", checkpoints)
plan = REPO / "experiments/agents/codex/plans/hap1-frozen-outer-evaluation.md"
shutil.copyfile(plan, reproduce / "plan.md")
for path in sorted((ROOT / "code").glob("*.py")):
    shutil.copyfile(path, reproduce / "source" / path.name)
pins[str(REPO / "models/n_terminal_trimming.py")] = sha(REPO / "models/n_terminal_trimming.py")
frozen = {str(p.relative_to(ROOT)): sha(p) for p in reproduce.rglob("*") if p.is_file()}
frozen.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "code").glob("*.py")})
config = {
    "registered_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "agent": "Codex / GPT-6",
    "python": sys.executable,
    "platform": platform.platform(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], text=True
    ),
    "recipe": str(recipe),
    "baseline": str(baseline),
    "input_sha256": pins,
    "frozen_sha256": frozen,
    "outer_outcomes_accessed": False,
    "environment": {
        k: os.environ.get(k)
        for k in ["PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
    },
    "primary_model": "sequence_0.1",
    "primary_comparator": "calibrated_precursor",
    "primary_physical": "1007_augmented_exact",
    "caps_seconds": [900, 1800, 900],
    "peak_gib_cap": 4,
    "training": "none; archived full-training checkpoints only",
    "assay": "positive log10 KO/WT MHC-I elution response; complete precursor support",
    "commands": [
        f"{sys.executable} {ROOT}/code/launch.py --phase {p}" for p in ["prepare", "evaluate"]
    ]
    + [f"{sys.executable} {ROOT}/code/verify.py"],
}
write(reproduce / "launch.json", config)
write(ROOT / "manifest.json", config)
lines = ["#!/bin/sh", "set -eu", f"cd '{REPO}'"]
lines += [f"export {k}='{v}'" for k, v in config["environment"].items() if v is not None]
lines += config["commands"]
(reproduce / "launch.sh").write_text("\n".join(lines) + "\n")
print("Frozen source and checkpoints without reading outer outcome values.", flush=True)
