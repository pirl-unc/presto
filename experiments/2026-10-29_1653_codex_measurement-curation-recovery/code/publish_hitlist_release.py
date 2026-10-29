"""Publish only the successful clean-main source-recovery release artifact."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

F = Path(__file__).resolve().parents[1]
ROOT = Path("/Users/iskander/code/presto/.worktrees/hitlist-source-release")
COMMIT = "ff5247f787ce79fbb640a00bcbe702fb7b7910cb"
RUN = "40948383762"
DESTINATION = Path("/Users/iskander/code/presto/artifacts") / F.name / "hitlist-release-2.1.3"


def capture(argv):
    return subprocess.check_output(argv, cwd=ROOT, text=True).strip()


def main():
    assert capture(["git", "rev-parse", "HEAD"]) == COMMIT
    assert capture(["git", "branch", "--show-current"]) == "main"
    assert not capture(["git", "status", "--porcelain", "--untracked-files=no"])
    recipe = dict(
        source_commit=COMMIT,
        workflow_run_id=RUN,
        version="2.1.3",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        release_checkout=str(ROOT),
        artifact_directory=str(DESTINATION),
        started_unix=time.time(),
        requested_gpu=None,
    )
    (F / "reproduce/hitlist-publication.json").write_text(json.dumps(recipe, indent=2) + "\n")
    shutil.copyfile(__file__, F / "reproduce/source/publish_hitlist_release.py")
    subprocess.run(
        [
            "gh",
            "run",
            "watch",
            RUN,
            "--repo",
            "pirl-unc/hitlist",
            "--exit-status",
            "--interval",
            "30",
        ],
        cwd=ROOT,
        check=True,
    )
    run = json.loads(
        capture(
            [
                "gh",
                "api",
                "repos/pirl-unc/hitlist/actions/runs/" + RUN,
            ]
        )
    )
    for key, expected in {
        "head_sha": COMMIT,
        "head_branch": "main",
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
    }.items():
        assert run[key] == expected, key
    DESTINATION.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            "gh",
            "run",
            "download",
            RUN,
            "--repo",
            "pirl-unc/hitlist",
            "--name",
            "hitlist-release-" + COMMIT,
            "--dir",
            str(DESTINATION),
        ],
        cwd=ROOT,
        check=True,
    )
    # deploy.sh verifies the exact dispatch, downloads a second trusted copy,
    # compares both distributions, publishes, then verifies public bytes.
    subprocess.run(
        [
            "./deploy.sh",
            "--artifacts",
            str(DESTINATION),
        ],
        cwd=ROOT,
        check=True,
        env={
            **os.environ,
            "PATH": "/private/tmp/hitlist-curation-release-env/bin:" + os.environ["PATH"],
        },
    )
    receipt = DESTINATION.parent / "pypi-2.1.3-verified.json"
    public = json.loads(receipt.read_text())
    assert public["source_commit"] == COMMIT and public["version"] == "2.1.3"
    shutil.copyfile(receipt, F / "results/hitlist-pypi-2.1.3-verified.json")
    shutil.copyfile(DESTINATION / "release.json", F / "results/hitlist-release-2.1.3.json")
    print("Hitlist 2.1.3 merged, fully tested and both public distributions verified", flush=True)


if __name__ == "__main__":
    main()
