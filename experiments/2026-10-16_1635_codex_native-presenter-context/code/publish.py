"""Stage immutable audit assets with datacache, then publish and verify."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from datacache import materialize
from verify import verify

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
TAG = "native-presenter-context-audit-v1"
RELEASE = "https://github.com/pirl-unc/presto/releases/tag/" + TAG
BASE = "https://github.com/pirl-unc/presto/releases/download/" + TAG + "/"
CACHE = Path.home() / ".hitlist/presto-presenter-context-audit/publication"


def pin(path):
    path = Path(path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def prepare():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    receipt = json.loads((FAMILY / "results/audit-receipt.json").read_text())
    files = {name: Path(record["path"]) for name, record in receipt["files"].items()}
    for name, path in files.items():
        assert pin(path) == {
            key: receipt["files"][name][key] for key in ("sha256", "size")
        }
    files.update(
        {
            "README.md": FAMILY / "README.md",
            "launch.json": FAMILY / "reproduce/launch.json",
            "launch.sh": FAMILY / "reproduce/launch.sh",
            "verify.py": FAMILY / "code/verify.py",
            "publish.py": Path(__file__),
            "plan.md": ROOT / "tasks/native-presenter-conditioning-spec.md",
        }
    )
    names = [*files, "publication.json", "SHA256SUMS"]

    def build(inputs, outputs):
        for name in files:
            shutil.copyfile(inputs[name], outputs[name])
        write(
            outputs["publication.json"],
            {
                "schema_version": 1,
                "release": RELEASE,
                "executed_recipe_commit": receipt["commit"],
                "publication_commit": commit,
                "original_run_preserved": True,
                "files": {
                    name: {"url": BASE + name, **pin(outputs[name])} for name in files
                },
            },
        )
        Path(outputs["SHA256SUMS"]).write_text(
            "".join(
                pin(outputs[name])["sha256"] + "  " + name + "\n"
                for name in names
                if name != "SHA256SUMS"
            )
        )

    paths = materialize(
        CACHE / commit,
        {name: {"path": str(path), **pin(path)} for name, path in files.items()},
        transform={
            "version": "native-presenter-context-publication-v1",
            "options": {"commit": commit},
        },
        outputs={name: {} for name in names},
        builder=build,
    )
    result = {
        "release": RELEASE,
        "commit": commit,
        "files": {
            name: {"path": str(path), "url": BASE + name, **pin(path)}
            for name, path in paths.items()
        },
    }
    write(FAMILY / "results/publication.json", result)
    print(json.dumps({"prepared": len(paths), "commit": commit}))


def publish():
    receipt = json.loads((FAMILY / "results/publication.json").read_text())
    files = receipt["files"]
    for record in files.values():
        assert pin(record["path"]) == {key: record[key] for key in ("sha256", "size")}
    gh(
        "release",
        "create",
        TAG,
        "--repo",
        "pirl-unc/presto",
        "--target",
        receipt["commit"],
        "--draft",
        "--title",
        "Native presenter processing context audit",
        "--notes-file",
        str(FAMILY / "README.md"),
        *[record["path"] for record in files.values()],
    )
    releases = json.loads(gh("api", "repos/pirl-unc/presto/releases?per_page=100"))
    release = next(item for item in releases if item["tag_name"] == TAG)
    assert {asset["name"] for asset in release["assets"]} == files.keys()
    for asset in release["assets"]:
        record = files[asset["name"]]
        assert (
            asset["size"] == record["size"]
            and asset["digest"] == "sha256:" + record["sha256"]
        )
    gh(
        "api",
        "--method",
        "PATCH",
        "repos/pirl-unc/presto/releases/" + str(release["id"]),
        "-F",
        "draft=false",
        "-f",
        "make_latest=false",
    )
    # This outer receipt pins all assets, including publication.json and SHA256SUMS.
    result = verify(FAMILY / "results/publication.json")
    write(FAMILY / "results/public-verification.json", result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "public_downloads"},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "publish"])
    args = parser.parse_args()
    (prepare if args.action == "prepare" else publish)()
