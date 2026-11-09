"""Read-only independent checks of the completed workspace restoration."""

import hashlib
import json
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIMARY = Path("/Users/iskander/code/presto")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def digest(path, algorithm="sha256", prefix=b""):
    value = hashlib.new(algorithm)
    value.update(prefix)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def check_asset(path, expected):
    assert path.is_file(), path
    assert path.stat().st_size == expected["size_bytes"], path
    assert digest(path) == expected["sha256"], path


plan = json.loads((HERE / "missing-tracked-files.json").read_text())
entries = plan["restore"] + plan["defer"]
journal = [json.loads(line) for line in (HERE / "restore-journal.jsonl").read_text().splitlines()]
completed = {row["path"]: row for row in journal if "method" in row}
assert len(completed) == len(entries) == 1396
for entry in entries:
    path = ROOT / entry["path"]
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1, path
    assert info.st_size == entry["size_bytes"], path
    assert stat.S_IMODE(info.st_mode) == int(entry["mode"][-3:], 8), path
    prefix = f"blob {info.st_size}\0".encode()
    assert digest(path, "sha1", prefix) == entry["oid"], path
    row = completed[entry["path"]]
    assert row["oid"] == entry["oid"] and row["restored_inode"] == info.st_ino, path
    source = PRIMARY / entry["path"]
    if source.exists():
        assert (info.st_dev, info.st_ino) != (source.stat().st_dev, source.stat().st_ino), path

assert git(ROOT, "rev-parse", "HEAD").decode().strip() == plan["head"]
assert not git(ROOT, "ls-files", "--deleted", "-z")
assert not git(ROOT, "diff", "--cached", "--name-only")
assert Path("/private/tmp/presto-modular-model").is_symlink()
assert Path("/private/tmp/presto-modular-model").resolve() == ROOT
assert Path(git(ROOT, "rev-parse", "--show-toplevel").decode().strip()) == ROOT

before = json.loads((HERE / "primary-before.json").read_text())
commands = {
    "head": ["rev-parse", "HEAD"],
    "status": ["status", "--porcelain", "--untracked-files=no"],
    "working_diff": ["diff", "--binary"],
    "staged_diff": ["diff", "--cached", "--binary"],
}
for name, args in commands.items():
    data = git(PRIMARY, *args)
    assert before[name] == {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}, name

old_status = (HERE / "modular-status.txt").read_text().splitlines()
surviving = {line for line in old_status if not line[:2].strip() == "D"}
assert surviving == set(git(ROOT, *commands["status"]).decode().splitlines())

families = [
    "2026-10-25_0344_codex_proteasome-quantity-source",
    "2026-10-25_0143_codex_intact-temporal-representation",
]
counts = {}
for family in families:
    directory = ROOT / "experiments" / family
    manifest = json.loads((directory / "artifact-manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        check_asset(directory / relative, expected)
    counts[family] = len(manifest["files"])

assets = json.loads(
    (ROOT / "experiments" / families[0] / "results/source-assets.json").read_text()
)["assets"]
for asset in assets:
    check_asset(Path(asset["path"]), asset)

result = {
    "passed": True,
    "verified_utc": datetime.now(timezone.utc).isoformat(),
    "restored_blobs_verified": len(entries),
    "logical_bytes_verified": sum(entry["size_bytes"] for entry in entries),
    "tracked_deletions": 0,
    "primary_head_tracked_status_working_and_staged_diff_unchanged": True,
    "surviving_modular_tracked_status_unchanged": True,
    "source_artifacts_verified": counts,
    "source_assets_verified_at_recorded_absolute_paths": len(assets),
    "durable_root": str(ROOT),
    "compatibility_symlink_verified": True,
    "private_primary_patch_snapshots_published": False,
    "scientific_results_recomputed": False,
}
with (HERE / "verification.json").open("x") as output:
    output.write(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
