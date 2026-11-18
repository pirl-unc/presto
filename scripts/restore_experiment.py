"""Restore a checksummed historical experiment without overwriting changed files."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def restore(receipt_path, destination):
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "presto.history-archive.v1":
        raise ValueError("unsupported history archive")
    archive = receipt_path.parent / receipt["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt["archive_sha256"]:
        raise ValueError("history archive hash mismatch")
    root = Path(destination).resolve()
    prefix = PurePosixPath(receipt["prefix"])
    expected = {row["path"]: row for row in receipt["files"]}
    plan = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            # git archive includes ancestor directory entries such as experiments/.
            if member.isdir() and name in prefix.parents and not name.is_absolute():
                continue
            if name.is_absolute() or ".." in name.parts or not name.is_relative_to(prefix):
                raise ValueError("archive path leaves the experiment")
            path = root / name
            if not path.parent.resolve().is_relative_to(root):
                raise ValueError("existing parent symlink leaves the destination")
            if member.isdir():
                continue
            entry = expected.pop(member.name, None)
            if entry is None:
                raise ValueError("unexpected archive member")
            if member.isfile():
                content = bundle.extractfile(member).read()
                if hashlib.sha256(content).hexdigest() != entry.get("sha256"):
                    raise ValueError("history member hash mismatch")
                if path.is_symlink() or (
                    path.exists() and (not path.is_file() or path.read_bytes() != content)
                ):
                    raise ValueError(f"refusing to overwrite changed file: {path}")
                plan.append((path, content, member.mode))
            elif member.issym():
                target = member.linkname
                if target != entry.get("symlink") or not (
                    path.parent / target
                ).resolve().is_relative_to(root / prefix):
                    raise ValueError("history symlink leaves the experiment")
                if path.is_symlink():
                    if os.readlink(path) != target:
                        raise ValueError(f"refusing to overwrite changed symlink: {path}")
                elif path.exists():
                    raise ValueError(f"refusing to overwrite file with symlink: {path}")
                plan.append((path, target, None))
            else:
                raise ValueError("unsupported archive member type")
    if expected:
        raise ValueError("history members missing from archive")
    for path, content, mode in plan:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() and not path.is_symlink():
            if mode is None:
                path.symlink_to(content)
            else:
                with path.open("xb") as stream:
                    stream.write(content)
                path.chmod(mode)
    return len(plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt")
    parser.add_argument("--destination", default=".")
    args = parser.parse_args()
    print(f"Verified/restored {restore(args.receipt, args.destination)} historical files")
