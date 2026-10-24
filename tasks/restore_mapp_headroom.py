"""Atomically replace owned duplicate snapshots with independent APFS CoW clones."""

import ctypes
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path("/private/tmp/presto-modular-model")
DATA = ROOT / "data/mapp2025_source"
RECEIPT = ROOT / "tasks/mapp-storage-headroom-receipt.json"
LIB = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
LIB.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
LIB.clonefile.restype = ctypes.c_int


def fingerprint(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return {"sha256": value.hexdigest(), "size_bytes": path.stat().st_size}


def clone(source, destination):
    if LIB.clonefile(os.fsencode(source), os.fsencode(destination), 0):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def main():
    if RECEIPT.exists():
        raise FileExistsError("Do not overwrite a prior maintenance receipt")
    canonical = DATA / "2026-10-24_2210_canonical"
    manifest = json.loads((canonical / "manifest.json").read_text())
    pairs = []
    manifests = {str(canonical / "manifest.json"): fingerprint(canonical / "manifest.json")}
    for directory in ["2026-10-24_2210_baseline_candidates", "2026-10-24_2244_composition"]:
        package = DATA / directory
        m = json.loads((package / "manifest.json").read_text())
        manifests[str(package / "manifest.json")] = fingerprint(package / "manifest.json")
        assert (
            m["source_package_manifest_sha256"]
            == manifests[str(canonical / "manifest.json")]["sha256"]
        )
        for relative, expected in m["files"].items():
            if relative.startswith("source_tables/") and expected["size_bytes"] >= 1024**2:
                source = canonical / "goldberg_mapp/typed" / Path(relative).name
                assert manifest["files"][str(source.relative_to(canonical))] == expected
                pairs.append((source, package / relative, expected))
    for relative, expected in manifest["files"].items():
        if (
            relative.startswith("goldberg_mapp/source_assets/")
            and expected["size_bytes"] >= 1024**2
        ):
            pairs.append(
                (DATA / "2026-10-24_2130" / Path(relative).name, canonical / relative, expected)
            )
    for source, target, expected in pairs:
        assert (
            source.is_file()
            and target.is_file()
            and not source.is_symlink()
            and not target.is_symlink()
        )
        assert source.stat().st_dev == target.stat().st_dev
        assert fingerprint(source) == fingerprint(target) == expected
    with tempfile.TemporaryDirectory(prefix="mapp-clone-isolation-", dir=ROOT / "tasks") as scratch:
        source = Path(scratch) / "source"
        target = Path(scratch) / "target"
        source.write_bytes(b"A" * 65536)
        clone(source, target)
        with target.open("r+b") as stream:
            stream.write(b"B")
        assert source.read_bytes() == b"A" * 65536 and target.read_bytes() == b"B" + b"A" * 65535
        assert source.stat().st_ino != target.stat().st_ino and target.stat().st_nlink == 1
    result = {
        "status": "in_progress",
        "started_unix": time.time(),
        "free_before_bytes": shutil.disk_usage(ROOT).free,
        "isolation_test_passed": True,
        "pairs": [],
        "manifests": manifests,
    }
    RECEIPT.write_text(json.dumps(result, indent=2) + "\n")
    for source, target, expected in pairs:
        old = target.stat()
        source_before = source.stat()
        temporary = target.with_name(target.name + ".cow-" + uuid.uuid4().hex)
        try:
            clone(source, temporary)
            os.chmod(temporary, stat.S_IMODE(old.st_mode))
            os.utime(temporary, ns=(old.st_atime_ns, old.st_mtime_ns))
            assert fingerprint(temporary) == fingerprint(source) == expected
            assert source.stat().st_mtime_ns == source_before.st_mtime_ns
            assert (
                source.stat().st_ino != temporary.stat().st_ino and temporary.stat().st_nlink == 1
            )
            os.replace(temporary, target)
            assert fingerprint(target) == expected
            result["pairs"].append(
                {
                    "source": str(source),
                    "target": str(target),
                    "fingerprint": expected,
                    "source_inode": source.stat().st_ino,
                    "target_inode": target.stat().st_ino,
                }
            )
            RECEIPT.write_text(json.dumps(result, indent=2) + "\n")
        finally:
            if temporary.exists():
                temporary.unlink()
    # Verify every file of all three packages, not just the changed snapshots.
    checked = 0
    for manifest_path, expected in manifests.items():
        path = Path(manifest_path)
        assert fingerprint(path) == expected
        for relative, record in json.loads(path.read_text())["files"].items():
            assert fingerprint(path.parent / relative) == record
            checked += 1
    result.update(
        status="complete",
        n_package_files_verified=checked,
        free_after_bytes=shutil.disk_usage(ROOT).free,
        finished_unix=time.time(),
        minimum_suite_headroom_bytes=2 * 1024**3,
    )
    result["suite_headroom_passed"] = (
        result["free_after_bytes"] > result["minimum_suite_headroom_bytes"]
    )
    RECEIPT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("pairs", "manifests")}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
