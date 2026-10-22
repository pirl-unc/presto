"""Verify the later recovered PRIDE archive without altering the initial audit."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

import datacache


FAMILY = Path(__file__).resolve().parents[2 if Path(__file__).parent.name == "source" else 1]
REPO = FAMILY.parents[1]


def main():
    start = time.perf_counter()
    spec = json.loads((FAMILY / "reproduce/recovered_archive.json").read_text())
    asset = spec["asset"]
    metadata = REPO / "data/availability_sources/snapshots/2026-10-22/PXD009092/files.json"
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == spec["files_metadata_sha256"]
    source = next(f for f in json.loads(metadata.read_text()) if f["fileName"] == "SEARCH.7z")
    assert source["fileSizeBytes"] == asset["size"]
    seed = os.environ.get("SOURCE_ARCHIVE")
    paths = datacache.install_bundle(
        REPO / "data/availability_sources" / asset["sha256"],
        {"PXD009092/SEARCH.7z": asset},
        source_paths={"PXD009092/SEARCH.7z": seed} if seed else None,
        download_options={"show_progress": False},
    )
    archive = Path(paths["PXD009092/SEARCH.7z"])
    assert hashlib.sha1(archive.read_bytes()).hexdigest() == source["checksum"]
    names = subprocess.check_output(["tar", "-tf", str(archive)], text=True).splitlines()
    assert len(names) == len(set(names)) == 13
    assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
    assert names == spec["members"]
    listing = subprocess.check_output(["tar", "-tvf", str(archive)], text=True)
    # Decode all members and check archive CRCs, without creating files or reading PDFs.
    subprocess.run(["tar", "-xOf", str(archive)], stdout=subprocess.DEVNULL, check=True)
    output = FAMILY / "recovery"
    output.mkdir(exist_ok=True)
    (output / "archive_members.txt").write_text(listing)
    receipt = {
        "passed": True, "size_bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_checksum_matching_sha1": source["checksum"],
        "n_members": len(names), "archive_crc_decode": "passed",
        "archive_path": str(archive.relative_to(REPO)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - start,
        "scope": "recovered original archive integrity only; contents not qualified as training data",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
