"""Preserve a completed candidate source download; no table admission."""

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
    request = json.loads((FAMILY / "reproduce/tryptic-retrieval.json").read_text())
    metadata = REPO / request["source_metadata"]
    frozen = json.loads((FAMILY / "reproduce/input_assets.json").read_text())
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == frozen["assets"]["PXD009360/files.json"]["sha256"]
    source = next(f for f in json.loads(metadata.read_text()) if f["fileName"] == "tryptic-search.tar.gz")
    path = Path(os.environ.get("SOURCE_ARCHIVE", request["destination"]))
    assert path.stat().st_size == source["fileSizeBytes"] == request["expected_bytes"]
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha1.update(block)
            sha256.update(block)
    assert sha1.hexdigest() == source["checksum"] == request["source_checksum"]
    subprocess.run(["gzip", "-t", str(path)], check=True)
    listing = subprocess.check_output(["tar", "-tzf", str(path)], text=True)
    names = listing.splitlines()
    assert len(names) == len(set(names))
    assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
    asset = {"url": request["url"], "size": path.stat().st_size, "sha256": sha256.hexdigest(), "decompress": False}
    paths = datacache.install_bundle(REPO / "data/availability_sources" / asset["sha256"],
        {"PXD009360/tryptic-search.tar.gz": asset}, source_paths={"PXD009360/tryptic-search.tar.gz": str(path)})
    output = FAMILY / "retrievals"
    output.mkdir(exist_ok=True)
    (output / "tryptic_members.txt").write_text(listing)
    receipt = {"passed": True, "asset": asset, "matching_source_sha1": sha1.hexdigest(),
        "gzip_integrity": "passed", "n_members": len(names),
        "immutable_path": str(Path(paths["PXD009360/tryptic-search.tar.gz"]).relative_to(REPO)),
        "seconds": time.perf_counter() - start,
        "scope": "transfer integrity only; processed tables and material mapping not yet audited",
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "tryptic_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
