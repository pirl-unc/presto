"""Compress transport bytes only; preserve every pinned input byte after extraction."""

from pathlib import Path
import gzip
import hashlib
import json
import tarfile
import time

F = Path(__file__).resolve().parents[1]
UPLOAD = F.parents[1] / "artifacts" / F.name / "upload"


def main():
    started = time.monotonic()
    destination = UPLOAD / "inputs.tar.gz"
    assert not destination.exists()
    files = sorted(p for p in (UPLOAD / "inputs").rglob("*") if p.is_file())
    with destination.open("xb") as output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    assert not path.is_symlink()
                    info = archive.gettarinfo(
                        str(path), arcname=str(path.relative_to(UPLOAD / "inputs"))
                    )
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    hashes = {
        str(p.relative_to(UPLOAD / "inputs")): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in files
    }
    with tarfile.open(destination) as archive:
        assert set(archive.getnames()) == set(hashes)
        for member in archive:
            assert member.isfile()
            assert (
                hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                == hashes[member.name]
            )
    receipt = dict(
        status="passed",
        archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        archive_bytes=destination.stat().st_size,
        expanded_bytes=sum(p.stat().st_size for p in files),
        files=hashes,
        seconds=time.monotonic() - started,
        scope="Lossless transport only; no observations or source-pinned runtime changed",
    )
    (F / "reproduce/input-transport.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
