"""Package the immutable completed run and verify every archived file."""

import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


FAMILY = Path(__file__).resolve().parents[1]
RESULTS = FAMILY / "results"
RUN = RESULTS / "run"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive = RESULTS / "raw-run.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    expected = json.loads((RUN / "artifacts.json").read_text())
    for name, checksum in expected.items():
        assert digest((RUN / name).read_bytes()) == checksum, name
    entries = {}
    with archive.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for path in sorted(RUN.rglob("*")):
                    if not path.is_file():
                        continue
                    assert not path.is_symlink(), path
                    name = path.relative_to(RESULTS).as_posix()
                    data = path.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    bundle.addfile(info, io.BytesIO(data))
                    entries[name] = digest(data)
    with tarfile.open(archive, "r:gz") as bundle:
        assert len(bundle.getmembers()) == len(entries)
        for member in bundle.getmembers():
            assert member.isfile() and member.name in entries
            assert not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
            assert digest(bundle.extractfile(member).read()) == entries[member.name], member.name
    for name in (
        "summary.json",
        "execution.json",
        "gradient-probe.json",
        "teacher.json",
        "artifacts.json",
    ):
        shutil.copyfile(RUN / name, RESULTS / name)
    shutil.copyfile(FAMILY / "analysis/verify.py", FAMILY / "reproduce/source/verify.py")
    receipt = {
        "status": "passed",
        "archive": archive.name,
        "archive_sha256": digest(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "files": len(entries),
        "uncompressed_bytes": sum(path.stat().st_size for path in RUN.rglob("*") if path.is_file()),
        "original_manifest_entries_verified": len(expected),
        "archive_members_verified": entries,
        "extraction": "tar -xzf results/raw-run.tar.gz -C results",
        "note": (
            "Paths and bytes checked against the immutable completed run; "
            "no fitting or scoring rerun."
        ),
    }
    (RESULTS / "package-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "archive_members_verified"}
        )
    )


if __name__ == "__main__":
    main()
