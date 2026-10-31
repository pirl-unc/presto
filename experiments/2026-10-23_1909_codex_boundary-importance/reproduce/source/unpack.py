"""Restore exact archived run artifacts without overwriting differing files."""
import hashlib
import json
import tarfile
from pathlib import Path

family = Path(__file__).resolve().parents[1]
receipt = json.loads((family / "results/raw-artifacts.json").read_text())
archive = family / receipt["archive"]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt["archive_sha256"]
with tarfile.open(archive, "r:gz") as stream:
    assert set(stream.getnames()) == set(receipt["files"])
    for member in stream.getmembers():
        assert member.isfile() and member.name.startswith("results/")
        assert ".." not in Path(member.name).parts
        data = stream.extractfile(member).read()
        expected = receipt["files"][member.name]
        assert len(data) == expected["bytes"]
        assert hashlib.sha256(data).hexdigest() == expected["sha256"]
        destination = family / member.name
        if destination.exists():
            assert destination.read_bytes() == data, destination
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
print(f"Verified/restored {len(receipt['files'])} exact run artifacts.")
