"""Restore exact archived outputs without overwriting differing files."""
import hashlib
import json
import tarfile
from pathlib import Path

family = Path(__file__).resolve().parents[1]
receipt = json.loads((family / "results/raw-artifacts.json").read_text())
seen = set()
for entry in receipt["archives"]:
    archive = family / entry["path"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == entry["sha256"]
    with tarfile.open(archive, "r:gz") as stream:
        assert set(stream.getnames()) == set(entry["members"])
        for member in stream.getmembers():
            assert member.isfile() and member.name.startswith("results/")
            assert ".." not in Path(member.name).parts and member.name not in seen
            seen.add(member.name)
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
assert seen == set(receipt["files"])
print(f"Verified/restored {len(seen)} exact run artifacts.")
