"""Historical recovery verifies bytes and preserves conflicting local work."""

import hashlib
import io
import json
import tarfile

import pytest

from presto.scripts.restore_experiment import restore


def archive(tmp_path, *, member_name="experiments/fixture/code.py"):
    payload = b"original evidence\n"
    path = tmp_path / "history.tar.gz"
    with tarfile.open(path, "w:gz") as bundle:
        parent = tarfile.TarInfo("experiments/")
        parent.type = tarfile.DIRTYPE
        bundle.addfile(parent)
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    receipt = tmp_path / "history.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "presto.history-archive.v1",
                "archive": path.name,
                "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prefix": "experiments/fixture",
                "files": [{"path": member_name, "sha256": hashlib.sha256(payload).hexdigest()}],
            }
        )
    )
    return receipt


def test_history_round_trip_and_refusal_to_replace_local_work(tmp_path):
    receipt = archive(tmp_path)
    destination = tmp_path / "restored"
    assert restore(receipt, destination) == 1
    assert restore(receipt, destination) == 1
    path = destination / "experiments/fixture/code.py"
    path.write_text("local work")
    with pytest.raises(ValueError, match="overwrite changed file"):
        restore(receipt, destination)
    assert path.read_text() == "local work"


def test_history_rejects_traversal_before_writing(tmp_path):
    receipt = archive(tmp_path, member_name="experiments/fixture/../../outside.py")
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="leaves the experiment"):
        restore(receipt, destination)
    assert not destination.exists()
