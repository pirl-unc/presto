"""Local launch-boundary checks; these tests never contact Modal."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location(
        "census_launch_review", Path(__file__).with_name("launch.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_live_launcher_is_rejected_before_image_preparation(
    tmp_path, monkeypatch, launcher
):
    snapshot = tmp_path / "snapshot"
    family = snapshot / "experiments" / launcher.FAMILY
    (family / "code").mkdir(parents=True)
    (family / "code/launch.py").write_text("# a different archived launcher\n")
    (family / "code/census.py").write_text("# unused worker\n")
    (family / "conditions.json").write_text(json.dumps({"condition": {}}))
    files = [
        {
            "relative_path": str(path.relative_to(snapshot)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    ]
    (snapshot / "source_receipt.json").write_text(json.dumps({"files": files}))
    monkeypatch.setattr(launcher, "EXPERIMENT", tmp_path / "local")

    def image_reached(**_):
        raise AssertionError("image construction reached despite changed live launcher")

    fake_modal = ModuleType("modal")
    fake_modal.__version__ = "1.1.4"
    fake_modal.Image = SimpleNamespace(debian_slim=image_reached)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    with pytest.raises(RuntimeError, match="Executing launcher differs from frozen source"):
        launcher.execute("condition", snapshot)


@pytest.mark.parametrize(
    "profile,workspace,version,message",
    [
        ("other", "iskandr", "1.1.4", "MODAL_PROFILE=iskandr"),
        ("iskandr", "other", "1.1.4", "Authenticated workspace 'other' differs"),
        ("iskandr", "iskandr", "1.1.5", "requires Modal 1.1.4"),
    ],
)
def test_wrong_destination_or_sdk_rejects_upload_before_volume_access(
    monkeypatch, launcher, profile, workspace, version, message
):
    fake_modal = ModuleType("modal")
    fake_modal.__version__ = version
    config = ModuleType("modal.config")
    config._profile = profile
    config.config = {
        "server_url": "unused",
        "token_id": "fixture-id",
        "token_secret": "fixture-secret",
    }
    lookup_calls = []

    async def lookup(*args):
        lookup_calls.append(True)
        return SimpleNamespace(username=workspace)

    config._lookup_workspace = lookup
    # There is deliberately no Volume interface: validation must stop first.
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "modal.config", config)
    with pytest.raises(RuntimeError, match=message):
        launcher.upload()
    assert len(lookup_calls) == int(profile == "iskandr" and version == "1.1.4")


def test_verified_destination_receipt_contains_no_credentials(monkeypatch, launcher):
    fake_modal = ModuleType("modal")
    fake_modal.__version__ = "1.1.4"
    config = ModuleType("modal.config")
    config._profile = "iskandr"
    config.config = {
        "server_url": "unused",
        "token_id": "fixture-id",
        "token_secret": "fixture-secret",
    }

    async def lookup(*args):
        return SimpleNamespace(username="iskandr")

    config._lookup_workspace = lookup
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "modal.config", config)
    _, receipt = launcher.modal_destination()
    assert receipt == {"profile": "iskandr", "workspace": "iskandr", "environment": "main"}


@pytest.mark.parametrize("duplicates", [0, 2])
def test_ambiguous_launcher_receipt_is_rejected(tmp_path, launcher, duplicates):
    relative = f"experiments/{launcher.FAMILY}/code/launch.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(Path(launcher.__file__).read_bytes())
    entry = {"relative_path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "source_receipt.json").write_text(json.dumps({"files": [entry] * duplicates}))
    with pytest.raises(RuntimeError, match="exactly one executing launcher entry"):
        launcher.execute("unused", tmp_path)
