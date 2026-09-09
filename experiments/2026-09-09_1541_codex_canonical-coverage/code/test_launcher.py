"""Local launch-boundary checks; these tests never contact Modal."""

import hashlib
import importlib.util
import json
import sys
import tomllib
from contextlib import nullcontext
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


def test_archive_contains_python_sources_for_every_declared_package(launcher):
    project = tomllib.loads((launcher.ROOT / "pyproject.toml").read_text())
    expected = set()
    for package in project["tool"]["setuptools"]["packages"]:
        directory = launcher.ROOT.joinpath(*package.split(".")[1:])
        expected.update(str(path.relative_to(launcher.ROOT)) for path in directory.glob("*.py"))
    archived = set(launcher.archive_paths(sorted(expected)))
    assert expected <= archived, sorted(expected - archived)


def test_archive_follows_added_package_without_changing_launcher(tmp_path, monkeypatch, launcher):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npackages = ["presto", "presto.future_package"]\n'
    )
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    assert launcher.archive_paths(
        ["future_package/__init__.py", "future_package/input.parquet", "data/raw.tsv"]
    ) == ["future_package/__init__.py"]


@pytest.mark.parametrize("attempt", ["", "..", "../retry", "/tmp/retry", "retry/child", "a b"])
def test_invalid_attempt_rejected_before_snapshot_access(tmp_path, launcher, attempt):
    with pytest.raises(ValueError, match="Unsafe or empty attempt"):
        launcher.execute("unused", tmp_path / "missing", attempt)


def test_retry_preserves_initial_receipts_and_cannot_overwrite(tmp_path, monkeypatch, launcher):
    snapshot = tmp_path / "snapshot"
    family = snapshot / "experiments" / launcher.FAMILY
    (family / "code").mkdir(parents=True)
    for name in ("launch.py", "census.py"):
        (family / "code" / name).write_bytes(Path(launcher.__file__).with_name(name).read_bytes())
    remote_initial = f"/results/{launcher.FAMILY}/condition"
    (family / "conditions.json").write_text(
        json.dumps({"condition": {"run_dir": remote_initial, "seed": 42}})
    )
    files = [
        {
            "relative_path": str(path.relative_to(snapshot)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    ]
    (snapshot / "source_receipt.json").write_text(json.dumps({"files": files}))
    local_family = tmp_path / "local"
    monkeypatch.setattr(launcher, "EXPERIMENT", local_family)
    image = SimpleNamespace()
    for method in (
        "pip_install", "pip_install_from_requirements", "add_local_dir", "run_commands", "env"
    ):
        setattr(image, method, lambda *args, **kwargs: image)
    spawned = []
    declarations = []

    def spawn(*args):
        spawned.append(args)
        return SimpleNamespace(object_id="fc-fixture", get=lambda: {"fixture": True})

    def declare(**kwargs):
        declarations.append(kwargs)
        return lambda fn: SimpleNamespace(spawn=spawn)

    app = SimpleNamespace(
        app_id="ap-fixture",
        function=declare,
        run=lambda **kwargs: nullcontext(),
    )
    modal = SimpleNamespace(
        __version__="1.1.4",
        Image=SimpleNamespace(debian_slim=lambda **kwargs: image),
        Volume=SimpleNamespace(from_name=lambda *args, **kwargs: None),
        App=lambda *args: app,
        enable_output=nullcontext,
    )
    destination = {"profile": "iskandr", "workspace": "iskandr", "environment": "main"}
    monkeypatch.setattr(launcher, "modal_destination", lambda: (modal, destination))
    launcher.execute("condition", snapshot)
    initial_path = local_family / "results/condition/handle.json"
    initial_bytes = initial_path.read_bytes()
    launcher.execute("condition", snapshot, "package_manifest")
    retry = json.loads(
        (local_family / "results/condition/attempts/package_manifest/handle.json").read_text()
    )
    remote_retry = remote_initial + "/attempts/package_manifest"
    assert spawned == [
        ("condition", "initial", remote_initial),
        ("condition", "package_manifest", remote_retry),
    ]
    assert all(declaration.get("serialized") is True for declaration in declarations)
    assert retry["args"] == {"run_dir": remote_retry, "seed": 42}
    assert retry["attempt"] == "package_manifest"
    assert retry["status"] == "remote_complete"
    assert retry["finished_unix"] >= retry["started_unix"]
    assert initial_path.read_bytes() == initial_bytes
    with pytest.raises(FileExistsError):
        launcher.execute("condition", snapshot, "package_manifest")
    assert len(spawned) == 2


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


def test_upload_receipt_preserves_verified_destination(tmp_path, monkeypatch, launcher):
    source = tmp_path / "input.txt"
    source.write_text("fixture")
    manifest = [
        {
            "local_path": str(source),
            "remote_path": "/inputs/family/input.txt",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    (tmp_path / "input_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(launcher, "EXPERIMENT", tmp_path)
    monkeypatch.setattr(launcher, "RAW", tmp_path / "receipt")
    uploaded = []

    class Batch:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((local_path, remote_path))

    def volume(name, *, environment_name):
        assert (name, environment_name) == ("presto-data", "main")
        return SimpleNamespace(batch_upload=Batch)

    destination = {"profile": "iskandr", "workspace": "iskandr", "environment": "main"}
    modal = SimpleNamespace(__version__="1.1.4", Volume=SimpleNamespace(from_name=volume))
    monkeypatch.setattr(launcher, "modal_destination", lambda: (modal, destination))
    launcher.upload()
    receipt = json.loads((tmp_path / "receipt/upload.json").read_text())
    assert receipt["destination"] == destination
    assert receipt["files"] == manifest
    assert uploaded == [(str(source), "family/input.txt")]
