"""The declared package list must match the filesystem.

`pyproject.toml` maps the repo root onto the `presto` package via
`package-dir = { "presto" = "." }`. Setuptools' `packages.find` cannot express
that mapping -- discovery reports `cli`, `data`, ... as top-level packages and
`import presto` stops working -- so the package list is enumerated by hand.

Hand-enumerated lists drift, and this one had: `presto.scripts.distributional_ba`
and `presto.scripts.distributional_ba.heads` both have `__init__.py` and were
both missing. Editable installs hide it, because the path hook makes the whole
tree importable regardless; a wheel or sdist ships without those modules and
`from presto.scripts.distributional_ba.heads import ...` fails at import.

CI installs with `-e .`, so CI could never have caught it.
"""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories that are not shipped and so are not expected in the list.
# In particular, experiment preparation writes frozen source trees to artifacts/.
NOT_SHIPPED = {"tests", "experiments", "artifacts", "build", "presto_old_code_starts"}


def _declared() -> set[str]:
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return set(cfg["tool"]["setuptools"]["packages"])


def _on_disk() -> set[str]:
    found = {"presto"}
    for init in REPO_ROOT.rglob("__init__.py"):
        rel = init.relative_to(REPO_ROOT).parent
        parts = rel.parts
        if not parts:
            continue  # the root __init__.py is `presto` itself
        if parts[0] in NOT_SHIPPED or any(p.startswith(".") for p in parts):
            continue
        found.add("presto." + ".".join(parts))
    return found


def test_every_package_on_disk_is_declared():
    missing = sorted(_on_disk() - _declared())
    assert missing == [], (
        f"these packages have __init__.py but are not in pyproject's `packages` "
        f"list, so a wheel or sdist ships without them: {missing}"
    )


def test_no_declared_package_is_missing_from_disk():
    """The other direction: a stale entry breaks the build outright."""
    stale = sorted(_declared() - _on_disk())
    assert stale == [], f"declared but not on disk: {stale}"


def test_discovery_ignores_only_the_generated_artifact_root(tmp_path, monkeypatch):
    for name in (
        "artifacts/run/prepared/commit/presto/__init__.py",
        "data/__init__.py",
        "data/artifacts/__init__.py",
        "new_package/__init__.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)

    assert _on_disk() == {"presto", "presto.data", "presto.data.artifacts", "presto.new_package"}


def test_preparing_snapshot_preserves_checkout_package_discovery(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    declared = _declared()
    for package in declared:
        directory = checkout.joinpath(*package.split(".")[1:])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()
    for name in ("pyproject.toml", "README.md", "__main__.py", "data/b2m_sequences.csv"):
        shutil.copy2(REPO_ROOT / name, checkout / name)
    (checkout / ".gitignore").write_text("artifacts/\n__pycache__/\n")
    launcher = Path("experiments/2026-09-09_1541_codex_canonical-coverage/code/launch.py")
    (checkout / launcher).parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / launcher, checkout / launcher)

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=checkout, check=True, capture_output=True, text=True
        )

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Packaging test",
        "-c",
        "user.email=packaging@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "Fixture for source preparation",
    )
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", checkout)
    assert _on_disk() == _declared()

    prepared = subprocess.run(
        [sys.executable, str(checkout / launcher), "prepare"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot = Path(prepared.stdout.strip())
    assert snapshot.is_relative_to(checkout / "artifacts")
    for package in declared:
        assert snapshot.joinpath(*package.split(".")[1:], "__init__.py").is_file()
    assert not git("status", "--porcelain").stdout
    assert _on_disk() == _declared()


def test_pytest_uses_importlib_mode(pytestconfig):
    """Sibling checkout roots must not shadow their editable packages.

    Pytest's default prepend mode puts the repository parent on ``sys.path``.
    Given sibling ``presto/`` and ``hitlist/`` checkouts, ``import hitlist``
    then resolves the latter checkout root as an empty namespace package and
    never reaches its editable package finder.
    """
    assert pytestconfig.getoption("importmode") == "importlib"


@pytest.mark.parametrize(
    "module",
    [
        "presto.scripts.distributional_ba",
        "presto.scripts.distributional_ba.heads",
    ],
)
def test_the_previously_missing_packages_import(module):
    """Named explicitly: these are the two that had drifted out."""
    __import__(module)
