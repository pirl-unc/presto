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

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories that are not shipped and so are not expected in the list.
NOT_SHIPPED = {"tests", "experiments", "build", "presto_old_code_starts"}


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
