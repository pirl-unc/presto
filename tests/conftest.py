"""Shared pytest configuration.

Determinism. A recurring class of flake in this suite is a numeric assertion
made over unseeded randomness: a randomly-initialized model compared against
itself at a tight tolerance, or `assert (grad != 0).any()` on random inputs
through a random head. Those pass almost always and fail occasionally, and
under xdist the failure lands on a different test each run, which makes it look
like an infrastructure problem rather than a missing seed.

Four such tests were found and seeded individually before this fixture existed
(`test_groove_baseline`, `test_distributional_ba` x2, `test_trainer`). Seeding
once per test removes the whole class rather than the instances, and costs
nothing for tests that do not use randomness.

This does not make the tests *correct* — an assertion that only holds for one
seed is still a weak assertion — but it makes them reproducible, so a failure
means something changed rather than that a draw went badly.
"""

import sys
from pathlib import Path

import pytest

# Make sibling test helpers importable as top-level modules.
#
# `pyproject.toml` maps the `presto` package to the repo root, so `tests/` sits
# inside the package and `import tests.source_probe` does not resolve under
# xdist. Several tests already did `sys.path.insert(0, "tests")` inline, which
# depends on the working directory; doing it once here, from this file's own
# location, does not.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


@pytest.fixture(autouse=True)
def _deterministic_torch_seed():
    """Seed torch before every test."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    torch.manual_seed(0)
