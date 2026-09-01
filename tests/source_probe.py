"""Helpers for tests that assert against module source.

Inspecting source is a last resort -- for logic buried in a closure or a
batched path where a behavioural assertion cannot reach. When it is the only
option, the danger is anchoring:

    start = source.index("outputs = model(")     # the FIRST one
    assert "forbidden=" not in source[start:end]

`str.index` returns the first match and silently ignores the rest, so the test
reads as though it covers the module while checking one site. That is how PR
#10 shipped a train/serve skew: it fixed the tiled flank path, and the test
anchored on that path's call while the single-peptide path -- which encodes
through a different helper -- kept the bug (presto#18).

These helpers make ambiguity an error instead of a silent narrowing. Use them
rather than `.index()`; `tests/test_test_quality.py` enforces that.
"""

from __future__ import annotations

import re


def occurrences(source: str, marker: str) -> list[int]:
    """Every offset of ``marker`` in ``source``."""
    return [m.start() for m in re.finditer(re.escape(marker), source)]


def unique_index(source: str, marker: str, *, where: str = "the source") -> int:
    """Offset of ``marker``, requiring it to appear exactly once.

    The count assertion is the point. A marker that has become ambiguous means
    the code grew a second site, and the caller's conclusion no longer covers
    the module -- which should fail loudly rather than quietly check the first.
    """
    found = occurrences(source, marker)
    assert found, f"{marker!r} does not appear in {where}"
    assert len(found) == 1, (
        f"{marker!r} appears {len(found)} times in {where}; this check assumes "
        "one site, so it would silently cover only the first. Either assert "
        "over every occurrence, or narrow the marker."
    )
    return found[0]


def region_between(source: str, start: str, end: str, *, where: str = "the source") -> str:
    """The slice between two markers, each required to be unique."""
    begin = unique_index(source, start, where=where)
    finish = unique_index(source, end, where=where)
    assert begin < finish, f"{start!r} appears after {end!r} in {where}; the region is inverted"
    return source[begin:finish]


def assert_every_occurrence(
    source: str, marker: str, requirement: str, *, window: int = 300, minimum: int = 1
) -> None:
    """Require ``requirement`` near **every** occurrence of ``marker``.

    The enumerating counterpart to the anchoring idiom: use it when a construct
    legitimately appears at several call sites and all of them must comply.
    ``minimum`` guards the other half of the failure -- a marker that stops
    matching entirely would otherwise vacuously pass.
    """
    found = occurrences(source, marker)
    assert len(found) >= minimum, (
        f"expected at least {minimum} occurrence(s) of {marker!r}, found "
        f"{len(found)}; if a call site was removed, update this test on purpose"
    )
    offenders = [offset for offset in found if requirement not in source[offset : offset + window]]
    assert not offenders, (
        f"{len(offenders)} of {len(found)} occurrence(s) of {marker!r} lack "
        f"{requirement!r} within {window} characters: offsets {offenders}"
    )
