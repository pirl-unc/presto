"""Guards against tests that verify the path the author had in mind.

presto#18. Four times in the #7-#17 series I wrote a test that passed against
code still containing the bug it was written to catch. One shipped: PR #10
claimed to fix a train/serve flank skew and fixed one of two call sites,
because the test anchored on the *first* match in the module source and the
other call site used a different helper.

Most of that failure mode is judgement and cannot be mechanized. Two parts can,
and this module enforces those:

1. **Anchoring source inspection on a single match.** `source.index(marker)`
   returns the first hit and silently ignores every other. A test written that
   way asserts something about one call site while reading as though it covers
   the module. If a test inspects source, it must enumerate.

2. **Assertions that cannot fail.** `assert x or True`, `assert True`, and the
   classic `assert (condition, "message")` -- a non-empty tuple, always truthy,
   so the message argument silently disables the check.

3. **Broad exceptions converted into skips.** A test that catches
   ``Exception`` and calls ``pytest.skip`` reports a broken API, corrupt cache,
   or assertion bug as unavailable infrastructure. Only a narrowly identified
   unavailable prerequisite may skip.

All three rules run over this repo's own test suite. They are deliberately narrow:
a meta-test with false positives gets suppressed, and then it guards nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

#: Package directories scanned for un-failable assertions.
#:
#: The anchoring rule is test-specific -- production code has no reason to
#: inspect its own source. An assertion that cannot fail is a defect anywhere,
#: so that rule runs over the package too. `experiments/` is excluded: those
#: are frozen snapshots of scripts as they ran, and editing them to satisfy a
#: linter would falsify the record.
PACKAGE_DIRS = ("data", "models", "training", "scripts", "inference", "cli")

#: Evidence that a source-inspecting test looked at every occurrence rather
#: than the first one.
ENUMERATION_MARKERS = (
    "finditer",
    "findall",
    "count(",
    "splitlines(",
    # The sanctioned helpers in tests/source_probe.py. They assert the marker
    # is unambiguous, or check every occurrence, so a test using them is not
    # anchoring even though it never calls finditer itself.
    "unique_index",
    "region_between",
    "occurrences",
    "assert_every_occurrence",
)


def _test_files() -> list[Path]:
    return sorted(p for p in TESTS_DIR.glob("test_*.py") if p.name != Path(__file__).name)


def _package_files() -> list[Path]:
    found: list[Path] = []
    for name in PACKAGE_DIRS:
        found.extend(sorted((REPO_ROOT / name).rglob("*.py")))
    return found


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _functions(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _calls_getsource(func) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute) and target.attr == "getsource":
                return True
    return False


def _is_pytest_skip_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "skip"
    )


def _catches_broad_exception(handler: ast.ExceptHandler) -> bool:
    caught = handler.type
    if caught is None:
        return True
    if isinstance(caught, ast.Name):
        return caught.id in {"Exception", "BaseException"}
    if isinstance(caught, ast.Tuple):
        return any(
            isinstance(item, ast.Name) and item.id in {"Exception", "BaseException"}
            for item in caught.elts
        )
    return False


def _broad_skip_handlers(tree: ast.Module) -> list[int]:
    return [
        handler.lineno
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler)
        and _catches_broad_exception(handler)
        and any(_is_pytest_skip_call(node) for node in ast.walk(handler))
    ]


def _anchor_calls(func) -> list[int]:
    """Line numbers of `.index(...)` / `.find(...)` calls inside `func`."""
    lines: list[int] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("index", "find"):
                lines.append(node.lineno)
    return lines


class TestSourceInspectionEnumerates:
    """A test that reads module source must check every occurrence.

    This is the rule PR #10 broke. `source.index("flank_n_tok = "
    "self.tokenizer.batch_encode(")` found the tiled path, checked it, and
    passed -- while the single-peptide path, which encodes through a different
    helper, kept the bug the test existed to catch.
    """

    def test_no_test_anchors_source_inspection_on_one_match(self):
        offenders: list[str] = []
        for path in _test_files():
            for func in _functions(_parse(path)):
                if not _calls_getsource(func):
                    continue
                body = ast.get_source_segment(path.read_text(), func) or ""
                if any(marker in body for marker in ENUMERATION_MARKERS):
                    continue
                for lineno in _anchor_calls(func):
                    offenders.append(f"{path.name}:{lineno} in {func.name}()")
        assert offenders == [], (
            "these tests inspect module source and locate their anchor with "
            ".index()/.find(), which silently checks only the first match:\n  "
            + "\n  ".join(offenders)
            + "\n\nEnumerate instead (re.finditer / re.findall / splitlines) "
            "and assert over every occurrence, plus assert how many you expect "
            "so a new call site fails loudly rather than being skipped."
        )


def _always_true_reason(test: ast.expr) -> str | None:
    """Why this assert expression can never fail, or None if it can."""
    if isinstance(test, ast.Constant):
        if test.value:
            return f"constant {test.value!r} is always truthy"
        return None
    if isinstance(test, ast.Tuple) and test.elts:
        return (
            "a non-empty tuple is always truthy -- this is almost certainly "
            'assert (cond, "message") with the parentheses in the wrong place'
        )
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        for value in test.values:
            if isinstance(value, ast.Constant) and value.value:
                return f"`or {value.value!r}` makes the whole expression truthy"
    return None


class TestAssertionsCanFail:
    """An assertion that cannot fail is worse than no assertion.

    It occupies the space where a check should be and reports success. I wrote
    `assert ... or True` during this series and caught it myself only by
    reading the diff back.
    """

    @staticmethod
    def _scan(paths) -> list[str]:
        offenders: list[str] = []
        for path in paths:
            try:
                tree = _parse(path)
            except SyntaxError:  # pragma: no cover - not our code to fix
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assert):
                    continue
                reason = _always_true_reason(node.test)
                if reason:
                    rel = path.relative_to(REPO_ROOT)
                    offenders.append(f"{rel}:{node.lineno} -- {reason}")
        return offenders

    def test_no_test_assertion_is_unconditionally_true(self):
        offenders = self._scan(_test_files())
        assert offenders == [], "these assertions can never fail:\n  " + "\n  ".join(offenders)

    def test_no_package_assertion_is_unconditionally_true(self):
        """Same rule, applied to shipped code.

        An `assert (cond, "message")` in the package is a runtime check that
        silently passes -- worse than in a test, because nothing else is
        watching that path.
        """
        offenders = self._scan(_package_files())
        assert offenders == [], "these assertions can never fail:\n  " + "\n  ".join(offenders)


class TestSkipsAreNarrow:
    """A runtime defect must not be reported as an unavailable prerequisite."""

    def test_no_broad_exception_handler_turns_a_failure_into_a_skip(self):
        offenders: list[str] = []
        for path in _test_files():
            for lineno in _broad_skip_handlers(_parse(path)):
                offenders.append(f"{path.name}:{lineno}")
        assert offenders == [], (
            "these broad exception handlers convert arbitrary test failures "
            "into skips:\n  "
            + "\n  ".join(offenders)
            + "\n\nCatch only the unavailable prerequisite, or let the "
            "original failure fail the test."
        )


class TestTheGuardsThemselvesWork:
    """Fault injection, applied to the meta-tests.

    A guard nobody has seen fail is a guard nobody should trust -- which is the
    whole subject of presto#18, so it would be poor form to skip it here.
    """

    @pytest.mark.parametrize(
        "snippet,expected",
        [
            ("assert True", "always truthy"),
            ("assert 1", "always truthy"),
            ('assert (x == 1, "boom")', "non-empty tuple"),
            ("assert x == 1 or True", "makes the whole expression truthy"),
        ],
    )
    def test_always_true_detector_catches_the_real_shapes(self, snippet, expected):
        node = ast.parse(snippet).body[0]
        assert isinstance(node, ast.Assert)
        reason = _always_true_reason(node.test)
        assert reason is not None and expected in reason

    @pytest.mark.parametrize(
        "snippet",
        [
            "assert x == 1",
            "assert x or y",
            "assert not x",
            'assert "a" in b',
            "assert x == 1, 'message'",
            "assert False",
        ],
    )
    def test_always_true_detector_leaves_real_assertions_alone(self, snippet):
        node = ast.parse(snippet).body[0]
        assert _always_true_reason(node.test) is None

    def test_anchor_detector_flags_the_pr10_shape(self):
        source = (
            "def test_thing():\n"
            "    import inspect\n"
            "    source = inspect.getsource(mod)\n"
            "    start = source.index('marker')\n"
            "    assert 'x' in source[start:start + 100]\n"
        )
        func = ast.parse(source).body[0]
        assert _calls_getsource(func)
        assert _anchor_calls(func) != []

    def test_anchor_detector_accepts_an_enumerating_test(self):
        source = (
            "def test_thing():\n"
            "    import inspect, re\n"
            "    source = inspect.getsource(mod)\n"
            "    sites = [m.start() for m in re.finditer('marker', source)]\n"
            "    assert len(sites) >= 2\n"
        )
        body = source
        assert any(marker in body for marker in ENUMERATION_MARKERS)

    def test_anchor_detector_ignores_list_index_outside_source_inspection(self):
        """`PROCESSING_STIMULI.index("none")` is not source inspection."""
        func = ast.parse("def test_thing():\n    assert VOCAB.index('none') == 0\n").body[0]
        assert not _calls_getsource(func)

    @pytest.mark.parametrize(
        "snippet",
        [
            "try:\n    query()\nexcept Exception:\n    pytest.skip('broken')",
            "try:\n    query()\nexcept BaseException:\n    pytest.skip('broken')",
            "try:\n    query()\nexcept:\n    pytest.skip('broken')",
            "try:\n    query()\nexcept (OSError, Exception):\n    pytest.skip('broken')",
        ],
    )
    def test_broad_skip_detector_catches_the_real_shapes(self, snippet):
        assert _broad_skip_handlers(ast.parse(snippet)) != []

    def test_broad_skip_detector_allows_a_narrow_prerequisite(self):
        source = "try:\n    import optional\nexcept ImportError:\n    pytest.skip('not installed')"
        assert _broad_skip_handlers(ast.parse(source)) == []
