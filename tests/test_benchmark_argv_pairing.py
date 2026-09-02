"""Benchmark argv tuples must stay well-formed flag/value pairs.

`DesignSpec.extra_args` and its siblings are the reproducibility contract
AGENTS.md requires be frozen: they record exactly what each experiment arm ran.
They are written as flat tuples of alternating flag and value --

    ("--d-model", "128", "--affinity-loss-mode", "full")

-- and `ruff format` exploded them to one element per line, so the visual
pairing that made a mistake obvious is gone. Delete or duplicate a single line
in a future edit and every subsequent pair shifts by one: `--affinity-loss-mode`
silently receives `"mhcflurry"`, the run launches with a wrong-but-valid config,
and nothing catches it.

Restoring the layout would mean re-pairing 1,216 lines across `scripts/` and
would last only until the next `ruff format`. Validating the structure catches
the same mistake permanently and regardless of formatting, so that is what this
does.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _argv_tuples():
    """(file, line, elements) for every literal tuple/list that looks like argv.

    Parsed rather than imported: these modules use bare sibling imports
    (`from experiment_registry import ...`) that only resolve when they are run
    as scripts, so importing them here fails. Static parsing also avoids any
    import side effects, and reaches tuples that live inside function bodies.

    "Looks like argv" means the first element is a string literal starting with
    `--`. That is precise enough to skip ordinary tuples and broad enough to
    catch every arm spec without naming the classes that hold them.
    """
    out = []
    for path in sorted(SCRIPTS_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - not our code to fix
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Tuple, ast.List)):
                continue
            elements = node.elts
            if len(elements) < 2:
                continue
            if not all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elements):
                continue
            values = [e.value for e in elements]
            if not values[0].startswith("--"):
                continue
            out.append((path.name, node.lineno, tuple(values)))
    return out


_CASES = _argv_tuples()
_IDS = [f"{name}:{line}" for name, line, _ in _CASES]


def test_some_argv_tuples_are_discovered():
    """Guards the guard: a refactor would otherwise make this vacuously pass."""
    assert len(_CASES) >= 10, (
        f"only {len(_CASES)} argv tuples discovered under scripts/; if the arm "
        "specs moved or changed shape, update the detector rather than leaving "
        "this test checking nothing"
    )


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_no_two_consecutive_values(case):
    """No value may directly follow another value.

    Strict flag/value alternation is *not* the contract -- these tuples
    legitimately carry `store_true` flags with no value
    (`--balanced-batches`, `--strict-mhc-resolution`). So the check is the
    weaker invariant that is actually true: two consecutive non-flag elements
    mean a flag was dropped or a value duplicated, which argparse would
    silently accept as a positional or reject far from the cause.

    Honest limitation: deleting a *value* leaves `--flag --nextflag`, which is
    indistinguishable from a legitimate boolean flag, so that specific edit is
    not caught here. Validating arity would need each script's real argparse
    parser, and those modules cannot be imported (bare sibling imports). This
    catches the duplication and shift cases; it does not catch all of them.
    """
    name, line, argv = case
    where = f"{name}:{line}"
    for position in range(1, len(argv)):
        previous, current = argv[position - 1], argv[position]
        if not previous.startswith("--") and not current.startswith("--"):
            raise AssertionError(
                f"{where}: {previous!r} at position {position - 1} is followed "
                f"by another value {current!r}. A flag was dropped or a value "
                "duplicated -- the pairing has shifted."
            )


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_flags_are_well_formed(case):
    """A flag with whitespace is a joined-argument mistake, not a flag."""
    name, line, argv = case
    for element in argv:
        if element.startswith("--"):
            assert " " not in element, (
                f"{name}:{line}: {element!r} contains a space, so a flag and "
                "its value were concatenated into one argv element"
            )
