"""Small software boundaries for the independent checker and guarded launcher."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch
from replay_support import numeric, relation


@pytest.mark.parametrize(
    "literal,expected",
    [
        ("NA", (None, "missing")),
        ("0", (0.0, "zero")),
        ("-0.000", (0.0, "zero")),
        ("1e-9999", (None, "outside_float_range")),
        ("1e9999", (None, "outside_float_range")),
        ("Infinity", (None, "nonfinite")),
        ("bad", (None, "invalid")),
        ("12.5", (12.5, "positive")),
        ("-2.5", (-2.5, "negative")),
    ],
)
def test_raw_numeric_states(literal, expected):
    assert numeric(literal) == expected


def test_rational_tolerance_keeps_exact_differences():
    from fractions import Fraction

    assert relation("0.1", "0.100", "1e-12", "1e-12") == ("decimal_equal", 0)
    assert relation("1", "1.000000000002", "1e-12", "1e-12") == (
        "within_numeric_tolerance",
        -Fraction(2, 10**12),
    )
    assert relation("1", "1.000000000003", "1e-12", "1e-12")[0] == "different"
    assert relation("NA", "0", "1e-12", "1e-12") == ("aligned_missing", None)


def test_memory_probe_excludes_inactive_pages(monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "darwin")

    def output(command, **kwargs):
        if command[0].endswith("sysctl"):
            return b"16384\n"
        return "Pages free: 2.\nPages speculative: 3.\nPages inactive: 999999.\n"

    monkeypatch.setattr(launch.subprocess, "check_output", output)
    assert launch.available_memory() == {
        "metric": "macos_free_plus_speculative",
        "bytes": 5 * 16384,
    }


def test_memory_probe_fails_closed_if_counter_is_missing(monkeypatch):
    monkeypatch.setattr(launch.sys, "platform", "darwin")
    monkeypatch.setattr(
        launch.subprocess,
        "check_output",
        lambda cmd, **kw: b"16384" if cmd[0].endswith("sysctl") else "Pages inactive: 999999.\n",
    )
    with pytest.raises(RuntimeError, match="Missing macOS memory counter"):
        launch.available_memory()


def test_preflight_never_launches_or_overwrites_a_refusal(tmp_path, monkeypatch):
    for name in ("code", "reproduce", "results", "launch_logs"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(launch, "__file__", str(tmp_path / "code/launch.py"))
    monkeypatch.setattr(launch.sys, "argv", ["launch.py", "build", "--attempt", "1"])
    monkeypatch.setattr(
        launch, "available_memory", lambda: {"metric": "synthetic_test_only", "bytes": 0}
    )
    (tmp_path / "reproduce/launch.json").write_text(
        json.dumps({"limits": {"startup_memory_bytes": 3 * 1024**3}})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A refused preflight must not launch a child")

    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    assert launch.main() == 2
    path = tmp_path / "results/build-attempt1-runtime.json"
    before = path.read_bytes()
    receipt = json.loads(before)
    assert receipt["status"] == "preflight_refused"
    assert receipt["child_launched"] is False and receipt["refusal"] == "startup_memory_floor"
    assert not list((tmp_path / "launch_logs").iterdir())
    with pytest.raises(FileExistsError):
        launch.main()
    assert path.read_bytes() == before
