"""Ensure a failed/stale predecessor cannot launch a neural fit or erase logs."""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "species_continuation", Path(__file__).with_name("continue.py")
)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


@pytest.mark.parametrize("status, error", [("failed", RuntimeError), ("running", TimeoutError)])
def test_predecessor_failure_stops_training(tmp_path, monkeypatch, status, error):
    repro = tmp_path / "reproduce"
    repro.mkdir()
    predecessor = tmp_path / "predecessor.json"
    predecessor.write_text(
        json.dumps(
            {
                "status": status,
                "started_utc": (datetime.now(timezone.utc) - timedelta(hours=26)).isoformat(),
            }
        )
    )
    monkeypatch.setattr(continuation, "REPRO", repro)
    monkeypatch.setattr(continuation, "FAMILY", tmp_path)
    monkeypatch.setattr(
        continuation, "check", lambda: {"predecessor": str(predecessor), "python": sys.executable}
    )
    calls = []
    monkeypatch.setattr(continuation, "execute", lambda args, log, timeout: calls.append(args))
    with pytest.raises(error):
        continuation.main()
    assert json.loads((repro / "continuation-state.json").read_text())["status"] == "failed"
    assert len(calls) == 1 and calls[0][1].endswith("close.py")


def test_existing_log_prevents_retry(tmp_path, monkeypatch):
    (tmp_path / "launch_logs").mkdir()
    log = tmp_path / "launch_logs/seed42.log"
    log.write_text("original failure evidence\n")
    monkeypatch.setattr(continuation, "FAMILY", tmp_path)
    monkeypatch.setattr(continuation, "check", lambda: {})
    with pytest.raises(FileExistsError):
        continuation.execute(["must-never-execute"], "seed42.log", 1)
    assert log.read_text() == "original failure evidence\n"
