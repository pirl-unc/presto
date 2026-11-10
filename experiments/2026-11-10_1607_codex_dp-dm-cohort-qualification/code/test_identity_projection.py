"""The identity scanner must never deserialize ignored outcome values."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "dp_dm_identity_projection", Path(__file__).with_name("identity_projection.py")
)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


def test_outcome_changes_do_not_change_projection(monkeypatch):
    allowed = frozenset({"peptide", "study_id"})
    row = {
        "value": {"nested": ["FORBIDDEN_OUTCOME", {"x": 'escaped \\" { ]'}]},
        "peptide": "ACDEFGHIKLM",
        "study_id": "study-1",
        "target": 234.5,
    }
    other = {**row, "value": "DIFFERENT_OUTCOME", "target": -999}
    raw1, raw2 = json.dumps(row).encode(), json.dumps(other).encode()
    original = json.loads
    decoded = []

    def tracked(value, *args, **kwargs):
        decoded.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(projection.json, "loads", tracked)
    assert projection.project_top_level_json(raw1, allowed) == {
        "peptide": "ACDEFGHIKLM",
        "study_id": "study-1",
    }
    assert projection.project_top_level_json(raw2, allowed) == {
        "peptide": "ACDEFGHIKLM",
        "study_id": "study-1",
    }
    assert not any(
        "FORBIDDEN_OUTCOME" in value or "DIFFERENT_OUTCOME" in value
        for value in decoded
    )
    assert "234.5" not in decoded and "-999" not in decoded


def test_outcome_allowlist_is_rejected():
    with pytest.raises(ValueError, match="outcome-like field"):
        projection.project_top_level_json(
            b'{"peptide":"ACD","value":1}', frozenset({"peptide", "value"})
        )
