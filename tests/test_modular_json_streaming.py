"""JSON cache serialization must retain contracts without duplicating large documents."""

import hashlib
import json
import tracemalloc

import pytest

from presto.training.modular_resume import atomic_json, digest


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"z": [None, True, False, 1, -2, 0.125, 1e-30], "a": {"text": "snowman ☃\nline"}},
        {"records": [{"prediction": -0.0, "source": "é", "labels": ["a", "b"]}] * 3},
    ],
)
def test_json_bytes_and_digest_match_existing_contract(tmp_path, value):
    destination = tmp_path / "cache.json"
    atomic_json(destination, value)
    expected = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    assert destination.read_bytes() == expected
    canonical = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    assert digest(value) == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_failed_serialization_preserves_destination_and_cleans_temporary(tmp_path, bad):
    destination = tmp_path / "cache.json"
    previous = b'{"previous": true}\n'
    destination.write_bytes(previous)
    with pytest.raises((ValueError, TypeError)):
        atomic_json(destination, {"first": ["already serialized"] * 10, "last": bad})
    assert destination.read_bytes() == previous
    assert not destination.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("action", ["persist", "digest"])
def test_large_cache_encoding_uses_bounded_extra_memory(tmp_path, action):
    # Input exists before tracing. A full encoded document is over5MiB, while
    # incremental serialization needs only a chunk and the filesystem buffer.
    value = {"records": [{"text": "x" * 1024, "prediction": 0.25}] * 5000}
    destination = tmp_path / "cache.json"
    tracemalloc.start()
    try:
        if action == "persist":
            atomic_json(destination, value)
        else:
            digest(value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024**2, f"encoding duplicated the document: {peak}bytes"
    if action == "persist":
        assert destination.stat().st_size > 5 * 1024**2
        assert json.loads(destination.read_text()) == value
