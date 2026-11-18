"""Restaging must retain original manifest bytes and reject changed inputs."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location("experiment_fit", Path(__file__).with_name("fit.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup_sources(tmp_path):
    source = tmp_path / "original.jsonl"
    source.write_text('{"id":"unchanged"}\n')
    out = tmp_path / "run"
    out.mkdir()
    manifest = {"sources": [{"path": str(source), "sha256": module.sha(source)}]}
    (out / "manifest.json").write_text(json.dumps(manifest))
    path = module.stage_sources(manifest, out, tmp_path / "worker")
    shutil.rmtree(tmp_path / "worker")
    return source, out, path


def test_restore_exact_paths_and_hashes_without_rewriting_contract(tmp_path):
    source, out, path = setup_sources(tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert module.resume_sources(out) == path
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert before == after
    local = Path(json.loads(path.read_text())["sources"][0]["path"])
    assert local.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        module.resume_sources(out)


def test_reject_changed_source_or_manifest(tmp_path):
    source, out, path = setup_sources(tmp_path)
    source.write_text("mutated source")
    with pytest.raises(AssertionError):
        module.resume_sources(out)
    source.write_text('{"id":"unchanged"}\n')
    path.write_text(path.read_text() + " ")
    with pytest.raises(AssertionError):
        module.resume_sources(out)
