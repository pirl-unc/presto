"""Sequential reads retain source integrity and the canonical training trajectory."""

from pathlib import Path
import json
import os

import pytest
import torch

from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training import modular_run as runner
from test_modular_resume import manifest_fixture, assert_same


def split_fixture(tmp_path):
    manifest = manifest_fixture(tmp_path, replay=False)
    value = json.loads(manifest.read_text())
    original = value["sources"][0]
    lines = Path(original["path"]).read_bytes().splitlines(keepends=True)
    sources = []
    for name, content in (("first", lines[:5]), ("second", lines[5:])):
        path = tmp_path / (name + ".jsonl")
        path.write_bytes(b"".join(content))
        sources.append({**original, "path": str(path), "sha256": file_sha256(path)})
    value["sources"] = sources
    value["model"].update(d_model=32, n_layers=1, n_heads=4, d_ff=64)
    manifest.write_text(json.dumps(value))
    return ObservationIndex(manifest)


def test_streams_and_random_access_have_identical_content_and_order(tmp_path):
    index = split_fixture(tmp_path)
    expected = [index[i].to_dict() for i in range(len(index))]
    first, second = iter(index), iter(index)
    assert next(first).to_dict() == expected[0]
    assert next(first).to_dict() == expected[1]
    assert next(second).to_dict() == expected[0]
    assert [row.to_dict() for row in second] == expected[1:]
    assert [row.to_dict() for row in first] == expected[2:]
    assert [row.to_dict() for row in index] == expected


@pytest.mark.parametrize("change", ["rewrite", "atomic_replace", "next_source"])
def test_changed_sources_are_rejected_even_with_buffered_rows(tmp_path, change):
    index = split_fixture(tmp_path)
    iterator = iter(index)
    next(iterator)
    path = index.source_paths[1 if change == "next_source" else 0]
    if change == "atomic_replace":
        info = path.stat()
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        replacement.replace(path)
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned input changed"):
        list(iterator)
    with pytest.raises(ValueError, match="pinned input changed"):
        index[5 if change == "next_source" else 0]


def test_closing_partial_iterator_releases_its_input(tmp_path, monkeypatch):
    index = split_fixture(tmp_path)
    real_open = Path.open
    opened = []

    def tracked_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        if path in index.source_paths:
            opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", tracked_open)
    iterator = iter(index)
    next(iterator)
    assert opened
    iterator.close()
    assert all(stream.closed for stream in opened)


def test_common_fit_matches_original_iteration_exactly(tmp_path, monkeypatch):
    index = split_fixture(tmp_path)

    def legacy_iterator(self):
        for i in range(len(self)):
            yield self[i]

    with monkeypatch.context() as patch:
        patch.setattr(ObservationIndex, "__iter__", legacy_iterator)
        expected = runner.fit_index(index, tmp_path / "legacy")
    actual = runner.fit_index(index, tmp_path / "streamed")
    assert actual["metrics"] == expected["metrics"]
    assert actual["steps"] == expected["steps"]
    for name in ("initial.pt", "last.pt", "selected.pt"):
        left = torch.load(tmp_path / "legacy" / name, weights_only=True)
        right = torch.load(tmp_path / "streamed" / name, weights_only=True)
        for key in ("state_dict", "optimizer", "element_coverage"):
            if key in left:
                assert_same(left[key], right[key])
