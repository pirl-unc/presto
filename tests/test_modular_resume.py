"""Interrupted-run equivalence and test-use guarantees on artificial data."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from presto.data.modular import AA, MHCInput, Observation
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    file_sha256,
    write_observations,
)
from presto.training import modular_run as runner
from presto.training.modular_resume import run_lock


def manifest_fixture(tmp_path, *, replay=True):
    rows = [
        Observation(
            f"a{i}",
            "software",
            "ic50",
            AA[i:] + AA[:i],
            (MHCInput(AA, AA[::-1], "I"),),
            value=10 ** (1 + i % 4),
            unit="nM",
            split="train" if i < 6 else "validation" if i < 8 else "test",
            group_id=f"a{i}",
            provenance={"original_measurement_id": f"a{i}"},
        )
        for i in range(10)
    ]
    if replay:
        rows += [
            Observation(
                f"i{i}",
                "software",
                "identity",
                "",
                (MHCInput(AA[i:] + AA[:i], AA[::-1]),),
                identity_labels={"mhc_class": "I" if i % 2 else "II"},
                group_id=f"i{i}",
                provenance={"original_measurement_id": f"i{i}"},
            )
            for i in range(4)
        ]
    written = write_observations(tmp_path / "rows.jsonl", rows)
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": {
            "endpoints": ["ic50"],
            "mhc_classes": ["I"],
            "dropout": 0.3,
            "identity_tasks": ["mhc_class"] if replay else [],
        },
        "training": {
            "batch_size": 2,
            "identity_batch_size": 3,
            "max_epochs": 3,
            "max_steps": 8,
            "checkpoint_every_steps": 2,
        },
        "sources": [
            {
                "path": written["path"],
                "sha256": written["sha256"],
                "source": "software",
                "population": "artificial",
                "label_policy": "software fixtures only",
                "split_policy": "disjoint artificial inputs",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def interrupt(monkeypatch, *, before_step=5):
    original = runner.train_step
    calls = []

    def step(*args, **kwargs):
        if len(calls) + 1 == before_step:
            raise RuntimeError("simulated process interruption")
        calls.append(
            [
                [row.observation_id for row in args[2]],
                [row.observation_id for row in kwargs.get("identity_rows", [])],
            ]
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "train_step", step)
    return original, calls


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("before_step", [2, 5, 6])
def test_fresh_process_resume_matches_dropout_replay_and_partial_epoch(
    tmp_path,
    monkeypatch,
    before_step,
):
    path = manifest_fixture(tmp_path)
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    original, trace = interrupt(monkeypatch, before_step=100)
    expected = runner.fit_index(ObservationIndex(path), reference)
    expected_trace = list(trace)
    monkeypatch.setattr(runner, "train_step", original)
    original, before = interrupt(monkeypatch, before_step=before_step)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.fit_index(ObservationIndex(path), resumed)
    state = torch.load(resumed / "resume.pt", weights_only=True)["metadata"]["resume"]["state"]
    if before_step == 2:
        assert state["steps"] == 0 and state["epoch"] == 1 and state["position"] == 0
    else:
        assert state["steps"] == 4 and state["epoch"] == 2 and state["position"] == 2
        assert state["epoch_losses"] and state["replay_position"]
    script = """
import json, sys, torch
from pathlib import Path
from presto.data.modular_sources import ObservationIndex
from presto.training import modular_run as runner
torch.set_num_threads(int(sys.argv[3]))
torch.set_num_interop_threads(int(sys.argv[4]))
original, trace = runner.train_step, []
def step(*args, **kwargs):
    trace.append([[r.observation_id for r in args[2]],
                  [r.observation_id for r in kwargs['identity_rows']]])
    return original(*args, **kwargs)
runner.train_step = step
runner.fit_index(ObservationIndex(sys.argv[1]), sys.argv[2], resume=True)
Path(sys.argv[2], 'resume-trace.json').write_text(json.dumps(trace))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            str(resumed),
            str(torch.get_num_threads()),
            str(torch.get_num_interop_threads()),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    remaining = json.loads((resumed / "resume-trace.json").read_text())
    assert before[: state["steps"]] + remaining == expected_trace
    assert before[state["steps"] :] == remaining[: len(before) - state["steps"]]
    actual = json.loads((resumed / "summary.json").read_text())
    expected.pop("elapsed_seconds")
    actual.pop("elapsed_seconds")
    assert actual == expected
    for name in ("last.pt", "selected.pt"):
        assert_same(
            torch.load(reference / name, weights_only=True),
            torch.load(resumed / name, weights_only=True),
        )
    for name in (
        "history.json",
        "coverage.json",
        "train-predictions.jsonl",
        "validation-predictions.jsonl",
        "test-predictions.jsonl",
    ):
        assert (reference / name).read_bytes() == (resumed / name).read_bytes()
    assert json.loads((resumed / "history.json").read_text())[-1]["complete_epoch"] is False


@pytest.mark.parametrize(
    "changed", ["manifest", "data", "runtime", "implementation", "optimizer", "initial"]
)
def test_resume_rejects_changed_contract_before_updates(tmp_path, monkeypatch, changed):
    path = manifest_fixture(tmp_path, replay=False)
    output = tmp_path / "fit"
    original, _ = interrupt(monkeypatch, before_step=3)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.fit_index(ObservationIndex(path), output)
    if changed in ("manifest", "data"):
        manifest = json.loads(path.read_text())
        if changed == "manifest":
            manifest["training"]["learning_rate"] = 0.01
        else:
            data = tmp_path / "rows.jsonl"
            rows = [json.loads(line) for line in data.read_text().splitlines()]
            rows[0]["value"] *= 2
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest["sources"][0]["sha256"] = file_sha256(data)
        path.write_text(json.dumps(manifest))
    elif changed in ("runtime", "implementation"):
        payload = torch.load(output / "resume.pt", weights_only=True)
        key = "backend" if changed == "runtime" else "implementation"
        payload["metadata"]["resume"]["contract"][key] = {}
        torch.save(payload, output / "resume.pt")
    elif changed == "optimizer":
        payload = torch.load(output / "resume.pt", weights_only=True)
        payload["metadata"]["resume"]["optimizer_names"][0].reverse()
        torch.save(payload, output / "resume.pt")
    else:
        (output / "admission.json").write_text("{}")
    monkeypatch.setattr(runner, "train_step", lambda *a, **k: pytest.fail("unexpected update"))
    with pytest.raises(ValueError, match="contract changed|parameter order|artifact changed"):
        runner.fit_index(ObservationIndex(path), output, resume=True)


def test_evaluation_recovery_reuses_test_cache_and_completed_result(tmp_path, monkeypatch):
    path = manifest_fixture(tmp_path, replay=False)
    output = tmp_path / "fit"
    original = runner._jsonl

    def broken_write(path, rows):
        if Path(path).name == "test-predictions.jsonl":
            raise RuntimeError("report interruption")
        return original(path, rows)

    monkeypatch.setattr(runner, "_jsonl", broken_write)
    with pytest.raises(RuntimeError, match="report interruption"):
        runner.fit_index(ObservationIndex(path), output)
    cache = (output / "evaluations/test.json").read_bytes()
    state = torch.load(output / "resume.pt", weights_only=True)["metadata"]["resume"]["state"]
    assert state["phase"] == "evaluation"
    # This mutable file may be ahead of the durable snapshot after a crash.
    (output / "selected.pt").write_bytes(b"uncommitted selection")
    monkeypatch.setattr(runner, "_jsonl", original)
    monkeypatch.setattr(runner, "evaluate", lambda *a, **k: pytest.fail("unexpected rescoring"))
    monkeypatch.setattr(runner, "train_step", lambda *a, **k: pytest.fail("unexpected update"))
    summary = runner.fit_index(ObservationIndex(path), output, resume=True)
    assert (output / "selected.pt").read_bytes() == state["selected_bytes"]
    assert (output / "evaluations/test.json").read_bytes() == cache
    receipt = (output / "completed.json").read_bytes()
    assert runner.fit_index(ObservationIndex(path), output, resume=True) == summary
    assert (output / "completed.json").read_bytes() == receipt
    (output / "test-predictions.jsonl").write_text("modified evidence")
    with pytest.raises(ValueError, match="completed run artifact changed"):
        runner.fit_index(ObservationIndex(path), output, resume=True)


def test_interrupted_test_evaluation_cannot_be_silently_repeated(tmp_path, monkeypatch):
    path = manifest_fixture(tmp_path, replay=False)
    output = tmp_path / "fit"
    original = runner.evaluate
    test_calls = []

    def crash(model, index, indices, *args, **kwargs):
        if indices and index[indices[0]].split == "test":
            test_calls.append(1)
            original(model, index, indices, *args, **kwargs)
            raise RuntimeError("test result not durable")
        return original(model, index, indices, *args, **kwargs)

    monkeypatch.setattr(runner, "evaluate", crash)
    with pytest.raises(RuntimeError, match="not durable"):
        runner.fit_index(ObservationIndex(path), output)
    monkeypatch.setattr(runner, "train_step", lambda *a, **k: pytest.fail("unexpected update"))
    with pytest.raises(ValueError, match="refusing to rescore"):
        runner.fit_index(ObservationIndex(path), output, resume=True)
    assert len(test_calls) == 1


def test_missing_snapshot_and_concurrent_writer_rejected(tmp_path):
    path = manifest_fixture(tmp_path, replay=False)
    output = tmp_path / "fit"
    with pytest.raises(ValueError, match="existing durable"):
        runner.fit_index(ObservationIndex(path), output, resume=True)
    with run_lock(output), pytest.raises(ValueError, match="another trainer is active"):
        runner.fit_index(ObservationIndex(path), output)


def test_resume_preserves_frozen_functions_and_inherited_scales(tmp_path, monkeypatch):
    from test_modular_continuation import trained_affinity, SCALE
    from presto.training.modular import load_checkpoint

    previous, prior = trained_affinity(tmp_path)
    path = manifest_fixture(tmp_path, replay=False)
    manifest = json.loads(path.read_text())
    rows = [replace(row, endpoint="kd") for row in ObservationIndex(path)]
    data = tmp_path / "rows.jsonl"
    data.unlink()
    written = write_observations(data, rows)
    manifest["sources"][0]["sha256"] = written["sha256"]
    manifest["model"]["endpoints"] = ["ic50", "kd"]
    manifest["pretrained"] = {"path": str(prior), "sha256": file_sha256(prior), "freeze": ["ic50"]}
    path.write_text(json.dumps(manifest))
    output = tmp_path / "fit"
    original, _ = interrupt(monkeypatch, before_step=3)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.fit_index(ObservationIndex(path), output)
    monkeypatch.setattr(runner, "train_step", original)
    result = runner.fit_index(ObservationIndex(path), output, resume=True)
    final, scales, payload = load_checkpoint(output / "selected.pt")
    assert result["coverage_gate"]["status"] == "passed"
    assert scales["ic50"] == SCALE
    assert result["support"]["ic50"]["status"] == "pretrained_frozen"
    assert final.frozen_parameters
    for name in final.frozen_parameters:
        assert torch.equal(final.get_parameter(name), previous.get_parameter(name))
    assert result["training_exposures"] == {"kd": 16}
    assert payload["metadata"]["pretrained_contract"] == manifest["pretrained"]


def test_resume_restores_patience_and_earlier_selection(tmp_path, monkeypatch):
    path = manifest_fixture(tmp_path, replay=False)
    manifest = json.loads(path.read_text())
    manifest["training"].update(max_steps=None, min_epochs=1, patience=1)
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "_selection_score", lambda *a: 1.0)
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    expected = runner.fit_index(ObservationIndex(path), reference)
    original, _ = interrupt(monkeypatch)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.fit_index(ObservationIndex(path), resumed)
    monkeypatch.setattr(runner, "train_step", original)
    actual = runner.fit_index(ObservationIndex(path), resumed, resume=True)
    assert actual["selected_epoch"] == expected["selected_epoch"] == 1
    assert actual["steps"] == expected["steps"] == 6
    assert_same(
        torch.load(reference / "selected.pt", weights_only=True),
        torch.load(resumed / "selected.pt", weights_only=True),
    )
    assert (reference / "history.json").read_bytes() == (resumed / "history.json").read_bytes()


def test_registered_cli_resume_preserves_recipe_and_closes_once(tmp_path, monkeypatch):
    from presto.cli.main import main
    from presto.scripts.train_iedb import main as script_main
    from presto.scripts import experiment_registry as registry

    path = manifest_fixture(tmp_path, replay=False)
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    monkeypatch.setattr(registry, "EXPERIMENTS_ROOT", experiments)
    monkeypatch.setattr(registry, "REGISTRY_MD", experiments / "experiment_log.md")
    output = experiments / "2026-10-21_0000_codex_resume-test"
    args = SimpleNamespace(observations_manifest=str(path), run_dir=str(output), device="cpu")
    original, _ = interrupt(monkeypatch, before_step=3)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.run_manifest(args)
    launch = (output / "reproduce/launch.json").read_bytes()
    failure = (output / "failure.json").read_bytes()
    monkeypatch.setattr(runner, "train_step", original)
    cli = [
        "--model-kind",
        "modular",
        "--observations-manifest",
        str(path),
        "--run-dir",
        str(output),
        "--modular-resume",
    ]
    assert main(["train", "unified", *cli]) == 0
    monkeypatch.setattr(runner, "evaluate", lambda *a, **k: pytest.fail("unexpected rescoring"))
    script_main(cli)
    assert (output / "reproduce/launch.json").read_bytes() == launch
    assert (output / "failure.json").read_bytes() == failure
    assert registry.REGISTRY_MD.read_text().count(f"#### Closure: {output.name}") == 1
    assert len(list((output / "reproduce").glob("resume-attempt-*.json"))) == 2
    with pytest.raises(SystemExit, match="2"):
        script_main(["--modular-resume"])
    with pytest.raises(ValueError, match="requires --run-dir"):
        runner.run_manifest(SimpleNamespace(observations_manifest=str(path), modular_resume=True))
    args.modular_resume, args.modular_preflight = True, True
    with pytest.raises(ValueError, match="cannot be combined"):
        runner.run_manifest(args)


@pytest.mark.parametrize(
    "source", ["models/presto.py", "cli/train.py", "training/trainer.py", "data/modular_ordinal.py"]
)
def test_promoted_core_and_context_source_changes_reject_resume(tmp_path, monkeypatch, source):
    from presto.training import modular_resume

    path = manifest_fixture(tmp_path, replay=False)
    output = tmp_path / "fit"
    interrupt(monkeypatch, before_step=3)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.fit_index(ObservationIndex(path), output)
    original = modular_resume.file_sha256
    root = Path(modular_resume.__file__).resolve().parents[1]
    monkeypatch.setattr(
        modular_resume,
        "file_sha256",
        lambda path: "0" * 64 if Path(path) == root / source else original(path),
    )
    monkeypatch.setattr(runner, "train_step", lambda *a, **k: pytest.fail("unexpected update"))
    with pytest.raises(ValueError, match="contract changed"):
        runner.fit_index(ObservationIndex(path), output, resume=True)


def test_step_validation_preserves_all_batches_and_exact_resume(tmp_path, monkeypatch):
    path = manifest_fixture(tmp_path)
    document = json.loads(path.read_text())
    document['training']['validation_every_steps'] = 2
    path.write_text(json.dumps(document))
    reference, resumed = tmp_path / 'interval-reference', tmp_path / 'interval-resumed'
    original, trace = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), reference)
    expected_trace = list(trace)
    monkeypatch.setattr(runner, 'train_step', original)
    original, before = interrupt(monkeypatch, before_step=6)
    with pytest.raises(RuntimeError, match='interruption'):
        runner.fit_index(ObservationIndex(path), resumed)
    monkeypatch.setattr(runner, 'train_step', original)
    original, after = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), resumed, resume=True)
    assert before[:4] + after == expected_trace
    assert len(expected_trace) == 8
    for epoch in (expected_trace[:3], expected_trace[3:6]):
        assert {name for batch in epoch for name in batch[0]} == {f'a{i}' for i in range(6)}
    history = json.loads((reference / 'history.json').read_text())
    assert [r['steps'] for r in history] == [2, 4, 6, 8]
    assert [r['epoch'] for r in history] == [1, 2, 2, 3]
    for name in ('last.pt', 'selected.pt'):
        assert_same(
            torch.load(reference / name, weights_only=True),
            torch.load(resumed / name, weights_only=True),
        )
    for name in (
        'history.json',
        'coverage.json',
        'validation-predictions.jsonl',
        'test-predictions.jsonl',
    ):
        assert (reference / name).read_bytes() == (resumed / name).read_bytes()


def test_resume_retries_pending_interval_validation_without_extra_updates(tmp_path, monkeypatch):
    path = manifest_fixture(tmp_path, replay=False)
    document = json.loads(path.read_text())
    document['training']['validation_every_steps'] = 2
    path.write_text(json.dumps(document))
    reference, resumed = tmp_path / 'reference', tmp_path / 'resumed'
    runner.fit_index(ObservationIndex(path), reference)
    original = runner.evaluate

    def fail(*args, **kwargs):
        raise RuntimeError('interrupted interval validation')

    monkeypatch.setattr(runner, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='interval validation'):
        runner.fit_index(ObservationIndex(path), resumed)
    state = torch.load(resumed / 'resume.pt', weights_only=True)['metadata']['resume']['state']
    assert state['steps'] == 2 and state['validation_pending']
    monkeypatch.setattr(runner, 'evaluate', original)
    runner.fit_index(ObservationIndex(path), resumed, resume=True)
    for name in ('last.pt', 'selected.pt'):
        assert_same(
            torch.load(reference / name, weights_only=True),
            torch.load(resumed / name, weights_only=True),
        )
    assert (reference / 'history.json').read_bytes() == (resumed / 'history.json').read_bytes()


def test_changed_sampler_source_invalidates_resume_before_updates(tmp_path, monkeypatch):
    import shutil
    from presto.training import modular_resume

    root = Path(modular_resume.__file__).resolve().parent.parent
    copied = tmp_path / 'frozen-source'
    for relative in modular_resume.SOURCE_FILES:
        destination = copied / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    monkeypatch.setattr(modular_resume, '__file__', str(copied / 'training/modular_resume.py'))
    path = manifest_fixture(tmp_path, replay=False)
    output = tmp_path / 'fit'
    interrupt(monkeypatch, before_step=3)
    with pytest.raises(RuntimeError, match='interruption'):
        runner.fit_index(ObservationIndex(path), output)
    sampler = copied / 'training/modular_sampling.py'
    assert sampler.exists()
    sampler.write_text(sampler.read_text() + '\n# Changed sampler implementation.\n')
    monkeypatch.setattr(runner, 'train_step', lambda *a, **k: pytest.fail('unexpected update'))
    with pytest.raises(ValueError, match='contract changed'):
        runner.fit_index(ObservationIndex(path), output, resume=True)


def test_resume_survives_a_different_host_and_records_the_drift(tmp_path, monkeypatch):
    path = manifest_fixture(tmp_path, replay=False)
    output = tmp_path / "fit"
    original = runner.train_step
    interrupt(monkeypatch, before_step=3)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.fit_index(ObservationIndex(path), output)
    payload = torch.load(output / "resume.pt", weights_only=True)
    host = payload["metadata"]["resume"]["host"]
    host.update(threads=host["threads"] + 7, machine="other-arch", torch_build="other build")
    torch.save(payload, output / "resume.pt")
    monkeypatch.setattr(runner, "train_step", original)
    summary = runner.fit_index(ObservationIndex(path), output, resume=True)
    assert summary["steps"] > 0
    history = [
        json.loads(line) for line in (output / "host-history.jsonl").read_text().split("\n") if line
    ]
    assert history[0]["drift"] == []
    assert {"threads", "machine", "torch_build"} <= set(history[-1]["drift"])
