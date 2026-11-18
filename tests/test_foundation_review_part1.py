"""Regression tests for foundation review items 1-4."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from presto.cli.main import create_parser
from presto.cli.observations import cmd_evaluate_observations, cmd_predict_observations
from presto.data.loaders import KineticsRecord
from presto.data.modular import AA, MHCInput, Observation
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    affinity_observations,
    write_observations,
)
from presto.scripts import experiment_registry as registry
from presto.training import modular_resume
from presto.training.modular_evaluation import summarize_strata


def test_registry_allowlists_env_vars_and_strips_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("PRESTO_MODAL_GPU", "H100!")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-secret-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-secret-value")
    monkeypatch.setenv("PRESTO_SECRET_KEY", "never-leak")
    monkeypatch.setenv("CUDA_UNLISTED_INTERNAL", "ignored")

    captured = registry._capture_launch_environment()
    assert captured["PRESTO_MODAL_GPU"] == "H100!"
    assert captured["MODAL_ENVIRONMENT"] == "main"
    assert "MODAL_TOKEN_ID" not in captured
    assert "MODAL_TOKEN_SECRET" not in captured
    assert "PRESTO_SECRET_KEY" not in captured
    assert "CUDA_UNLISTED_INTERNAL" not in captured

    repro = registry.write_reproducibility_bundle(
        out_dir=tmp_path / "exp",
        source_script="scripts/train_iedb.py",
        argv=["scripts/train_iedb.py", "--epochs", "1"],
        environment={
            "PRESTO_MODAL_GPU": "H100!",
            "AWS_SECRET_ACCESS_KEY": "redacted",
            "HF_TOKEN": "redacted",
        },
        git_state={"commit": "abc1234", "branch": "main", "dirty": False},
    )
    payload = json.loads((repro / "launch.json").read_text(encoding="utf-8"))
    assert payload["environment"] == {"PRESTO_MODAL_GPU": "H100!"}
    launch_sh = (repro / "launch.sh").read_text(encoding="utf-8")
    assert "AWS_SECRET_ACCESS_KEY" not in launch_sh
    assert "HF_TOKEN" not in launch_sh
    assert "#   dirty: no" in launch_sh


def test_initialize_experiment_dir_captures_git_state_pre_creation_and_serializes_log(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "experiments" / "experiment_log.md"
    monkeypatch.setattr(registry, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    monkeypatch.setattr(registry, "REGISTRY_MD", log_path)

    target = tmp_path / "experiments" / "2026-11-18_1200_codex_clean-check"
    existed_during_capture = []

    def fake_capture():
        existed_during_capture.append(target.exists())
        return {"commit": "deadbeef", "branch": "feature", "dirty": False}

    monkeypatch.setattr(registry, "capture_git_state", fake_capture)
    monkeypatch.setattr(registry, "_capture_git_state", fake_capture)

    out = registry.initialize_experiment_dir(
        out_dir=str(target),
        slug="clean-check",
        title="Clean check",
        source_script="cli/train.py",
        agent_label="codex",
        metadata={"dataset_contract": {"name": "fixture"}, "training": {}, "tested": ["a"]},
    )
    assert out == target
    assert existed_during_capture == [False]
    meta = json.loads((target / "reproduce" / "launch.json").read_text(encoding="utf-8"))
    assert meta["git"] == {"commit": "deadbeef", "branch": "feature", "dirty": False}

    def append_one(idx: int) -> None:
        registry._append_markdown_stub(
            {
                "id": f"2026-11-18_1200_codex_concurrent-{idx}",
                "agent": "codex",
                "experiment_dir": f"experiments/2026-11-18_1200_codex_concurrent-{idx}",
                "source_script": "cli/train.py",
                "status": "launched",
            }
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append_one, range(16)))

    text = log_path.read_text(encoding="utf-8")
    for idx in range(16):
        assert f"### 2026-11-18_1200_codex_concurrent-{idx}" in text


def test_portable_locking_and_atomic_unique_temp_writes(tmp_path, monkeypatch):
    target_bytes = tmp_path / "artifact.bin"
    target_json = tmp_path / "artifact.json"
    modular_resume.atomic_bytes(target_bytes, b"initial")
    modular_resume.atomic_json(target_json, {"v": 0})

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(
            pool.map(
                lambda i: modular_resume.atomic_bytes(target_bytes, f"payload-{i}".encode()),
                range(12),
            )
        )
        list(
            pool.map(
                lambda i: modular_resume.atomic_json(target_json, {"v": i}),
                range(12),
            )
        )

    assert target_bytes.read_bytes().startswith(b"payload-")
    assert "v" in json.loads(target_json.read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))

    with pytest.raises(ValueError, match="Out of range float values"):
        modular_resume.atomic_json(target_json, {"bad": float("nan")})
    assert not list(tmp_path.glob(".*.tmp"))

    # Exercise Windows msvcrt fallback when fcntl is unavailable.
    events = []
    locked = {"held": False}

    class FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(fd, mode, nbytes):
            if mode == FakeMsvcrt.LK_NBLCK:
                if locked["held"]:
                    raise OSError("already locked")
                locked["held"] = True
                events.append("lock")
            elif mode == FakeMsvcrt.LK_UNLCK:
                locked["held"] = False
                events.append("unlock")

    monkeypatch.setattr(modular_resume, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)
    run_dir = tmp_path / "run-lock-dir"
    with modular_resume.run_lock(run_dir):
        with pytest.raises(ValueError, match="another trainer is active"):
            with modular_resume.run_lock(run_dir):
                pass
    assert events == ["lock", "unlock"]


def test_cli_and_training_package_import_isolation():
    code = """
import json
import sys
import presto.training as tr

heavy = (
    "presto.training.trainer",
    "presto.training.modular",
    "presto.training.modular_run",
    "presto.training.modular_resume",
    "presto.inference.modular",
    "presto.inference.predictor",
)
assert all(mod not in sys.modules for mod in heavy), [m for m in heavy if m in sys.modules]
assert "Trainer" in dir(tr) and "run_manifest" in dir(tr)

from presto.cli.main import main
rc = main(["weights", "list", "--json"])
assert rc == 0
loaded = [m for m in heavy if m in sys.modules]
assert not loaded, f"unexpected training/inference imports during weights list: {loaded}"
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_cuda_rng_saves_and_restores_onto_active_resume_device(monkeypatch):
    get_calls = []
    set_calls = []
    fake_state_1 = torch.arange(16, dtype=torch.uint8)
    fake_state_0 = torch.arange(16, 32, dtype=torch.uint8)

    def fake_get_rng_state(device=None):
        get_calls.append(torch.device(device))
        return fake_state_1.clone()

    def fake_set_rng_state(new_state, device=None):
        set_calls.append((new_state.clone(), torch.device(device)))

    monkeypatch.setattr(torch.cuda, "get_rng_state", fake_get_rng_state)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: pytest.fail("get_rng_state_all should not be used"),
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state", fake_set_rng_state)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda *a, **k: pytest.fail("set_rng_state_all should not be used"),
    )

    sampler = random.Random(7)
    replay = random.Random(11)
    saved = modular_resume.rng_state(sampler, replay, "cuda:1")
    assert get_calls == [torch.device("cuda:1")]
    assert torch.equal(saved["cuda"], fake_state_1)

    modular_resume.restore_rng(saved, sampler, replay, device="cuda:2")
    assert len(set_calls) == 1
    assert torch.equal(set_calls[0][0], fake_state_1)
    assert set_calls[0][1] == torch.device("cuda:2")

    # Legacy list checkpoint restores matching index or falls back to slot 0, and skips on CPU.
    legacy = dict(saved, cuda=[fake_state_0, fake_state_1])
    modular_resume.restore_rng(legacy, sampler, replay, device="cuda:0")
    assert torch.equal(set_calls[-1][0], fake_state_0)
    assert set_calls[-1][1] == torch.device("cuda:0")

    before_len = len(set_calls)
    modular_resume.restore_rng(saved, sampler, replay, device="cpu")
    assert len(set_calls) == before_len


def test_observation_index_enforces_declared_split_and_preserves_mixed_sources(tmp_path):
    mhc = (MHCInput(AA, AA[::-1], "I"),)
    train_row = Observation(
        "obs-train",
        "fixture",
        "ic50",
        "ACDEFGHIK",
        mhc,
        value=50.0,
        unit="nM",
        split="train",
        group_id="g-train",
        provenance={"original_measurement_id": "m-train"},
    )
    val_row = Observation(
        "obs-val",
        "fixture",
        "ic50",
        "LMNPQRSTV",
        mhc,
        value=500.0,
        unit="nM",
        split="validation",
        group_id="g-val",
        provenance={"original_measurement_id": "m-val"},
    )
    mixed = write_observations(tmp_path / "mixed.jsonl", [train_row, val_row])
    base_source = {
        "path": "mixed.jsonl",
        "sha256": mixed["sha256"],
        "source": "fixture",
        "population": "synthetic",
        "label_policy": "synthetic",
        "split_policy": "synthetic",
    }

    # Mixed manifest without source-level split continues to work.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema": MANIFEST_VERSION, "sources": [base_source]}),
        encoding="utf-8",
    )
    index = ObservationIndex(manifest_path)
    assert len(index) == 2

    # Declared split disagrees with validation row inside mixed.jsonl -> rejected.
    manifest_path.write_text(
        json.dumps({"schema": MANIFEST_VERSION, "sources": [{**base_source, "split": "train"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="row split disagrees"):
        ObservationIndex(manifest_path)

    # Invalid declared split string -> rejected before row scan.
    manifest_path.write_text(
        json.dumps({"schema": MANIFEST_VERSION, "sources": [{**base_source, "split": "predict"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported declared source split"):
        ObservationIndex(manifest_path)


def test_affinity_observations_emits_both_kon_and_koff():
    mhc = MHCInput(AA, AA[::-1], "I")
    record = KineticsRecord(
        peptide="SIINFEKL",
        mhc_allele="HLA-A*02:01",
        kon=1.0e5,
        kon_qualifier=0,
        koff=1.0e-3,
        koff_qualifier=1,
        source="iedb",
        evidence_row_id="kin-42",
    )
    rows = affinity_observations(record, mhc, split="train", group_id="grp-1")
    by_endpoint = {row.endpoint: row for row in rows}
    assert set(by_endpoint) == {"kon", "koff"}
    assert by_endpoint["kon"].unit == "1/(M*s)"
    assert by_endpoint["kon"].value == pytest.approx(1.0e5)
    assert by_endpoint["kon"].qualifier == "="
    assert by_endpoint["koff"].unit == "1/s"
    assert by_endpoint["koff"].value == pytest.approx(1.0e-3)
    assert by_endpoint["koff"].qualifier == ">"


def test_summarize_strata_normalizes_missing_mhc_classes():
    rows = [
        {
            "endpoint": "elution_observation",
            "target": 1.0,
            "prediction": 0.8,
            "loss": 0.2,
            "mhc_classes": [None],
            "qualifier": "=",
            "observation": {"provenance": {"study_id": "pmid:1"}},
        },
        {
            "endpoint": "elution_observation",
            "target": 0.0,
            "prediction": -0.5,
            "loss": 0.3,
            "mhc_classes": ["I", None],
            "qualifier": "=",
            "observation": {"provenance": {"study_id": "pmid:1"}},
        },
    ]
    strata = summarize_strata(rows)
    classes = {item["mhc_class"] for item in strata}
    assert classes == {"unknown", "I;unknown"}


def test_streaming_predict_batches_and_cleans_up_partial_outputs(tmp_path, monkeypatch):
    mhc = (MHCInput(AA, AA[::-1], "I"),)
    queries = [
        Observation(
            f"q-{i}",
            "fixture",
            "ic50",
            AA[:9],
            mhc,
            value=None,
            split="predict",
        ).to_dict()
        for i in range(5)
    ]
    inputs_path = tmp_path / "queries.jsonl"
    inputs_path.write_text("".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8")

    batch_sizes = []

    class FakePredictor:
        def predict(self, batch, unsupported_apc_policy="error"):
            batch_sizes.append(len(batch))
            return [
                {"observation_id": row.observation_id, "prediction": float(idx)}
                for idx, row in enumerate(batch)
            ]

    monkeypatch.setattr(
        "presto.cli.observations._checkpoint",
        lambda args: (FakePredictor(), "a" * 64),
    )

    out_path = tmp_path / "predictions.json"
    parser = create_parser()
    args = parser.parse_args(
        [
            "predict",
            "observations",
            "--checkpoint",
            "dummy.pt",
            "--inputs",
            str(inputs_path),
            "--output",
            str(out_path),
            "--batch-size",
            "2",
        ]
    )
    assert cmd_predict_observations(args) == 0
    assert batch_sizes == [2, 2, 1]
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(parsed["predictions"]) == 5
    assert parsed["checkpoint_sha256"] == "a" * 64

    # Failure mid-stream unlinks the newly created output file.
    bad_queries = list(queries)
    bad_queries[-1] = dict(bad_queries[-1], split="train", value=10.0, unit="nM")
    bad_inputs = tmp_path / "bad_queries.jsonl"
    bad_inputs.write_text("".join(json.dumps(q) + "\n" for q in bad_queries), encoding="utf-8")
    bad_out = tmp_path / "bad_predictions.json"
    bad_args = parser.parse_args(
        [
            "predict",
            "observations",
            "--checkpoint",
            "dummy.pt",
            "--inputs",
            str(bad_inputs),
            "--output",
            str(bad_out),
            "--batch-size",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="label-free predict observations"):
        cmd_predict_observations(bad_args)
    assert not bad_out.exists()


def test_evaluate_observations_removes_partial_output_dir_on_write_failure(tmp_path, monkeypatch):
    mhc = (MHCInput(AA, AA[::-1], "I"),)
    val_row = Observation(
        "obs-val-1",
        "fixture",
        "ic50",
        "ACDEFGHIK",
        mhc,
        value=100.0,
        unit="nM",
        split="validation",
        group_id="g-1",
        provenance={"original_measurement_id": "m-1", "study_id": "pmid:1"},
    )
    written = write_observations(tmp_path / "val.jsonl", [val_row])
    manifest = {
        "schema": MANIFEST_VERSION,
        "sources": [
            {
                "path": "val.jsonl",
                "sha256": written["sha256"],
                "source": "fixture",
                "split": "validation",
                "population": "synthetic",
                "label_policy": "synthetic",
                "split_policy": "synthetic",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fake_predictor = SimpleNamespace(model=SimpleNamespace(config=None), scales={})
    monkeypatch.setattr(
        "presto.cli.observations._checkpoint",
        lambda args: (fake_predictor, "b" * 64),
    )
    monkeypatch.setattr(
        "presto.training.modular_evaluation.evaluate",
        lambda *a, **k: (
            {"ic50": {"n": 1, "loss": 0.1}},
            [
                {
                    "observation_id": "obs-val-1",
                    "endpoint": "ic50",
                    "target": 2.0,
                    "prediction": float("nan"),  # triggers ValueError under allow_nan=False
                    "loss": 0.1,
                    "mhc_classes": ["I"],
                    "qualifier": "=",
                    "observation": val_row.to_dict(),
                }
            ],
        ),
    )

    eval_dir = tmp_path / "eval-out"
    parser = create_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "observations",
            "--checkpoint",
            "dummy.pt",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(eval_dir),
        ]
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        cmd_evaluate_observations(args)
    assert not eval_dir.exists()
