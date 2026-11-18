"""The installed workflow uses one shared model and never scans unselected TEST files."""

from dataclasses import replace
import json

import pytest
import torch

from presto.cli.main import create_parser
from presto.cli.observations import (
    _evaluation_manifest,
    cmd_evaluate_observations,
    cmd_predict_observations,
)
from presto.data.modular import MHCInput, Observation
from presto.data.modular_sources import MANIFEST_VERSION, ObservationIndex, write_observations
from presto.evaluation.presentation import paired_metrics
from presto.training.modular_run import fit_index


def test_unselected_test_path_is_never_opened(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": "validation.jsonl", "split": "validation"},
                    {"path": "SEALED-DOES-NOT-EXIST.jsonl", "split": "test"},
                ]
            }
        )
    )
    result = _evaluation_manifest(path, "validation")
    assert [source["path"] for source in result["sources"]] == [str(tmp_path / "validation.jsonl")]
    path.write_text(json.dumps({"sources": [{"path": "mixed.jsonl"}]}))
    with pytest.raises(ValueError, match="explicit split"):
        _evaluation_manifest(path, "validation")


def test_pair_metrics_require_complete_pairs_and_count_ties():
    def record(cls, pair, target, score):
        return {
            "endpoint": "elution_observation",
            "mhc_classes": [cls],
            "target": target,
            "prediction": score,
            "observation": {"provenance": {"pair_id": pair}},
        }

    rows = [
        record("I", "a", 1, 2),
        record("I", "a", 0, 1),
        record("II", "b", 1, 0),
        record("II", "b", 0, 0),
    ]
    assert paired_metrics(rows) == {
        "I": {"pairs": 1, "win_rate": 1.0},
        "II": {"pairs": 1, "win_rate": 0.5},
    }
    with pytest.raises(ValueError, match="incomplete"):
        paired_metrics(rows[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        paired_metrics(rows + [rows[0]])


def test_train_reload_evaluate_and_predict_both_classes(tmp_path):
    torch.set_num_threads(2)
    peptides = [
        "ACDEFGHIK",
        "LMNPQRSTV",
        "WYACDEFGHIKLM",
        "NPQRSTVWYACD",
        "CDEFGHIKL",
        "MNPQRSTVW",
        "YACDEFGHIKLMN",
        "PQRSTVWYACDE",
    ]
    rows = [
        Observation(
            f"fixture-{i}",
            "software-workflow-fixture",
            "ic50",
            peptide,
            (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I" if i % 4 < 2 else "II"),),
            value=10.0 if i % 2 else 1000.0,
            unit="nM",
            split="train" if i < 4 else "validation",
            group_id=f"group-{i}",
            provenance={"original_measurement_id": f"measurement-{i}"},
        )
        for i, peptide in enumerate(peptides)
    ]
    sources = []
    for split in ("train", "validation"):
        receipt = write_observations(
            tmp_path / f"{split}.jsonl", [row for row in rows if row.split == split]
        )
        sources.append(
            {
                "path": f"{split}.jsonl",
                "sha256": receipt["sha256"],
                "source": "software-workflow-fixture",
                "split": split,
                "population": "artificial software check",
                "label_policy": "artificial numeric targets; no biological claims",
                "split_policy": "distinct artificial peptides and groups",
            }
        )
    manifest = {
        "schema": MANIFEST_VERSION,
        "sources": sources,
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 32,
            "endpoints": ["ic50"],
            "identity_tasks": [],
            "dropout": 0,
        },
        "training": {
            "max_epochs": 1,
            "max_steps": 1,
            "batch_size": 4,
            "test_policy": "no_test_training_diagnostic",
            "no_test_reason": "Artificial integration fixture; no biological TEST outcomes",
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    fit_index(ObservationIndex(path), tmp_path / "fit")
    checkpoint = str(tmp_path / "fit/selected.pt")
    # A sealed, nonexistent payload must not be resolved or hashed by evaluation.
    manifest["sources"].append({"path": "SEALED-NOT-PRESENT.jsonl", "split": "test"})
    path.write_text(json.dumps(manifest))
    parser = create_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "observations",
            "--checkpoint",
            checkpoint,
            "--manifest",
            str(path),
            "--output-dir",
            str(tmp_path / "eval"),
        ]
    )
    assert cmd_evaluate_observations(args) == 0
    receipt = json.loads((tmp_path / "eval/receipt.json").read_text())
    assert receipt["rows"] == 4 and receipt["test_accessed"] is False
    requests = tmp_path / "queries.jsonl"
    requests.write_text(
        "".join(
            json.dumps(replace(row, value=None, split="predict").to_dict()) + "\n"
            for row in rows[-4:]
        )
    )
    args = parser.parse_args(
        [
            "predict",
            "observations",
            "--checkpoint",
            checkpoint,
            "--inputs",
            str(requests),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert cmd_predict_observations(args) == 0
    assert len(json.loads((tmp_path / "out.json").read_text())["predictions"]) == 4
    args.checkpoint_sha256 = "0" * 64
    with pytest.raises(ValueError, match="checkpoint SHA"):
        cmd_predict_observations(args)
