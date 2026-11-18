"""Evidence calculations for the fixed reference-membership experiment."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-10-21_0512_codex_weirdo-reference-pretraining/code/launch.py"
)
spec = importlib.util.spec_from_file_location("reference_experiment", PATH)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_report_helpers_import_without_optional_baseline_library():
    code = """
import importlib.abc, runpy, sys
class BlockSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise ModuleNotFoundError('scikit-learn deliberately unavailable')
sys.meta_path.insert(0, BlockSklearn())
module = runpy.run_path(sys.argv[1], run_name='dependency_probe')
report, _ = module['binary_report']([0, 1], [0.5, 0.5])
assert report['auprc'] == 0.5
assert not any(name == 'sklearn' or name.startswith('sklearn.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(PATH)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_binary_report_constant_and_perfect_predictions():
    labels = np.asarray([0, 1, 0, 0])
    report, losses = experiment.binary_report(labels, np.full(4, 0.25))
    assert report["auroc"] == 0.5
    assert report["auprc"] == 0.25
    assert report["ece_10_bins"] == 0
    assert report["probability_sd"] == 0
    assert report["brier"] == 0.1875
    assert report["ce"] == pytest.approx(-0.25 * np.log(0.25) - 0.75 * np.log(0.75))
    assert len(losses) == 4
    perfect, _ = experiment.binary_report(labels, labels)
    assert perfect["auroc"] == perfect["auprc"] == 1
    assert perfect["brier"] == perfect["ece_10_bins"] == 0
    assert sum(bin_["n"] for bin_ in perfect["calibration"]) == 4
    with pytest.raises(ValueError, match="finite aligned"):
        experiment.binary_report(labels, [0, np.nan, 0, 0])


def test_component_bootstrap_keeps_members_and_unequal_group_sizes():
    # One component has three measurements; the other has one. The mean is
    # observation-weighted, while whole components are resampled as clusters.
    result = experiment.paired_interval([-1, -1, -1, 1], ["a", "a", "a", "b"])
    assert result["mean_difference"] == -0.5
    assert result["components"] == 2
    assert result["interval_95"] == [-1, 1]
    constant = experiment.paired_interval([-0.2] * 8, ["a"] * 8)
    assert constant["interval_95"] == pytest.approx([-0.2, -0.2])


def test_export_alignment_rejects_wrong_target_and_duplicate(tmp_path):
    row = SimpleNamespace(
        observation_id="one",
        peptide="ACDEFGHI",
        split="validation",
        group_id="g",
        identity_labels={"peptide_source:human": "present"},
    )
    record = {
        "observation_id": "one",
        "identity_task": "peptide_source:human",
        "peptide": row.peptide,
        "group_id": "g",
        "split": "validation",
        "target": 1,
        "probabilities": [0.1, 0.9],
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(record) + "\n")
    assert experiment.network_matrix(path, [row], ["human"])[0, 0] == 0.9
    record["target"] = 0
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="provenance/label mismatch"):
        experiment.network_matrix(path, [row], ["human"])
    path.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        experiment.network_matrix(path, [row], ["human"])
