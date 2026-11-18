"""Independent resampling examples protect scientific reporting semantics."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location(
    "experiment_analysis", Path(__file__).with_name("analyze.py")
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
complete_pairs = analysis.complete_pairs
paired_draws = analysis.paired_draws
protein_groups_available = analysis.protein_groups_available
weighted_auc_draws = analysis.weighted_auc_draws


def test_weighted_auc_matches_explicit_repeated_rows_with_ties():
    y = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([0.2, 0.2, 0.8, 0.8, 0.9, 0.1])
    weights = np.array([[1, 3, 2, 1, 0, 2], [0, 1, 0, 2, 4, 3]])
    expected = [roc_auc_score(np.repeat(y, w), np.repeat(scores, w)) for w in weights]
    np.testing.assert_allclose(weighted_auc_draws(y, scores, weights), expected)


def test_partial_source_pair_ids_do_not_become_paired_cohort():
    a = pd.DataFrame(
        {
            "pair": ["a", "a", "b", "c"],
            "target": [0, 1, 0, 1],
            "prediction_real": [0.1, 0.9, 0.8, 0.7],
            "prediction_other": [0.2, 0.7, 0.6, 0.8],
            "bootstrap_group": ["x", "x", "y", "y"],
        }
    )
    assert not complete_pairs(a)
    actual = paired_draws(a, np.array([[0, 1], [1, 1]]), ["x", "y"])
    assert np.isfinite(actual[:, 0]).all()
    assert np.isnan(actual[:, 1]).all()


def test_complete_pairs_use_half_credit_ties_and_cluster_multiplicity():
    a = pd.DataFrame(
        {
            "pair": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
            "prediction_real": [0.5, 0.5, 0.2, 0.9],
            "prediction_other": [0.8, 0.4, 0.1, 0.8],
            "bootstrap_group": ["x", "x", "y", "y"],
        }
    )
    assert complete_pairs(a)
    actual = paired_draws(a, np.array([[0, 0], [0, 1], [1, 1]]), ["x", "y"])
    np.testing.assert_allclose(actual[:, 1], [0.5, 0.25, 0])


def test_study_components_are_not_protein_clusters():
    assert not protein_groups_available(pd.DataFrame({"group": ["study-component:PXD025716"]}))
    assert not protein_groups_available(pd.DataFrame({"group": [None]}))
    assert protein_groups_available(
        pd.DataFrame({"group": ["ENSP00000292314.2", "ENSP00000464258.2"]})
    )
