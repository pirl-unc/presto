"""Train-only fixed baselines and source-verified IC50 evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from presto.training.holdout_eval import binding_threshold_metrics, regression_metrics

ENDPOINT = "half maximal inhibitory concentration (IC50)"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
FEATURES = ["length", *["fraction_" + aa for aa in AMINO_ACIDS], "class_I", "class_II"]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def features(frame):
    classes = frame.native_mhc_class.astype(str)
    if not set(classes) <= {"I", "II"}:
        raise ValueError("Unexpected IC50 MHC class")
    peptides = frame.peptide.astype(str)
    if any(not p or set(p) - set(AMINO_ACIDS) for p in peptides):
        raise ValueError("Invalid peptide for fixed baseline features")
    return np.asarray(
        [
            [
                len(p),
                *[p.count(aa) / len(p) for aa in AMINO_ACIDS],
                float(c == "I"),
                float(c == "II"),
            ]
            for p, c in zip(peptides, classes)
        ],
        dtype=float,
    )


def fit_baselines(train):
    if set(train.source_partition) != {"train"}:
        raise ValueError("Baseline fitting requires training observations only")
    exact = train[train.qualifier == 0]
    x, y = features(exact), exact.target_log10_nM.to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite training target")
    if len(y) < 50:
        raise ValueError("Insufficient exact training evidence for fixed baselines")
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    standardized = (x - mean) / scale
    intercept = float(y.mean())
    coefficient = np.linalg.solve(
        standardized.T @ standardized + np.eye(x.shape[1]), standardized.T @ (y - intercept)
    )
    return {
        "feature_names": FEATURES,
        "alpha": 1.0,
        "n_exact_training_observations": len(y),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficient.tolist(),
        "intercept": intercept,
        "fit_scope": "Exact train only; no held-out scaling, calibration or tuning.",
    }


def predict_baselines(frame, state):
    x = (features(frame) - np.asarray(state["feature_mean"])) / np.asarray(state["feature_scale"])
    return {
        "train_mean": np.full(len(frame), state["intercept"]),
        "length_composition_class_ridge": state["intercept"]
        + x @ np.asarray(state["coefficients"]),
    }


def metrics(target, prediction, qualifier):
    target, prediction, qualifier = map(np.asarray, (target, prediction, qualifier))
    if not len(target) or not (target.shape == prediction.shape == qualifier.shape):
        raise ValueError("Evaluation requires nonempty aligned vectors")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite target/prediction")
    if not set(qualifier.tolist()) <= {-1, 0, 1}:
        raise ValueError("Invalid censor code")
    exact, censored = qualifier == 0, qualifier != 0
    result = regression_metrics(target[exact], prediction[exact]) if exact.any() else {}
    result.pop("n", None)
    for name in ("rmse", "mae", "pearson", "spearman"):
        result.setdefault(name, None)
    error = prediction - target
    residual = np.where(
        qualifier < 0, np.maximum(error, 0), np.where(qualifier > 0, np.minimum(error, 0), error)
    )
    result.update(
        n_observations=len(target),
        n_exact=int(exact.sum()),
        n_left_censored=int((qualifier < 0).sum()),
        n_right_censored=int((qualifier > 0).sum()),
        censor_aware_mse=float(np.mean(residual**2)),
        censor_satisfaction_fraction=float(np.mean(np.abs(residual[censored]) <= 1e-6))
        if censored.any()
        else None,
    )
    binary = binding_threshold_metrics(target, prediction, qualifier, threshold_nm=500)
    cutoff = np.log10(500)
    eligible = (
        exact | ((qualifier < 0) & (target <= cutoff)) | ((qualifier > 0) & (target > cutoff))
    )
    labels, predicted = target[eligible] <= cutoff, prediction[eligible] <= cutoff
    binary.pop("n", None)
    binary.update(
        n_eligible_observations=int(eligible.sum()),
        n_ambiguous_bounds=int((~eligible).sum()),
        n_positive=int(labels.sum()),
        n_negative=int((~labels).sum()),
        threshold_nM=500,
        ambiguity_policy="Collapsed strict/inclusive qualifiers use conservative bounds.",
    )
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc", "auprc"):
        binary.setdefault(name, None)
    if not predicted.any():
        binary["precision"] = None
    if not labels.any():
        binary["recall"] = None
    if not labels.any() or labels.all():
        binary["balanced_accuracy"] = None
        binary["auroc"] = None
        binary["auprc"] = None
    if not (predicted.sum() + labels.sum()):
        binary["f1"] = None
    result["binding_500nM"] = binary
    result["undefined_metrics"] = [k for k, v in result.items() if v is None]
    binary["undefined_metrics"] = [k for k, v in binary.items() if v is None]
    return result


def paired_rmse_bootstrap(frame, predictions, n_resamples=1000, target_column="target_log10_nM"):
    exact = frame.qualifier.to_numpy(int) == 0
    target = frame[target_column].to_numpy(float)[exact]
    peptides = frame.peptide.to_numpy()[exact]
    values = {name: np.asarray(value)[exact] for name, value in predictions.items()}
    groups = [np.flatnonzero(peptides == peptide) for peptide in sorted(set(peptides))]
    rng = np.random.default_rng(4242)
    differences = {name: [] for name in values if name != "presto"}
    for _ in range(n_resamples):
        indexes = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), size=len(groups))]
        )
        scores = {
            name: np.sqrt(np.mean((value[indexes] - target[indexes]) ** 2))
            for name, value in values.items()
        }
        for name in differences:
            differences[name].append(float(scores["presto"] - scores[name]))
    return {
        "seed": 4242,
        "n_resamples": n_resamples,
        "n_exact_observations": len(target),
        "n_peptides": len(groups),
        "sampling_unit": "peptide; all same-peptide observations travel together",
        "independence_limit": "Unknown biological identity and shared studies remain unresolved.",
        "paired_presto_minus_baseline_rmse_95_intervals": {
            name: np.quantile(values, [0.025, 0.975]).tolist()
            for name, values in differences.items()
        },
    }


def expected_rows(membership_path, contexts_path):
    membership = pd.read_csv(membership_path, sep="\t", keep_default_na=False, dtype=str)
    contexts = pd.read_csv(contexts_path, sep="\t", keep_default_na=False, dtype=str)
    if not membership.sample_id.is_unique or not contexts.evidence_row_id.is_unique:
        raise ValueError("Duplicate frozen source/model identity")
    frame = membership[membership.source_endpoint == ENDPOINT].copy()
    if len(frame) != 2614 or set(frame.sample_source) - {"iedb", "cedar"}:
        raise ValueError("Direct IC50 cohort differs from the frozen preflight")
    context = contexts.rename(
        columns={c: "native_" + c for c in contexts if c != "evidence_row_id"}
    )
    frame = frame.merge(
        context, on="evidence_row_id", how="left", validate="one_to_one", indicator=True
    )
    if (
        not (frame._merge == "both").all()
        or frame.peptide.tolist() != frame.native_peptide.tolist()
    ):
        raise ValueError("Native context identity/peptide mismatch")
    frame = frame.drop(columns="_merge")
    if set(frame.source_units) != {"nM"}:
        raise ValueError("IC50 physical units changed")
    raw = pd.to_numeric(frame.source_value).to_numpy(float)
    if not np.isfinite(raw).all() or not (raw > 0).all():
        raise ValueError("Invalid IC50 source values")
    frame["raw_target_log10_nM"] = np.log10(raw)
    import torch
    from presto.training.target_policy import AFFINITY_TARGET_POLICY

    frame["effective_target_log10_nM"] = AFFINITY_TARGET_POLICY(
        torch.tensor(raw, dtype=torch.float32)
    ).numpy()
    frame["target_log10_nM"] = frame.effective_target_log10_nM
    frame["physical_value_clipped"] = (raw < 0.001) | (raw > 50000.0)
    q = {"": 0, "=": 0, "<": -1, "<=": -1, "≤": -1, ">": 1, ">=": 1, "≥": 1}
    frame["qualifier"] = frame.source_inequality.map(q)
    if frame.qualifier.isna().any():
        raise ValueError("Unknown raw censor operator")
    frame["qualifier"] = frame.qualifier.astype(int)
    if frame.groupby("source_partition").size().to_dict() != {
        "train": 2074,
        "val": 265,
        "test": 275,
    }:
        raise ValueError("Direct IC50 partition membership changed")
    return frame


def require_effective_target_contract(frame, declaration):
    """Verify the new explicit policy and all raw/effective labels before model work."""
    from presto.training.output_contract import (
        OutputConfiguration,
        build_output_contract,
    )
    from presto.training.target_policy import affinity_target_policies

    contract = build_output_contract(OutputConfiguration(**declaration["model_contract"]))
    if declaration["source_contract"].get("affinity_target_policies") != affinity_target_policies(
        contract
    ):
        raise ValueError("Prospective manifest differs from the executable affinity target policy")
    raw = frame.source_value.to_numpy(float)
    expected = np.log10(np.clip(raw.astype(np.float32), 0.001, 50000.0))
    if not np.allclose(frame.target_log10_nM, expected, atol=1e-6, rtol=0):
        raise ValueError("Effective targets differ from declared float32 clipping policy")
    if not np.allclose(frame.raw_target_log10_nM, np.log10(raw), atol=1e-12, rtol=0):
        raise ValueError("Raw measurement target changed")
    q = {"": 0, "=": 0, "<": -1, "<=": -1, "≤": -1, ">": 1, ">=": 1, "≥": 1}
    if not np.array_equal(frame.qualifier, frame.source_inequality.map(q)):
        raise ValueError("Original censor direction changed")


def verified_predictions(path, expected):
    rows = pd.read_csv(path, keep_default_na=False, dtype=str)
    rows = rows[rows.task == "binding_ic50"].copy()
    if not rows.sample_id.is_unique or set(rows.sample_id) != set(expected.sample_id):
        raise ValueError("IC50 predictions differ from exact frozen membership")
    rows = rows.set_index("sample_id").loc[expected.sample_id]
    for field in ("evidence_row_id", "peptide", "source_partition"):
        if rows[field].tolist() != expected[field].tolist():
            raise ValueError("Prediction source lineage changed: " + field)
    if not np.allclose(
        rows.y_true.to_numpy(float), expected.target_log10_nM.to_numpy(float), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Prediction target differs from declared effective log10(nM) target")
    if not np.array_equal(rows.qualifier.to_numpy(int), expected.qualifier.to_numpy(int)):
        raise ValueError("Prediction censor direction changed")
    prediction = rows.y_pred.to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite IC50 prediction")
    return prediction


TARGET_SPACES = {"effective": "effective_target_log10_nM", "raw": "raw_target_log10_nM"}


def report_spaces(frame, predictions):
    return {
        space: {
            name: metrics(frame[column].to_numpy(float), pred, frame.qualifier.to_numpy(int))
            for name, pred in predictions.items()
        }
        for space, column in TARGET_SPACES.items()
    }


def inspection_rows(frame, predictions):
    frame = frame.copy()
    q = frame.qualifier.to_numpy(int)
    for name, pred in predictions.items():
        frame[name + "_prediction_log10_nM"] = pred
        for space, column in TARGET_SPACES.items():
            target = frame[column].to_numpy(float)
            frame[name + "_residual_" + space + "_log10_nM"] = pred - target
            frame[name + "_censor_satisfied_" + space] = np.where(
                q < 0, pred <= target, np.where(q > 0, pred >= target, None)
            )
    return frame


def compare(work, membership_path, contexts_path):
    frame = expected_rows(membership_path, contexts_path)
    train = frame[frame.source_partition == "train"].copy()
    state = fit_baselines(train)
    state["target_policy"] = "float32 log10(clip(nM, 0.001, 50000)); original qualifier retained"
    write(work / "baseline-state.json", state)
    initial = verified_predictions(work / "train_initial_predictions.csv", train)
    trained = verified_predictions(work / "train_selected_predictions.csv", train)
    train_predictions = {"initial": initial, "selected": trained, **predict_baselines(train, state)}
    training = report_spaces(train, train_predictions)
    inspection_rows(train, train_predictions).to_csv(
        work / "training-known-examples.tsv", sep="\t", index=False
    )
    comparisons, intervals = {}, {}
    for split in ("val", "test"):
        expected = frame[frame.source_partition == split].copy()
        predictions = {
            "presto": verified_predictions(work / f"{split}_predictions.csv", expected),
            **predict_baselines(expected, state),
        }
        comparisons[split] = report_spaces(expected, predictions)
        intervals[split] = {
            space: paired_rmse_bootstrap(expected, predictions, target_column=column)
            for space, column in TARGET_SPACES.items()
        }
        inspection_rows(expected, predictions).to_csv(
            work / f"{split}-known-examples.tsv", sep="\t", index=False
        )
    updates = json.loads((work / "output_updates.json").read_text())
    ic50 = next(
        row
        for row in updates["outputs"]
        if row["endpoint"] == "assays.IC50_nM" and row["column"] == ""
    )
    invalid = sum(row["nonfinite_output_gradient_elements"] for row in updates["outputs"])
    parameter_rows = list(updates["additional_parameter_rows"])
    for row in updates["outputs"]:
        parameter_rows.extend(row["parameter_rows"])
    invalid += sum(
        p["initial_nonfinite_elements"]
        + p["nonfinite_gradient_elements"]
        + p["nonfinite_update_elements"]
        for p in parameter_rows
    )
    finite_optimization = (
        updates["optimizer_steps"] == 200
        and updates["batches_aborted"] == 0
        and invalid == 0
        and ic50["output_gradient_abs_sum"] > 0
    )
    validation = comparisons["val"]["effective"]["presto"]
    positive_rho = validation["spearman"] is not None and validation["spearman"] > 0
    ci = intervals["val"]["effective"]["paired_presto_minus_baseline_rmse_95_intervals"]
    decision = {
        "finite_optimization": finite_optimization,
        "n_nonfinite_tracked_elements": invalid,
        "training_rmse_improved": training["effective"]["selected"]["rmse"]
        < training["effective"]["initial"]["rmse"],
        "validation_quality_screen_passed": positive_rho and ci["train_mean"][1] < 0,
        "preferred_over_ridge": positive_rho and ci["length_composition_class_ridge"][1] < 0,
        "primary_target_space": "effective: float32 log10(clip(nM, 0.001, 50000))",
        "raw_target_diagnostics": "Reported separately; original measurements/bounds remain unchanged.",
        "selection_rule": "Canonical complete-validation aggregate loss selects checkpoint; test is descriptive only.",
        "biological_independence_established": False,
        "all_output_readiness_established": False,
    }
    decision["bounded_fitting_check_passed"] = (
        finite_optimization and decision["training_rmse_improved"]
    )
    write(work / "training-diagnostic.json", training)
    write(work / "comparison.json", comparisons)
    write(work / "uncertainty.json", intervals)
    write(work / "decision.json", decision)
    return {
        "comparison": comparisons,
        "training_diagnostic": training,
        "decision": decision,
        "optimizer_steps": updates["optimizer_steps"],
        "ic50_update_coverage": ic50,
    }
