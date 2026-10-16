"""Verify the frozen fit, recompute persisted predictions, and create inspection artifacts."""

import hashlib
import importlib.metadata
import io
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
from datacache import materialize
from presto.training.evaluation_ledger import reconcile_prediction_artifact

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = Path(__file__).resolve().parents[1]


def pin(path):
    p = Path(path)
    return {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "size_bytes": p.stat().st_size,
    }


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def metric_check(frame, prefix, expected, space):
    y = frame[space + "_target_log10_nM"].to_numpy(float)
    p = frame[prefix + "_prediction_log10_nM"].to_numpy(float)
    q = frame.qualifier.to_numpy(int)
    exact = q == 0
    computed = {
        "rmse": float(np.sqrt(np.mean((y[exact] - p[exact]) ** 2))),
        "mae": float(np.mean(np.abs(y[exact] - p[exact]))),
    }
    for key, left, right in (
        ("pearson", y[exact], p[exact]),
        (
            "spearman",
            pd.Series(y[exact]).rank().to_numpy(),
            pd.Series(p[exact]).rank().to_numpy(),
        ),
    ):
        computed[key] = (
            float(np.corrcoef(left, right)[0, 1])
            if np.ptp(left) and np.ptp(right)
            else None
        )
    error = p - y
    hinge = np.where(
        q < 0, np.maximum(error, 0), np.where(q > 0, np.minimum(error, 0), error)
    )
    computed["censor_aware_mse"] = float(np.mean(hinge**2))
    computed["censor_satisfaction_fraction"] = float(
        np.mean(abs(hinge[q != 0]) <= 1e-6)
    )
    for key, value in computed.items():
        if value is None:
            if expected[key] is not None:
                # #116: canonical mean-centering can leave a tiny residual for
                # an exactly constant vector. Retain original reports, but
                # independent correlation is undefined, not evidence of fit.
                assert key == "pearson" and not np.ptp(p[exact])
                assert abs(expected[key]) < 1e-14
        else:
            assert np.isclose(value, expected[key], atol=1e-10, rtol=1e-9), (
                key,
                value,
                expected[key],
            )
    cutoff = np.log10(500)
    eligible = exact | ((q < 0) & (y <= cutoff)) | ((q > 0) & (y > cutoff))
    labels, scores, predicted = (
        y[eligible] <= cutoff,
        -p[eligible],
        p[eligible] <= cutoff,
    )
    tp, tn = int((labels & predicted).sum()), int((~labels & ~predicted).sum())
    fp, fn = int((~labels & predicted).sum()), int((labels & ~predicted).sum())
    binary = {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }
    positive_scores, negative_scores = scores[labels], scores[~labels]
    binary["auroc"] = float(
        np.mean(
            (positive_scores[:, None] > negative_scores[None, :])
            + 0.5 * (positive_scores[:, None] == negative_scores[None, :])
        )
    )
    # Distinct score thresholds preserve ties as groups, unlike row-wise AP.
    previous_recall, ap = 0.0, 0.0
    for threshold in sorted(set(scores), reverse=True):
        chosen = scores >= threshold
        recall = labels[chosen].sum() / labels.sum()
        ap += (recall - previous_recall) * labels[chosen].mean()
        previous_recall = recall
    binary["auprc"] = float(ap)
    for key, value in binary.items():
        if value is None:
            assert expected["binding_500nM"][key] is None
        else:
            assert np.isclose(
                value, expected["binding_500nM"][key], atol=1e-10, rtol=1e-9
            ), key
    return {**computed, **{"binding_500nM_" + k: v for k, v in binary.items()}}


def bootstrap_check(frame, expected, space):
    """Recompute paired intervals from peptide-level SSE totals and draw counts."""
    exact = frame[frame.qualifier == 0].copy()
    names = ["presto", "train_mean", "length_composition_class_ridge"]
    squared = pd.DataFrame(
        {
            name: (
                exact[name + "_prediction_log10_nM"] - exact[space + "_target_log10_nM"]
            )
            ** 2
            for name in names
        }
    )
    squared["peptide"] = exact.peptide
    totals = squared.groupby("peptide", sort=True)[names].sum().to_numpy()
    counts = exact.groupby("peptide", sort=True).size().to_numpy()
    assert expected["seed"] == 4242 and expected["n_resamples"] == 1000
    assert expected["n_exact_observations"] == len(exact)
    assert expected["n_peptides"] == len(counts)
    rng = np.random.default_rng(4242)
    differences = []
    for _ in range(1000):
        draws = rng.integers(len(counts), size=len(counts))
        weights = np.bincount(draws, minlength=len(counts))
        rmse = np.sqrt((weights @ totals) / (weights @ counts))
        differences.append(rmse[0] - rmse[1:])
    intervals = np.quantile(differences, [0.025, 0.975], axis=0).T
    for name, interval in zip(names[1:], intervals):
        assert np.allclose(
            interval,
            expected["paired_presto_minus_baseline_rmse_95_intervals"][name],
            atol=1e-10,
            rtol=1e-9,
        ), (space, name)


def main():
    receipt = json.loads((FAMILY / "results/fit.json").read_text())
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(receipt["local_paths"]["run.tar.gz"]).parent
    )
    assert receipt["summary"]["status"] == "success"
    for name, expected in receipt["files"].items():
        assert pin(root / name) == {k: expected[k] for k in ("sha256", "size_bytes")}
    inventory = json.loads((root / "inventory.json").read_text())
    files, modification_times = {}, {}
    with tarfile.open(root / "run.tar.gz") as archive:
        for member in archive.getmembers():
            assert (
                member.isfile()
                and member.name in inventory
                and member.name not in files
            )
            data = archive.extractfile(member).read()
            assert {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            } == inventory[member.name]
            files[member.name] = data
            modification_times[member.name] = member.mtime
    assert files.keys() == inventory.keys()
    protocol = json.loads((FAMILY / "results/protocol-notes.json").read_text())
    clarified = datetime.fromisoformat(protocol["recorded_utc"]).timestamp()
    model_gate_time = modification_times["model-construction-policy-check.json"]
    initial_checkpoint_time = modification_times["initial_checkpoint.pt"]
    assert clarified < model_gate_time <= initial_checkpoint_time
    timeline = {
        "clarification_recorded_utc": protocol["recorded_utc"],
        "model_gate_file_written_utc": datetime.fromtimestamp(
            model_gate_time, timezone.utc
        ).isoformat(),
        "initial_checkpoint_file_written_utc": datetime.fromtimestamp(
            initial_checkpoint_time, timezone.utc
        ).isoformat(),
        "clarification_preceded_model_construction": True,
    }
    invocation = json.loads(files["invocation.json"])
    assert invocation["dirty"] is False
    for name in ("fit.py", "baselines.py", "common.py", "diagnostic.py"):
        assert hashlib.sha256(files[name]).hexdigest() == invocation["code_files"][name]
        assert files[name] == (FAMILY / "reproduce/source" / name).read_bytes()
    frozen_pins = json.loads(files["frozen-inputs.json"])
    for name, expected in frozen_pins.items():
        assert hashlib.sha256(files[name]).hexdigest() == expected["sha256"]
        assert len(files[name]) == expected["size"]
    declared = json.loads(files["prospective-supported-outputs.json"])
    support = json.loads(files["supported_outputs.json"])
    assert support["mode"] == "declared" and support["requirements_met"] is True
    supported = [row for row in support["outputs"] if row["status"] != "undeclared"]
    assert len(supported) == 1 and supported[0]["endpoint"] == "assays.IC50_nM"
    assert supported[0]["status"] == "requirements_met"
    assert support["source_contract"] == declared["source_contract"]
    frozen = declared["source_contract"]["frozen_membership"]
    for name, expected in frozen["artifacts"].items():
        assert inventory["frozen_membership/" + name] == expected
    membership = pd.read_csv(
        io.BytesIO(files["membership.tsv"]), sep="\t", keep_default_na=False, dtype=str
    )
    native = pd.read_csv(
        io.BytesIO(files["selected-source-contexts.tsv"]),
        sep="\t",
        keep_default_na=False,
        dtype=str,
    )
    assert (
        len(membership) == 9053
        and membership.sample_id.is_unique
        and native.evidence_row_id.is_unique
    )
    joined = native.set_index("evidence_row_id").loc[membership.evidence_row_id]
    assert joined.peptide.tolist() == membership.peptide.tolist()
    comparison, training = (
        json.loads(files["comparison.json"]),
        json.loads(files["training-diagnostic.json"]),
    )
    tables = {
        split: pd.read_csv(
            io.BytesIO(files[name]), sep="\t", keep_default_na=False, dtype=str
        )
        for split, name in (
            ("train", "training-known-examples.tsv"),
            ("val", "val-known-examples.tsv"),
            ("test", "test-known-examples.tsv"),
        )
    }
    for table in tables.values():
        for column in table:
            if column in (
                "target_log10_nM",
                "effective_target_log10_nM",
                "raw_target_log10_nM",
                "qualifier",
            ) or column.endswith("_prediction_log10_nM"):
                table[column] = pd.to_numeric(table[column])
        # pandas writes float32 targets with their shortest float32 decimal
        # representation. Restore that declared type before float64 arithmetic;
        # treating those decimals as original float64 changes their values.
        for column in ("target_log10_nM", "effective_target_log10_nM"):
            table[column] = table[column].astype(np.float32).astype(float)
    checked = []
    for split, table in tables.items():
        source = membership[
            (membership.source_partition == split)
            & (
                membership.source_endpoint
                == "half maximal inhibitory concentration (IC50)"
            )
        ]
        assert table.sample_id.is_unique and set(table.sample_id) == set(
            source.sample_id
        )
        source = source.set_index("sample_id").loc[table.sample_id].reset_index()
        pd.testing.assert_frame_equal(
            table[source.columns].reset_index(drop=True),
            source,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        contexts = native.set_index("evidence_row_id").loc[table.evidence_row_id]
        for column in contexts.columns:
            actual = table["native_" + column].reset_index(drop=True)
            expected = contexts[column].reset_index(drop=True).rename(actual.name)
            pd.testing.assert_series_equal(
                actual,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        raw = pd.to_numeric(table.source_value).to_numpy(float)
        assert np.allclose(table.raw_target_log10_nM, np.log10(raw), atol=1e-12, rtol=0)
        assert np.allclose(
            table.effective_target_log10_nM,
            np.log10(np.clip(raw.astype(np.float32), 0.001, 50000)),
            atol=1e-6,
            rtol=0,
        )
        assert np.array_equal(table.target_log10_nM, table.effective_target_log10_nM)
        q = table.source_inequality.map(
            {"": 0, "=": 0, "<": -1, "<=": -1, "≤": -1, ">": 1, ">=": 1, "≥": 1}
        )
        assert np.array_equal(q, table.qualifier)
        references = training if split == "train" else comparison[split]
        for space, predictors in references.items():
            assert space in {"effective", "raw"}
            for predictor, expected in predictors.items():
                checked.append(
                    {
                        "split": split,
                        "target_space": space,
                        "predictor": predictor,
                        **metric_check(table, predictor, expected, space),
                    }
                )
    train = tables["train"]
    exact = train[train.qualifier == 0]
    state = json.loads(files["baseline-state.json"])
    assert state["alpha"] == 1.0
    assert state["n_exact_training_observations"] == len(exact) == 1599
    assert np.isclose(
        state["intercept"], float(exact.target_log10_nM.mean()), atol=1e-12, rtol=0
    )
    for split, table in tables.items():
        assert np.allclose(table.train_mean_prediction_log10_nM, state["intercept"])
        x = np.array(
            [
                [
                    len(p),
                    *[p.count(aa) / len(p) for aa in "ACDEFGHIKLMNPQRSTVWY"],
                    float(c == "I"),
                    float(c == "II"),
                ]
                for p, c in zip(table.peptide, table.native_mhc_class)
            ]
        )
        predicted = (
            state["intercept"]
            + ((x - state["feature_mean"]) / state["feature_scale"])
            @ state["coefficients"]
        )
        assert np.allclose(
            predicted, table.length_composition_class_ridge_prediction_log10_nM
        )
        if split == "train":
            fit = x[table.qualifier.to_numpy() == 0]
            assert np.allclose(fit.mean(0), state["feature_mean"])
            scale = fit.std(0)
            scale[scale == 0] = 1
            assert np.allclose(scale, state["feature_scale"])
            fit = (fit - fit.mean(0)) / scale
            coefficient = np.linalg.solve(
                fit.T @ fit + np.eye(fit.shape[1]),
                fit.T @ (exact.target_log10_nM - state["intercept"]),
            )
            assert np.allclose(coefficient, state["coefficients"])
    ledger_reports = {}
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("val", "test"):
            ledger = json.loads(files[split + "_loss_ledger.json"])
            batches = ledger["batches"]
            collection = SimpleNamespace(
                batch_losses=[b["reconstructed_losses"] for b in batches],
                batch_observation_counts=[b["observation_counts"] for b in batches],
                batch_support=[b["support"] for b in batches],
            )
            path = Path(tmp) / "predictions.csv"
            path.write_bytes(files[split + "_predictions.csv"])
            replayed = reconcile_prediction_artifact(
                path, collection, [b["canonical"] for b in batches]
            )
            summary = json.loads(files[split + "_summary.json"])
            assert np.isclose(
                replayed["overall_loss"], summary["overall_loss"], rtol=2e-5, atol=2e-6
            )
            ledger_reports[split] = {
                "n_batches": len(batches),
                "overall_loss": replayed["overall_loss"],
                "all_persisted_objective_losses_reconciled": True,
            }
    decision = json.loads(files["decision.json"])
    uncertainty = json.loads(files["uncertainty.json"])
    for split in ("val", "test"):
        for space in ("effective", "raw"):
            bootstrap_check(tables[split], uncertainty[split][space], space)
    updates = json.loads(files["output_updates.json"])
    direct = next(
        row
        for row in updates["outputs"]
        if row["endpoint"] == "assays.IC50_nM" and row["column"] == ""
    )
    parameters = list(updates["additional_parameter_rows"])
    for row in updates["outputs"]:
        parameters.extend(row["parameter_rows"])
    n_nonfinite = sum(
        row["nonfinite_output_gradient_elements"] for row in updates["outputs"]
    )
    n_nonfinite += sum(
        row["initial_nonfinite_elements"]
        + row["nonfinite_gradient_elements"]
        + row["nonfinite_update_elements"]
        for row in parameters
    )
    finite_optimization = (
        updates["optimizer_steps"] == 200
        and updates["batches_aborted"] == 0
        and n_nonfinite == 0
        and direct["output_gradient_abs_sum"] > 0
    )
    improved = (
        training["effective"]["selected"]["rmse"]
        < training["effective"]["initial"]["rmse"]
    )
    assert decision["n_nonfinite_tracked_elements"] == n_nonfinite
    assert decision["finite_optimization"] == finite_optimization
    assert decision["training_rmse_improved"] == improved
    assert decision["bounded_fitting_check_passed"] == (
        finite_optimization and improved
    )
    ci = uncertainty["val"]["effective"][
        "paired_presto_minus_baseline_rmse_95_intervals"
    ]
    positive = (
        comparison["val"]["effective"]["presto"]["spearman"] is not None
        and comparison["val"]["effective"]["presto"]["spearman"] > 0
    )
    assert decision["validation_quality_screen_passed"] == (
        positive and ci["train_mean"][1] < 0
    )
    assert decision["preferred_over_ridge"] == (
        positive and ci["length_composition_class_ridge"][1] < 0
    )
    report = {
        "n_verified_archive_members": len(files),
        "n_source_verified_ic50_observations": 2614,
        "frozen_membership": frozen,
        "canonical_manifest_gate_passed": True,
        "n_source_verified_membership_observations": len(membership),
        "protocol_clarification_timeline": timeline,
        "independent_paired_bootstrap_recomputation": True,
        "numerical_reporting_notes": [
            "Serialized effective targets are restored to their declared float32 type before independent float64 arithmetic.",
            "Constant-predictor Pearson is null in independent metrics; original reports retain tiny finite rounding artifacts (#116).",
            "The original effective-space 500 nM classifier rounds four exact training boundary labels to nonbinder; raw-space physical diagnostics retain them. No held-out boundary labels differ (#116).",
        ],
        "supported_outputs": supported,
        "heldout_loss_reconciliation": ledger_reports,
        "decision": decision,
        "comparison": comparison,
        "training_diagnostic": training,
        "independent_metric_recomputation": checked,
        "limitations": [
            "Studies overlap; donor/specimen/acquisition identity is unknown.",
            "All other outputs remain undeclared; no broad readiness claim.",
            "Presto uses all observed families/censoring; baselines fit exact IC50 only.",
            "Effective targets clip source values/bounds; raw-target errors are separate diagnostics.",
            "Labels were inspected for qualification; this is exploratory peptide-disjoint evaluation.",
            "Eight elution rows record rat hosts with human HLA. Host/MHC species conflation (#115) is uncorrected in this historical run; its shared-objective influence is unquantified. See the additive native species audit.",
        ],
    }
    names = [
        "qualification-summary.json",
        "metric-comparison.tsv",
        "ic50-inspection.tsv",
        "prediction-panels.png",
        "loss-curves.png",
    ]

    def builder(inputs, outputs):
        write(outputs["qualification-summary.json"], report)
        pd.DataFrame(checked).to_csv(
            outputs["metric-comparison.tsv"], sep="\t", index=False
        )
        inspection = []
        columns = [
            "sample_id",
            "evidence_row_id",
            "source_partition",
            "peptide",
            "native_mhc_class",
            "source_mhc_alleles",
            "resolved_mhc_alleles",
            "source_endpoint",
            "source_value",
            "source_units",
            "source_inequality",
            "raw_target_log10_nM",
            "effective_target_log10_nM",
            "qualifier",
            "selected_prediction_log10_nM",
            "selected_residual_effective_log10_nM",
            "selected_residual_raw_log10_nM",
            "selected_censor_satisfied_effective",
            "selected_censor_satisfied_raw",
            "train_mean_prediction_log10_nM",
            "length_composition_class_ridge_prediction_log10_nM",
            "pmid",
            "doi",
            "reference_iri",
            "assay_iri",
            "native_cell_line_name",
            "native_cell_type",
            "native_condition_id",
            "native_condition_status",
            "native_condition_reference",
            "native_assay_method",
        ]
        for split, table in tables.items():
            compact = table.copy()
            if split != "train":
                for suffix in (
                    "prediction_log10_nM",
                    "residual_effective_log10_nM",
                    "residual_raw_log10_nM",
                    "censor_satisfied_effective",
                    "censor_satisfied_raw",
                ):
                    compact["selected_" + suffix] = compact["presto_" + suffix]
            inspection.append(compact[columns].sort_values("sample_id"))
        pd.concat(inspection, ignore_index=True).to_csv(
            outputs["ic50-inspection.tsv"], sep="\t", index=False
        )
        plt.rcParams.update(
            {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for axis, (split, pred, title) in zip(
            axes.flat,
            [
                ("train", "initial", "Training IC50: initialization"),
                ("train", "selected", "Training IC50: selected checkpoint"),
                ("val", "presto", "Validation IC50: selected checkpoint"),
                ("test", "presto", "Test IC50: final descriptive evaluation"),
            ],
        ):
            table = tables[split]
            exact = table.qualifier == 0
            x, y = table.target_log10_nM, table[pred + "_prediction_log10_nM"]
            axis.scatter(
                x[exact],
                y[exact],
                s=12,
                alpha=0.35,
                label=f"Exact ({int(exact.sum())})",
                color="#16627A",
            )
            axis.scatter(
                x[~exact],
                y[~exact],
                marker=">",
                s=18,
                alpha=0.4,
                label=f"Lower bound ({int((~exact).sum())})",
                color="#B75A30",
            )
            low, high = min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2
            axis.plot([low, high], [low, high], ls="--", color="gray", lw=1)
            axis.set(
                xlim=(low, high),
                ylim=(low, high),
                xlabel="Effective clipped target / bound [log10(nM)]",
                ylabel="Prediction [log10(nM)]",
                title=title,
            )
            axis.legend(fontsize=8)
        fig.savefig(outputs["prediction-panels.png"], dpi=180)
        plt.close(fig)
        history = pd.read_csv(io.BytesIO(files["metrics.csv"]))
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for split in ("train", "val"):
            curve = history[(history.split == split) & (history.metric == "loss")]
            axis.plot(curve.step, pd.to_numeric(curve.value), marker="o", label=split)
        axis.set(
            xlabel="Epoch",
            ylabel="Canonical aggregate loss",
            title="Bounded fit: checkpoint selection uses complete validation",
        )
        axis.set_xticks([1, 2])
        axis.legend()
        fig.savefig(outputs["loss-curves.png"], dpi=180)
        plt.close(fig)

    # Local paths participate in datacache dependency identity. A public replay
    # gets its own namespace; never force-refresh the original generation (#118).
    input_location = hashlib.sha256(str(root.resolve()).encode()).hexdigest()
    paths = materialize(
        Path.home()
        / ".hitlist/presto-effective-target-fit/analysis"
        / pin(__file__)["sha256"]
        / input_location,
        {
            name: {"path": str(root / name), "sha256": pin(root / name)["sha256"]}
            for name in ("run.tar.gz", "inventory.json")
        },
        transform={
            "version": "effective-target-fit-analysis-v1",
            "options": {
                "code_sha256": pin(__file__)["sha256"],
                "protocol_notes_sha256": pin(FAMILY / "results/protocol-notes.json")[
                    "sha256"
                ],
                "analysis_versions": {
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "pandas", "matplotlib", "presto")
                },
            },
        },
        outputs={name: {} for name in names},
        builder=builder,
    )
    for name, path in paths.items():
        shutil.copyfile(path, FAMILY / "results" / name)
    for name in (
        "comparison.json",
        "training-diagnostic.json",
        "uncertainty.json",
        "decision.json",
        "baseline-state.json",
        "model-construction-policy-check.json",
        "affinity_target_audit.json",
        "supported_outputs.json",
        "output_coverage.csv",
        "split_support.csv",
        "metrics.csv",
        "regularization.json",
        "val_summary.json",
        "test_summary.json",
    ):
        (FAMILY / "results" / name).write_bytes(files[name])
    receipt = {
        "raw_run": {
            n: pin(root / n) for n in ("run.tar.gz", "inventory.json", "summary.json")
        },
        "analysis_outputs": {n: {"path": str(p), **pin(p)} for n, p in paths.items()},
    }
    write(FAMILY / "results/analysis-receipt.json", receipt)
    print(
        json.dumps(
            {
                "decision": decision,
                "n_verified_archive_members": len(files),
                "paths": receipt["analysis_outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
