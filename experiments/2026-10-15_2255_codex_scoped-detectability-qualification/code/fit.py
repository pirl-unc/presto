"""Registered bounded canonical fit with train-only baselines and frozen membership."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import datacache
import hitlist
import numpy as np
import pandas as pd
import presto
import torch
from datacache import materialize, validate_file
from hitlist.downloads import data_dir
from presto.scripts.train_iedb import run
from presto.training.holdout_eval import binary_metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def features(peptides):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return np.asarray(
        [[len(p), *(p.count(aa) / len(p) for aa in amino_acids)] for p in peptides],
        dtype=np.float64,
    )


def metrics(labels, probabilities):
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Nonfinite or invalid probability in held-out evaluation")
    result = binary_metrics(labels, probabilities)
    result["n_samples"] = int(result.pop("n"))
    result["n_positive"] = int(result["n_positive"])
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    result["log_loss"] = float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    )
    result["brier"] = float(np.mean((probabilities - labels) ** 2))
    return result


def bootstrap_comparison(labels, probabilities):
    """Paired peptide bootstrap; intervals do not represent independent experiments."""
    rng = np.random.default_rng(4242)
    draws = {name: [] for name in probabilities}
    for _ in range(1000):
        indices = rng.integers(0, len(labels), size=len(labels))
        target = labels[indices]
        if len(np.unique(target)) != 2:
            continue
        for name, values in probabilities.items():
            result = metrics(target, values[indices])
            draws[name].append([result[key] for key in ("log_loss", "auroc", "auprc")])
    result = {
        "seed": 4242,
        "n_requested_resamples": 1000,
        "n_valid_resamples": len(draws["presto"]),
        "sampling_unit": "peptide within one search; shared proteins are not independent",
        "percentile_95_intervals": {},
        "paired_presto_minus_baseline_95_intervals": {},
    }
    for name, values in draws.items():
        array = np.asarray(values)
        result["percentile_95_intervals"][name] = {
            key: np.quantile(array[:, index], [0.025, 0.975]).tolist()
            for index, key in enumerate(("log_loss", "auroc", "auprc"))
        }
        if name != "presto":
            delta = np.asarray(draws["presto"]) - array
            result["paired_presto_minus_baseline_95_intervals"][name] = {
                key: np.quantile(delta[:, index], [0.025, 0.975]).tolist()
                for index, key in enumerate(("log_loss", "auroc", "auprc"))
            }
    return result


def compare_baselines(work):
    membership = pd.read_parquet(work / "membership/membership.parquet")
    if membership.sample_id.duplicated().any():
        raise ValueError("Duplicate frozen sample IDs")
    train = membership[membership.source_partition == "train"]
    labels = train.detectability_label.to_numpy(dtype=float)
    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4, random_state=42),
    )
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        baseline.fit(features(train.peptide), labels)
    baseline_fit_seconds = time.monotonic() - started
    scaler, logistic = baseline.steps[0][1], baseline.steps[1][1]
    baseline_state = {
        "feature_names": ["length", *["fraction_" + aa for aa in "ACDEFGHIKLMNPQRSTVWY"]],
        "train_prevalence": float(labels.mean()),
        "n_training_peptides": len(train),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": logistic.coef_.tolist(),
        "intercept": logistic.intercept_.tolist(),
        "classes": logistic.classes_.tolist(),
        "n_iterations": logistic.n_iter_.tolist(),
        "configuration": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-4,
            "class_weight": None,
        },
        "fit_seconds": baseline_fit_seconds,
        "fit_scope": "train only; no held-out calibration, threshold or feature selection",
    }
    (work / "baseline_state.json").write_text(json.dumps(baseline_state, indent=2) + "\n")
    results, uncertainty = [], {}
    for split in ("val", "test"):
        expected = membership[membership.source_partition == split].set_index("sample_id")
        predictions = pd.read_csv(work / f"{split}_predictions.csv", keep_default_na=False)
        if set(predictions.task) != {"ms_detectability"}:
            raise ValueError(
                "Unexpected held-out task; this experiment declares detectability only"
            )
        if predictions.sample_id.duplicated().any() or set(predictions.sample_id) != set(
            expected.index
        ):
            raise ValueError("Held-out predictions do not match exact frozen membership")
        predictions = predictions.set_index("sample_id").loc[expected.index]
        for field in (
            "peptide",
            "source_partition",
            "source_bundle_id",
            "scoped_peptide_id",
            "candidate_id",
        ):
            if predictions[field].tolist() != expected[field].tolist():
                raise ValueError(f"Held-out prediction lineage changed: {field}")
        target = expected.detectability_label.to_numpy(dtype=float)
        if not np.array_equal(predictions.y_true.to_numpy(dtype=float), target):
            raise ValueError("Held-out labels changed")
        probabilities = {
            "presto": predictions.y_prob.to_numpy(dtype=float),
            "train_prevalence": np.full(len(expected), labels.mean()),
            "length_composition_logistic": baseline.predict_proba(features(expected.peptide))[:, 1],
        }
        # Retain all membership context, evidence categories and lineage beside every score.
        readable = expected.reset_index().copy()
        for name, values in probabilities.items():
            readable[name + "_probability"] = values
            results.append({"split": split, "predictor": name, **metrics(target, values)})
        readable.to_csv(work / f"{split}_comparison.tsv", sep="\t", index=False)
        uncertainty[split] = bootstrap_comparison(target, probabilities)
    pd.DataFrame(results).to_csv(work / "comparison.csv", index=False)
    (work / "comparison.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    (work / "uncertainty.json").write_text(
        json.dumps(uncertainty, indent=2, allow_nan=False) + "\n"
    )
    return results


def main():
    family = Path(__file__).resolve().parents[1]
    repository = family.parents[1]
    config_path = family / "reproduce/fit.json"
    manifest_path = family / "reproduce/supported_outputs.json"
    config = json.loads(config_path.read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)
    )
    if dirty:
        raise RuntimeError("Commit the prospective fitting contract and code before running")
    options = {
        "stage": "fit",
        "commit": commit,
        "dirty": dirty,
        "presto_version": presto.__version__,
        "recipe_sha256": sha(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "pandas", "pyarrow", "numpy", "scipy", "scikit-learn")
            },
            "hitlist_runtime": hitlist.__version__,
            "datacache_runtime": datacache.__version__,
        },
        "dependency_imports": {"hitlist": hitlist.__file__, "datacache": datacache.__file__},
        "torch_threads": 1,
        "gpu": "none",
        "wall_clock_limit_seconds": 1800,
    }
    key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(1)

    def builder(inputs, outputs):
        with TemporaryDirectory(prefix="presto-scoped-fit-") as temporary:
            work = Path(temporary)
            started = datetime.now(timezone.utc).isoformat()
            before = time.monotonic()
            status, membership, comparison = "success", {}, []

            def timeout(signum, frame):
                raise TimeoutError("Canonical fit and baseline evaluation exceeded 30 minutes")

            handler = signal.signal(signal.SIGALRM, timeout)
            signal.alarm(1800)
            try:
                with (
                    (work / "training.log").open("w") as log,
                    contextlib.redirect_stdout(log),
                    contextlib.redirect_stderr(log),
                ):
                    run(
                        argparse.Namespace(
                            **config,
                            supported_output_manifest=str(inputs["supported_outputs.json"]),
                            run_dir=str(work),
                            checkpoint=str(work / "checkpoint.pt"),
                        )
                    )
                funnel = json.loads((work / "data_funnel.json").read_text())
                membership = funnel["frozen_membership"]
                destination = work / "membership"
                destination.mkdir()
                for name, pin in membership["artifacts"].items():
                    source = Path(membership["directory"]) / name
                    validate_file(source, pin["sha256"])
                    shutil.copyfile(source, destination / name)
                comparison = compare_baselines(work)
            except Exception:
                status = "failed"
                (work / "error.txt").write_text(traceback.format_exc())
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, handler)
            elapsed = time.monotonic() - before
            for name in inputs:
                shutil.copyfile(inputs[name], work / name)
            inventory = {
                path.relative_to(work).as_posix(): {
                    "sha256": sha(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(work.rglob("*"))
                if path.is_file()
            }
            Path(outputs["inventory.json"]).write_text(json.dumps(inventory, indent=2) + "\n")
            with tarfile.open(outputs["run.tar.gz"], "w:gz") as archive:
                for name in inventory:
                    archive.add(work / name, arcname=name, recursive=False)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            summary = {
                **options,
                "status": status,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": elapsed,
                "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
                "rss_scope": "runner process only; child processes excluded",
                "config": config,
                "membership": membership,
                "comparison": comparison,
                "artifacts": {
                    name: {
                        "sha256": sha(outputs[name]),
                        "size_bytes": Path(outputs[name]).stat().st_size,
                    }
                    for name in ("inventory.json", "run.tar.gz")
                },
            }
            Path(outputs["summary.json"]).write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n"
            )

    paths = materialize(
        data_dir() / "presto-scoped-qualification" / "fit" / key,
        {
            name: {"path": str(path), "sha256": sha(path)}
            for name, path in {
                "fit.json": config_path,
                "supported_outputs.json": manifest_path,
                "fit.py": Path(__file__),
            }.items()
        },
        transform={"version": "canonical-scoped-fit-v1", "options": options},
        outputs={name: {} for name in ("run.tar.gz", "inventory.json", "summary.json")},
        builder=builder,
    )
    summary = json.loads(Path(paths["summary.json"]).read_text())
    result = {
        "directory": str(Path(paths["summary.json"]).parent),
        "summary_sha256": sha(paths["summary.json"]),
        **summary,
    }
    print(json.dumps(result, indent=2), flush=True)
    if summary["status"] != "success":
        raise RuntimeError(f"Fit failed; preserved evidence: {result['directory']}")


if __name__ == "__main__":
    main()
