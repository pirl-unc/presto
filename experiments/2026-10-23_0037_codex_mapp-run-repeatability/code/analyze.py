"""Registered complete-run repeatability; no model fitting or specimen pairing."""

import gzip
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
COHORT = ROOT / "data/availability_recipes/2026-10-23-mapp-complete-run-composition-v1"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation(x, y, rank=False):
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float((spearmanr if rank else pearsonr)(x, y).statistic)


def metrics(records, baseline=False):
    pred, target, mse = [], [], []
    for r in records:
        p = np.zeros(len(r["prediction"])) if baseline else np.array(r["prediction"])
        t = np.array(r["target"])
        mse.append(float(np.mean((p - t) ** 2)))
        pred.extend(p)
        target.extend(t)
    if not records:
        return {"n_records": 0}
    return {
        "n_records": len(records),
        "n_parents": len({r["parent_sequence_sha256"] for r in records}),
        "n_peptide_predictions": len(pred),
        "mse": float(np.mean(mse)),
        "rmse": float(np.sqrt(np.mean(mse))),
        "pearson": correlation(pred, target),
        "spearman": correlation(pred, target, True),
        "prediction_sd": float(np.std(pred)),
        "target_sd": float(np.std(target)),
    }


def main():
    assert not (FAMILY / "results/repeatability-summary.json").exists()
    verified = read(FAMILY / "results/independent-admission.json")
    assert (
        verified["passed"]
        and sha(COHORT / "manifest.json") == verified["recipe_manifest_sha256"]
    )
    for name, info in read(COHORT / "manifest.json")["files"].items():
        assert sha(COHORT / name) == info["sha256"]
    receipt = {
        "argv": sys.argv,
        "python": sys.version,
        "numpy": np.__version__,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "recipe_manifest_sha256": sha(COHORT / "manifest.json"),
        "source_sha256": sha(__file__),
        "environment_overrides": {
            k: os.environ[k]
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            if k in os.environ
        },
        "training": None,
        "test_scope": "previously observed source/development strata; no new blind test",
    }
    dump(FAMILY / "reproduce/analysis.json", receipt)
    shutil.copy2(__file__, FAMILY / "reproduce/source/analyze.py")
    tick = time.monotonic()
    observations = rows(COHORT / "run_observations.jsonl")
    all_values = {
        (r["peptide_row_id"], r["channel"]["channel_id"]): r for r in observations
    }
    channels = defaultdict(set)
    for r in observations:
        channels[r["condition"]].add(r["channel"]["channel_id"])
    channels = {k: sorted(v) for k, v in channels.items()}
    assert set(channels) == {"UT", "TNF+IFN"} and all(
        len(v) == 3 for v in channels.values()
    )
    examples = [r for r in rows(COHORT / "examples.jsonl") if r["condition"] == "UT"]
    effects, predictions = [], []
    for e in examples:
        peptides = [r["peptide_row_id"] for r in e["candidates"]]
        raw = np.array(
            [
                [
                    [all_values[p, c]["measurement"]["value"] for p in peptides]
                    for c in channels[condition]
                ]
                for condition in ("UT", "TNF+IFN")
            ]
        )
        assert np.isfinite(raw).all() and (raw > 0).all()
        log = np.log10(raw)
        centered = log - log.mean(axis=2, keepdims=True)
        delta = centered[1].mean(0) - centered[0].mean(0)
        variance = centered.var(axis=1, ddof=1)
        noise = variance.sum(0) / 3
        common = {k: e[k] for k in ("parent_sequence_sha256", "component_id", "split")}
        common["peptide_ids"] = peptides
        effects.append(
            {
                **common,
                "mean_log10_change": delta.tolist(),
                "within_condition_sample_variance": variance.tolist(),
                "expected_contrast_sampling_variance": noise.tolist(),
                "observed_mean_squared_change": float(np.square(delta).mean()),
                "mean_contrast_sampling_variance": float(noise.mean()),
                "variance_adjusted_mean_squared_change": float(
                    (np.square(delta) - noise).mean()
                ),
                "centered_log10_run_values": centered.tolist(),
            }
        )
        for u in range(3):
            for t in range(3):
                p = np.delete(centered[1], t, axis=0).mean(0) - np.delete(
                    centered[0], u, axis=0
                ).mean(0)
                target = centered[1, t] - centered[0, u]
                predictions.append(
                    {
                        **common,
                        "task": "condition_change",
                        "heldout_channels": [channels["UT"][u], channels["TNF+IFN"][t]],
                        "training_channels": [
                            c for i, c in enumerate(channels["UT"]) if i != u
                        ]
                        + [c for i, c in enumerate(channels["TNF+IFN"]) if i != t],
                        "prediction": p.tolist(),
                        "target": target.tolist(),
                        "baseline": "zero_change",
                    }
                )
        for c, condition in enumerate(("UT", "TNF+IFN")):
            for h in range(3):
                p = np.delete(centered[c], h, axis=0).mean(0)
                predictions.append(
                    {
                        **common,
                        "task": "same_condition_composition",
                        "condition": condition,
                        "heldout_channels": [channels[condition][h]],
                        "training_channels": [
                            v for i, v in enumerate(channels[condition]) if i != h
                        ],
                        "prediction": p.tolist(),
                        "target": centered[c, h].tolist(),
                        "baseline": "uniform_composition",
                    }
                )
    summary = {
        "by_scope": {},
        "by_task": {},
        "n_complete_parents": len(examples),
        "n_complete_peptides": sum(len(e["candidates"]) for e in examples),
        "base_run_value_statuses": dict(
            Counter(r["measurement"]["value_status"] for r in observations)
        ),
        "base_runs_with_direct_msms": sum(
            bool(r["direct_msms_evidence_ids"]) for r in observations
        ),
        "complete_runs_with_direct_msms": sum(
            bool(r["direct_msms_evidence_ids"])
            for r in observations
            if r["admitted_complete_parent_set"]
        ),
        "scope": "Selected complete captured-intensity observations; acquisition repeatability, not calibrated biological noise or concentration",
    }
    for split in ("all", "train", "validation", "test"):
        subset = [r for r in effects if split == "all" or r["split"] == split]
        squared = float(np.mean([r["observed_mean_squared_change"] for r in subset]))
        noise = float(np.mean([r["mean_contrast_sampling_variance"] for r in subset]))
        summary["by_scope"][split] = {
            "n_parents": len(subset),
            "observed_change_rms": float(np.sqrt(squared)),
            "expected_sampling_noise_rms": float(np.sqrt(noise)),
            "observed_mean_squared_change": squared,
            "mean_contrast_sampling_variance": noise,
            "variance_adjusted_mean_squared_change": squared - noise,
            "fraction_parents_positive_variance_adjusted_change": sum(
                r["variance_adjusted_mean_squared_change"] > 0 for r in subset
            )
            / len(subset),
        }
        for task in ("condition_change", "same_condition_composition"):
            records = [
                r
                for r in predictions
                if r["task"] == task and (split == "all" or r["split"] == split)
            ]
            summary["by_task"][split + ":" + task] = {
                "other_run_mean": metrics(records),
                "zero_baseline": metrics(records, baseline=True),
            }
    dump(FAMILY / "results/parent-effects.json", effects)
    payload = "".join(
        json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in predictions
    ).encode()
    (FAMILY / "results/run-predictions.jsonl.gz").write_bytes(
        gzip.compress(payload, mtime=0)
    )
    summary["seconds"] = time.monotonic() - tick
    summary["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump(FAMILY / "results/repeatability-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
