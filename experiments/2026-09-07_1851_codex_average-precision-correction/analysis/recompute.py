"""Reissue only verifiable AP corrections from frozen canonical prediction dumps."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "frozen_holdout_eval", EXP / "reproduce/source/holdout_eval.py"
)
holdout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(holdout)
CORRECT_AP = holdout.auprc


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_ap(labels, scores):
    positive = labels > 0.5
    n_positive = int(positive.sum())
    if n_positive == 0 or n_positive == len(labels):
        return None
    hits = positive[np.argsort(-scores, kind="mergesort")].astype(float)
    return float(np.sum(np.cumsum(hits) / np.arange(1, len(hits) + 1) * hits) / n_positive)


def main():
    started = time.perf_counter()
    manifest = json.loads((EXP / "reproduce/inputs.json").read_text())
    result_dir = EXP / "results"
    result_dir.mkdir(exist_ok=True)
    types = manifest["loss_types"]
    rows_out, files_out = [], []
    for entry in manifest["selected"]:
        path = ROOT / entry["path"]
        assert digest(path) == entry["sha256"], path
        with path.open() as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            observations = list(reader)
        accumulators = {}
        skipped_tasks = set()
        for row in observations:
            task = row["task"]
            if task not in types or (types[task] == "censor" and "qualifier" not in fields):
                skipped_tasks.add(task)
                continue
            acc = accumulators.setdefault(task, holdout.TaskPredictionAccumulator(task, types[task]))
            acc.add(
                [float(row["y_true"])], [float(row["y_pred"])], [1.0],
                sample_ids=[row.get("sample_id", "")],
                sources=[row.get("source", "")],
                qualifiers=[int(row.get("qualifier", 0))],
                mapping_categories=[row.get("source_mapping_category", "")],
            )
        original = None
        if entry["summary_path"]:
            summary_path = ROOT / entry["summary_path"]
            assert digest(summary_path) == entry["summary_sha256"], summary_path
            original = json.loads(summary_path.read_text())
        corrected = json.loads(json.dumps(original)) if original else None
        replacements = []
        for task, acc in accumulators.items():
            calls = []

            def trace_ap(labels, scores):
                new = CORRECT_AP(labels, scores)
                if new is not None:
                    _, counts = np.unique(scores, return_counts=True)
                    calls.append({
                        "n": len(labels), "n_positive": int((labels > 0.5).sum()),
                        "tied_groups": int((counts > 1).sum()),
                        "tied_rows": int(counts[counts > 1].sum()),
                        "old_ap": legacy_ap(labels, scores), "new_ap": new,
                    })
                return new

            holdout.auprc = trace_ap
            try:
                metrics = acc.metrics()
            finally:
                holdout.auprc = CORRECT_AP
            ap_metrics = [(k, v) for k, v in metrics.items() if k.endswith("auprc")]
            assert len(ap_metrics) == len(calls)
            for (metric, value), call in zip(ap_metrics, calls):
                assert value == call["new_ap"]
                old_archived = (original or {}).get("tasks", {}).get(task, {}).get(metric)
                matches = old_archived is not None and math.isclose(
                    old_archived, call["old_ap"], rel_tol=1e-9, abs_tol=1e-12
                )
                changed = not math.isclose(call["new_ap"], call["old_ap"], abs_tol=1e-12, rel_tol=0)
                status = (
                    "verified_correction" if matches and changed else
                    "verified_unchanged" if matches else
                    "no_archived_metric" if old_archived is None else
                    "archived_value_mismatch"
                )
                if matches and changed:
                    corrected["tasks"][task][metric] = value
                    replacements.append(f"{task}.{metric}")
                rows_out.append({
                    "path": entry["path"], "split": entry["split"],
                    "task": task, "metric": metric, **call,
                    "delta": call["new_ap"] - call["old_ap"],
                    "archived_ap": old_archived, "status": status,
                })
        corrected_path = None
        if replacements:
            # Mixed provenance is explicit: all non-reissued metrics retain their
            # original estimator/version and original numerical value.
            corrected["ap_correction"] = {
                "estimator": holdout.AUPRC_ESTIMATOR,
                "source_summary": entry["summary_path"],
                "source_summary_sha256": entry["summary_sha256"],
                "source_predictions": entry["path"],
                "source_predictions_sha256": entry["sha256"],
                "reissued_metrics": replacements,
                "other_metrics": "Unchanged historical values; not revalidated by this correction.",
            }
            corrected_path = f"results/reissued/{entry['id']}.json"
            target = EXP / corrected_path
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps(corrected, indent=2) + "\n")
        files_out.append({
            **entry, "rows": len(observations), "skipped_tasks": sorted(skipped_tasks),
            "reissued_metrics": len(replacements), "corrected_summary": corrected_path,
        })
    with (result_dir / "ap_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    # Independent local reference; this is not a package/runtime dependency.
    import sklearn
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(52)
    max_error = 0.0
    for case in range(500):
        labels = rng.integers(0, 2, size=int(rng.integers(2, 400)))
        labels[:2] = [0, 1]
        scores = rng.normal(size=len(labels)) if case % 2 else rng.integers(0, 5, len(labels))
        error = abs(CORRECT_AP(labels, scores) - average_precision_score(labels, scores))
        max_error = max(max_error, error)
        assert error < 1e-12
    statuses = {status: sum(row["status"] == status for row in rows_out) for status in
                sorted({row["status"] for row in rows_out})}
    result = {
        "kind": "Retrospective AP correction; no model run or training",
        "agent_model": "Codex / GPT-6", "estimator": holdout.AUPRC_ESTIMATOR,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "selected_files": len(files_out), "selected_rows": sum(f["rows"] for f in files_out),
        "excluded_files": len(manifest["excluded"]), "metric_statuses": statuses,
        "changed_files": sum(f["reissued_metrics"] > 0 for f in files_out),
        "max_absolute_ap_delta": max(abs(row["delta"]) for row in rows_out),
        "reference": {"library": "scikit-learn", "version": sklearn.__version__,
                      "cases": 500, "max_absolute_error": max_error},
        "runtime_seconds": time.perf_counter() - started,
        "files": files_out,
    }
    (result_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
