"""Independent sequence-to-feature/NumPy-score replay; test only chosen models."""

import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import qr
from scipy.special import logsumexp
from scipy.stats import rankdata

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
COHORT = ROOT / "data/availability_recipes/2026-10-22-mapp-captured-composition-min2-v1"
OUTPUT = ROOT / "artifacts" / FAMILY.name / "linear-fits"
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    if a is None:
        assert b is None
    else:
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-11)


def correlation(a, b, rank=False):
    a, b = np.asarray(a), np.asarray(b)
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    if rank:
        a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / math.sqrt(float(a @ a) * float(b @ b)))


def metric(records):
    losses, pred, target, paired = [], [], [], {}
    for r in records:
        q, t = np.array(r["logq"]), np.array(r["logt"])
        p, y = (q - q.mean()) / np.log(10), (t - t.mean()) / np.log(10)
        if np.ptp(q) == 0:
            p = np.zeros_like(p)
        losses.append(
            [
                np.dot(np.exp(q), q - t),
                np.dot(np.exp(t), t - q),
                np.abs(np.exp(q) - np.exp(t)).sum() / 2,
                np.square(p - y).mean(),
            ]
        )
        pred.extend(p)
        target.extend(y)
        paired.setdefault(r["parent_sequence_sha256"], {})[r["condition"]] = (p, y)
    dp, dt, squared = [], [], []
    for parent in sorted(paired):
        u, t = paired[parent]["UT"], paired[parent]["TNF+IFN"]
        p, y = t[0] - u[0], t[1] - u[1]
        dp.extend(p)
        dt.extend(y)
        squared.append(np.square(p - y).mean())
    means = np.array(losses).mean(0)
    return {
        "n_examples": len(records),
        "n_parents": len(paired),
        "n_peptide_condition_predictions": len(pred),
        "reverse_kl": float(means[0]),
        "forward_kl": float(means[1]),
        "total_variation": float(means[2]),
        "centered_log10_rmse": float(np.sqrt(means[3])),
        "centered_log10_pearson": correlation(pred, target),
        "centered_log10_spearman": correlation(pred, target, True),
        "prediction_centered_log10_sd": float(np.std(pred)),
        "target_centered_log10_sd": float(np.std(target)),
        "condition_change_log10_rmse": float(np.sqrt(np.mean(squared))),
        "condition_change_pearson": correlation(dp, dt),
        "condition_change_spearman": correlation(dp, dt, True),
        "condition_change_prediction_sd": float(np.std(dp)),
        "condition_change_target_sd": float(np.std(dt)),
    }


def rebuild(examples, parents, architecture, contract):
    matrices = []
    for e in examples:
        sequence = parents[e["parent_sequence_sha256"]]
        index = np.array([AA.index(a) for a in sequence] + [20])
        onehot = np.eye(21)[index]
        features = []
        for c in e["candidates"]:
            assert len(c["start_positions_0based"]) == 1
            start = c["start_positions_0based"][0]
            end = start + len(c["sequence"])
            assert sequence[start:end] == c["sequence"]
            composition = np.bincount(index[start:end], minlength=20) / (end - start)
            features.append(
                np.r_[
                    [(end - start) / 40],
                    composition,
                    onehot[start, :20],
                    onehot[end - 1, :20],
                    onehot[start - 1 if start else len(sequence)],
                    onehot[end],
                    [start / max(len(sequence) - 1, 1), end / len(sequence)],
                ]
            )
        x = np.array(features)
        x -= x.mean(0)
        if architecture == "length":
            x = x[:, :1]
        if architecture == "interaction":
            x = np.c_[x, float(e["condition"] == "TNF+IFN") * x]
        matrices.append(x)
    train = np.concatenate(
        [x for x, e in zip(matrices, examples) if e["split"] == "train"]
    )
    rms = np.sqrt((train * train).sum(0) / len(train))
    scales = np.where(rms > 1e-12, rms, 1)
    active = np.flatnonzero(rms > 1e-12)
    _, r, pivots = qr(train[:, active] / scales[active], mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    rank = int((diagonal > diagonal.max() * 1e-10).sum())
    selected = np.sort(active[pivots[:rank]])
    np.testing.assert_array_equal(selected, contract["selected_columns"])
    np.testing.assert_array_equal(pivots, contract["qr_pivots"])
    close(rms, contract["train_rms"])
    close(diagonal, contract["qr_abs_diagonal"])
    assert rank == contract["rank"]
    projection = np.linalg.lstsq(
        train[:, selected] / scales[selected], train / scales, rcond=1e-10
    )[0]
    close(projection, contract["training_projection"])
    for split in ("train", "validation", "test"):
        x = np.concatenate(
            [x for x, e in zip(matrices, examples) if e["split"] == split]
        )
        residual = x / scales - x[:, selected] / scales[selected] @ projection
        audit = contract["exposure_audit"][split]
        close(float(np.abs(residual).max()), audit["max_abs_residual"])
        assert (
            int(np.any(abs(residual) > 1e-6, axis=1).sum())
            == audit["n_rows_residual_gt_1e_minus6"]
        )
    return [
        (x[:, selected] / scales[selected]).astype(np.float32).astype(np.float64)
        for x in matrices
    ]


def main():
    tick = time.monotonic()
    torch.set_num_threads(1)
    receipt = read(FAMILY / "reproduce/linear-training.json")
    for name, value in receipt["source_hashes"].items():
        assert sha(FAMILY / "reproduce/source/linear" / name) == value
    assert sha(COHORT / "manifest.json") == receipt["recipe_manifest_sha256"]
    examples = rows(COHORT / "examples.jsonl")
    lookup = {e["example_id"]: i for i, e in enumerate(examples)}
    parents = {
        r["parent_sequence_sha256"]: r["sequence"]
        for r in rows(COHORT / "parents.jsonl")
    }
    contracts = read(FAMILY / "results/feature-contracts.json")
    matrices = {a: rebuild(examples, parents, a, c) for a, c in contracts.items()}
    summary = read(FAMILY / "results/linear-summary.json")
    assert len(summary["runs"]) == 30
    selected_names = {r["name"] for r in summary["selected"]}
    assert len(selected_names) == 6
    max_error = 0.0
    validation_batch_differences = {}
    values_verified = 0
    best_metrics = {}
    for run in summary["runs"]:
        directory = OUTPUT / run["name"]
        assert read(directory / "summary.json") == run
        history = read(directory / "history.json")
        assert [h["step"] for h in history] == list(range(0, 1001, 25))
        best = min(history, key=lambda h: h["validation"]["reverse_kl"])
        assert (
            best["step"] == run["selected_step"]
            and best["validation"]["reverse_kl"] == run["validation_reverse_kl"]
        )
        initial = torch.load(directory / "initial.pt", weights_only=True)
        chosen = torch.load(directory / "selected.pt", weights_only=True)
        last = torch.load(directory / "last.pt", weights_only=True)
        assert not initial["theta"].any() and chosen["step"] == run["selected_step"]
        assert (
            int(last["coverage"].sum())
            == run["n_nonzero_gradient"]
            == run["n_parameters"]
        )
        assert int((last["theta"] != 0).sum()) == run["n_final_changed"]
        assert int(chosen["coverage"].sum()) == run["n_selected_nonzero_gradient"]
        assert int((chosen["theta"] != 0).sum()) == run["n_selected_changed"]
        assert all(int(s["step"]) == 1000 for s in last["optimizer"]["state"].values())
        assert (
            torch.isfinite(last["theta"]).all()
            and torch.isfinite(last["max_abs_gradient"]).all()
        )
        assert (
            run["updates"] == 1000
            and run["scored_valid_candidates"] == 918000
            and run["scored_padded_candidates"] == 16128000
        )
        assert run["target_terms"] == (918000 if run["method"] == "exact" else 9216000)
        all_path = directory / "all-split-predictions.jsonl.gz"
        assert all_path.exists() == (run["name"] in selected_names)
        predictions = rows(
            all_path
            if all_path.exists()
            else directory / "validation-predictions.jsonl.gz"
        )
        assert len(predictions) == (202 if all_path.exists() else 30)
        theta = chosen["theta"].numpy().astype(np.float64)
        for row in predictions:
            i = lookup[row["example_id"]]
            e = examples[i]
            assert row["peptide_ids"] == [c["peptide_row_id"] for c in e["candidates"]]
            assert row["logt"] == [c["target_log_probability"] for c in e["candidates"]]
            assert row["source_log10_intensities"] == [
                c["source_log10_intensity"] for c in e["candidates"]
            ]
            for key in ("split", "condition", "component_id", "parent_sequence_sha256"):
                assert row[key] == e[key]
            logits = matrices[run["architecture"]][i] @ theta
            logq = logits - logsumexp(logits)
            err = float(np.max(abs(logq - row["logq"])))
            max_error = max(max_error, err)
            np.testing.assert_allclose(logq, row["logq"], atol=3e-5, rtol=0)
            values_verified += len(logq)
        val = rows(directory / "validation-predictions.jsonl.gz")
        assert len(val) == 30 and all(r["split"] == "validation" for r in val)
        computed = metric(val)
        for k, v in computed.items():
            close(v, best["validation"][k])
        if all_path.exists():
            by_id = {
                r["example_id"]: r for r in predictions if r["split"] == "validation"
            }
            difference = max(
                float(
                    np.max(np.abs(np.array(r["logq"]) - by_id[r["example_id"]]["logq"]))
                )
                for r in val
            )
            assert difference < 3e-5
            validation_batch_differences[run["name"]] = difference
            record = next(r for r in summary["selected"] if r["name"] == run["name"])
            all_metrics = {}
            for split in ("train", "validation", "test"):
                m = metric([r for r in predictions if r["split"] == split])
                for k, v in m.items():
                    close(v, record["metrics"][split][k])
                all_metrics[split] = m
            best_metrics[run["name"]] = all_metrics
    for winner in summary["selected"]:
        candidates = [
            r
            for r in summary["runs"]
            if all(r[k] == winner[k] for k in ("architecture", "method", "seed"))
        ]
        expected = min(
            candidates,
            key=lambda r: (r["validation_reverse_kl"], -r["l2"], r["selected_step"]),
        )
        assert winner["name"] == expected["name"]
    probe = dict(np.load(FAMILY / "results/linear-gradient-probe.npz"))
    check = read(FAMILY / "results/linear-gradient-check.json")
    close(
        np.linalg.norm(probe["estimates"].mean(0) - probe["exact"])
        / np.linalg.norm(probe["exact"]),
        check["sampled_mean_relative_error"],
    )
    close(
        np.linalg.norm(
            probe["estimates"].std(0, ddof=1) / np.sqrt(len(probe["estimates"]))
        ),
        check["mean_gradient_se_l2"],
    )
    output = {
        "passed": True,
        "n_fits": 30,
        "n_test_selected_models": 6,
        "n_prediction_values_verified": values_verified,
        "max_numpy_logq_error": max_error,
        "validation_only_vs_all_split_batch_logq_differences": validation_batch_differences,
        "seconds": time.monotonic() - tick,
        "verifier_sha256": sha(__file__),
        "artifact_hashes": {
            str(p.relative_to(OUTPUT)): sha(p)
            for p in sorted(OUTPUT.rglob("*"))
            if p.is_file()
        },
    }
    (FAMILY / "results/independent-linear-metrics.json").write_text(
        json.dumps(best_metrics, indent=2, sort_keys=True) + "\n"
    )
    (FAMILY / "results/independent-linear-verification.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: v for k, v in output.items() if k != "artifact_hashes"}))


if __name__ == "__main__":
    main()
