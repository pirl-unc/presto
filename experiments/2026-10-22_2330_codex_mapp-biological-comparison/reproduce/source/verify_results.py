"""Independent checkpoint, NumPy forward and saved-metric replay; no trainer imports."""

import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import rankdata

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
ARTIFACTS = ROOT / "artifacts" / FAMILY.name
COHORT = ROOT / "data/availability_recipes/2026-10-22-mapp-captured-composition-min2-v1"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYUO"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corr(x, y, rank=False):
    x, y = np.asarray(x), np.asarray(y)
    if min(len(x), len(y)) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if rank:
        x, y = rankdata(x), rankdata(y)
    x, y = x - x.mean(), y - y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def independent_metrics(records):
    if not records:
        return {"n_examples": 0}
    pq, pt, losses, paired = [], [], [], {}
    for r in records:
        qlog, tlog = np.array(r["logq"]), np.array(r["logt"])
        q, t = np.exp(qlog), np.exp(tlog)
        p, y = (qlog - qlog.mean()) / np.log(10), (tlog - tlog.mean()) / np.log(10)
        losses.append(
            [
                np.dot(q, qlog - tlog),
                np.dot(t, tlog - qlog),
                np.abs(q - t).sum() / 2,
                np.square(p - y).mean(),
            ]
        )
        pq.extend(p)
        pt.extend(y)
        paired.setdefault(r["parent_sequence_sha256"], {})[r["condition"]] = (p, y)
    dp, dt, delta_mse = [], [], []
    for parent in sorted(paired):
        a, b = paired[parent]["UT"], paired[parent]["TNF+IFN"]
        p, y = b[0] - a[0], b[1] - a[1]
        dp.extend(p)
        dt.extend(y)
        delta_mse.append(np.square(p - y).mean())
    mean = np.asarray(losses).mean(0)
    return {
        "n_examples": len(records),
        "n_parents": len(paired),
        "n_peptide_condition_predictions": len(pq),
        "reverse_kl": float(mean[0]),
        "forward_kl": float(mean[1]),
        "total_variation": float(mean[2]),
        "centered_log10_rmse": float(np.sqrt(mean[3])),
        "centered_log10_pearson": corr(pq, pt),
        "centered_log10_spearman": corr(pq, pt, True),
        "prediction_centered_log10_sd": float(np.std(pq)),
        "target_centered_log10_sd": float(np.std(pt)),
        "condition_change_log10_rmse": float(np.sqrt(np.mean(delta_mse))),
        "condition_change_pearson": corr(dp, dt),
        "condition_change_spearman": corr(dp, dt, True),
        "condition_change_prediction_sd": float(np.std(dp)),
        "condition_change_target_sd": float(np.std(dt)),
    }


def score(records):
    out = {}
    for split in ("train", "validation", "test"):
        selected = [r for r in records if r["split"] == split]
        out[split] = independent_metrics(selected)
        out[split + "_two"] = independent_metrics(
            [r for r in selected if len(r["peptide_ids"]) == 2]
        )
        out[split + "_at_least_three"] = independent_metrics(
            [r for r in selected if len(r["peptide_ids"]) >= 3]
        )
    return out


def compare(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif a is None:
        assert b is None
    else:
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def forward(sequence, example, weights, use_context):
    """Reconstruct features/normalization independently using valid lengths only."""
    size = len(sequence)
    context = np.asarray(example["context"] if use_context else [])
    residues = np.eye(23)[[ALPHABET.index(aa) for aa in sequence] + [22]]
    prefix = np.vstack([np.zeros((1, 22)), np.cumsum(residues[:-1, :22], axis=0)])

    def network(x, name):
        return (
            np.tanh(x @ weights[name + ".0.weight"].T + weights[name + ".0.bias"])
            @ weights[name + ".2.weight"].T
        ).ravel()

    positions = np.arange(size)
    counts = np.minimum(40, size - positions)
    start_x = np.column_stack(
        [
            residues[np.where(positions > 0, positions - 1, size)],
            residues[positions],
            positions / max(size - 1, 1),
            counts / 40,
            np.broadcast_to(context, (size, len(context))),
        ]
    )
    start_logits = network(start_x, "start_head") + np.log(counts)
    start_logp = start_logits - logsumexp(start_logits)
    needed = sorted(
        {s for c in example["candidates"] for s in c["start_positions_0based"]}
    )
    conditional = {}
    for start in needed:
        lengths = np.arange(1, min(40, size - start) + 1)
        starts = np.full(len(lengths), start)
        ends = starts + lengths
        x = np.column_stack(
            [
                residues[starts],
                residues[ends - 1],
                residues[np.where(starts > 0, starts - 1, size)],
                residues[ends],
                (prefix[ends] - prefix[starts]) / lengths[:, None],
                lengths / 40,
                starts / max(size - 1, 1),
                ends / size,
                np.broadcast_to(context, (len(lengths), len(context))),
            ]
        )
        logits = network(x, "length_head")
        conditional[start] = logits - logsumexp(logits)
    peptide = np.array(
        [
            logsumexp(
                [
                    start_logp[s] + conditional[s][len(c["sequence"]) - 1]
                    for s in c["start_positions_0based"]
                ]
            )
            for c in example["candidates"]
        ]
    )
    mass = logsumexp(peptide)
    return peptide - mass, float(mass)


def main():
    tick = time.monotonic()
    torch.set_num_threads(1)
    receipt = read(FAMILY / "reproduce/training.json")
    for name, digest in receipt["source_hashes"].items():
        assert sha(FAMILY / "reproduce/source/training" / name) == digest
    manifest = read(COHORT / "manifest.json")
    assert sha(COHORT / "manifest.json") == receipt["recipe_manifest_sha256"]
    for name, info in manifest["files"].items():
        assert sha(COHORT / name) == info["sha256"]
    parents = {
        r["parent_sequence_sha256"]: r["sequence"]
        for r in rows(COHORT / "parents.jsonl")
    }
    examples = {r["example_id"]: r for r in rows(COHORT / "examples.jsonl")}
    expected_work = {
        "start_rows": 0,
        "conditional_rows": 0,
        "selected_occurrences": 0,
        "selected_candidates": 0,
    }
    for e in examples.values():
        if e["split"] != "train":
            continue
        starts = [s for c in e["candidates"] for s in c["start_positions_0based"]]
        expected_work["start_rows"] += len(parents[e["parent_sequence_sha256"]]) * 40
        expected_work["conditional_rows"] += len(set(starts)) * 40 * 40
        expected_work["selected_occurrences"] += len(starts) * 40
        expected_work["selected_candidates"] += len(e["candidates"]) * 40
    summary = read(FAMILY / "results/training-summary.json")
    assert len(summary["runs"]) == 9
    checks, all_metrics = [], {}
    max_forward_error, max_mass_error, n_predictions = 0.0, 0.0, 0
    for run in summary["runs"]:
        name = f"{run['variant']}-seed{run['seed']}"
        directory = ARTIFACTS / name
        assert read(directory / "summary.json") == run
        history = read(directory / "history.json")
        assert [r["epoch"] for r in history] == list(range(1, 41))
        assert [r["updates"] for r in history] == [18 * i for i in range(1, 41)]
        assert run["selected_epoch"] == 1 + int(
            np.argmin([r["validation"]["reverse_kl"] for r in history])
        )
        assert run["optimizer_updates"] == 720
        expected = dict(expected_work)
        sampled = run["variant"] == "reinforce64-context"
        expected["target_terms"] = (
            144 * 40 * 64 if sampled else expected["selected_candidates"]
        )
        if sampled:
            expected["policy_draws"] = expected["target_terms"]
        assert expected == run["training_work"]
        states = {
            key: torch.load(directory / f"{key}.pt", weights_only=True)
            for key in ("initial", "selected", "last")
        }
        masks = torch.load(directory / "gradient-coverage.pt", weights_only=True)
        if sampled:
            paired = torch.load(
                ARTIFACTS / f"exact-context-seed{run['seed']}/initial.pt",
                weights_only=True,
            )
            assert all(
                torch.equal(value, paired["state_dict"][key])
                for key, value in states["initial"]["state_dict"].items()
            )
        weights = {
            key: value.numpy().astype(np.float64)
            for key, value in states["selected"]["state_dict"].items()
        }
        assert (
            len(weights) == 6
            and sum(w.size for w in weights.values()) == run["n_parameters"]
        )
        zero_columns = {}
        for key, info in run["coverage"].items():
            initial = states["initial"]["state_dict"][key]
            assert info["n_elements"] == initial.numel()
            for state, scope, prefix in (
                ("last", "final", ""),
                ("selected", "selected", "selected_"),
            ):
                assert (
                    int((states[state]["state_dict"][key] != initial).sum())
                    == info[prefix + "n_changed_elements"]
                )
                assert (
                    int(masks[scope][key].sum())
                    == info[prefix + "n_nonzero_gradient_elements"]
                )
                assert torch.isfinite(states[state]["state_dict"][key]).all()
            if key.endswith(".0.weight"):
                zero_columns[key] = torch.where(~masks["final"][key].any(0))[0].tolist()
        predictions = rows(directory / "predictions.jsonl.gz")
        assert (
            len(predictions)
            == len(examples)
            == len({r["example_id"] for r in predictions})
        )
        for row in predictions:
            e = examples[row["example_id"]]
            for key in ("parent_sequence_sha256", "component_id", "split", "condition"):
                assert row[key] == e[key]
            assert row["peptide_ids"] == [c["peptide_row_id"] for c in e["candidates"]]
            assert row["source_log10_intensities"] == [
                c["source_log10_intensity"] for c in e["candidates"]
            ]
            np.testing.assert_allclose(
                row["logt"],
                [c["target_log_probability"] for c in e["candidates"]],
                atol=0,
                rtol=0,
            )
            logq, mass = forward(
                parents[e["parent_sequence_sha256"]],
                e,
                weights,
                run["variant"] != "exact-no-context",
            )
            max_forward_error = max(
                max_forward_error, float(np.max(np.abs(logq - row["logq"])))
            )
            max_mass_error = max(
                max_mass_error, abs(mass - row["unidentified_selected_log_mass"])
            )
            np.testing.assert_allclose(logq, row["logq"], rtol=0, atol=8e-6)
            np.testing.assert_allclose(
                mass, row["unidentified_selected_log_mass"], rtol=0, atol=8e-6
            )
            np.testing.assert_allclose(np.exp(row["logq"]).sum(), 1, rtol=0, atol=2e-6)
            assert np.isfinite(row["logq"]).all()
            n_predictions += len(logq)
        recomputed = score(predictions)
        compare(recomputed, run["metrics"])
        compare(
            recomputed["validation"], history[run["selected_epoch"] - 1]["validation"]
        )
        all_metrics[name] = recomputed
        checks.append(
            {
                "run": name,
                "selected_epoch": run["selected_epoch"],
                "updates": 720,
                "untrained_input_columns": zero_columns,
            }
        )
    uniform = rows(ARTIFACTS / "uniform-predictions.jsonl.gz")
    assert len(uniform) == len(examples)
    for row in uniform:
        np.testing.assert_allclose(
            row["logq"], -math.log(len(row["peptide_ids"])), atol=1e-15
        )
        assert row["unidentified_selected_log_mass"] is None
    all_metrics["uniform"] = score(uniform)
    compare(all_metrics["uniform"], read(FAMILY / "results/uniform-metrics.json"))
    (FAMILY / "results/independent-metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    output = {
        "passed": True,
        "n_fits": len(checks),
        "n_prediction_values": n_predictions,
        "n_examples_per_fit": len(examples),
        "checks": checks,
        "max_numpy_logq_error": max_forward_error,
        "max_numpy_logmass_error": max_mass_error,
        "elapsed_seconds": time.monotonic() - tick,
        "verifier_sha256": sha(__file__),
        "artifact_hashes": {
            str(p.relative_to(ARTIFACTS)): sha(p)
            for p in sorted(ARTIFACTS.rglob("*"))
            if p.is_file()
        },
        "scope": "Independent NumPy forward, metrics, work counts, checkpoints, feature coverage and input/source hashes; not biological origin or latent-mass identification",
    }
    (FAMILY / "results/independent-results-verification.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in ("artifact_hashes", "checks")}
        )
    )


if __name__ == "__main__":
    main()
