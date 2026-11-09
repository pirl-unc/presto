"""Independently reconstruct teacher, observations and all reported metrics."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import reference_metrics as reference

FAMILY = Path(__file__).resolve().parents[1]
RUN = FAMILY / "results/run"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    lp, lq = np.log(np.maximum(p, 1e-30)), np.log(np.maximum(q, 1e-30))
    return {
        "tv": float(np.mean(np.sum(abs(p - q), axis=1)) / 2),
        "forward_kl": float(np.mean(np.sum(p * (lp - lq), axis=1))),
        "reverse_kl": float(np.mean(np.sum(q * (lq - lp), axis=1))),
        "rmse_probability": float(np.sqrt(np.mean(np.square(p - q)))),
        "pearson_probability": reference.corr(p.ravel(), q.ravel()),
        "spearman_probability": reference.corr(rankdata(p.ravel()), rankdata(q.ravel())),
        "entropy": float(np.mean(np.sum(-q * lq, axis=1))),
        "target_entropy": float(np.mean(np.sum(-p * lp, axis=1))),
        "predicted_probability_sd": float(np.std(q)),
        "target_probability_sd": float(np.std(p)),
    }


def metrics(source, observations, prediction, config):
    p = np.array([r["target"] for r in source])
    q = np.asarray(prediction, dtype=float)
    weights = np.array([r["weights"] for r in observations])
    frequencies = (
        np.array([r[f"counts{config['budget']}"] for r in observations]) / config["budget"]
    )
    assert np.isfinite(q).all() and np.all(q >= 0)
    np.testing.assert_allclose(q.sum(1), 1, atol=4e-7, rtol=0)
    observed_p, observed_q = p * weights, q * weights
    observed_p /= observed_p.sum(1)[:, None]
    observed_q /= observed_q.sum(1)[:, None]
    ce = float(-np.mean(np.sum(frequencies * np.log(np.maximum(observed_q, 1e-30)), axis=1)))
    l2 = float(np.mean(np.sum(np.square(observed_q - frequencies), axis=1)))
    objective = ce if config["method"] == "ce" else l2
    if config["method"] == "naive":
        objective += float(np.mean(1 - np.sum(observed_q**2, axis=1)) / config["draws"])
    return {
        "rows": len(source),
        "observed_count_ce": ce,
        "observed_count_l2": l2,
        "expected_training_cost": objective,
        "latent": distribution(p, q),
        "observed": distribution(observed_p, observed_q),
        "invisible_mass_mae": float(np.mean(abs(np.sum((p - q) * (weights == 0), axis=1)))),
        "invisible_target_mass": float(np.mean(np.sum(p * (weights == 0), axis=1))),
        "invisible_predicted_mass": float(np.mean(np.sum(q * (weights == 0), axis=1))),
    }


def compare(expected, actual):
    for key, value in actual.items():
        if isinstance(value, dict):
            compare(expected[key], value)
        elif value is None:
            assert expected[key] is None
        else:
            np.testing.assert_allclose(expected[key], value, atol=3e-6, rtol=3e-6, err_msg=key)


def main():
    manifest, launch = read(RUN / "artifacts.json"), read(FAMILY / "reproduce/launch.json")
    for name, digest in manifest.items():
        assert sha(RUN / name) == digest, name
    assert sha(FAMILY.parents[1] / launch["plan"]) == launch["plan_sha256"]
    for name, digest in launch["sources"].items():
        assert sha(FAMILY / "code" / name) == sha(FAMILY / "reproduce/source" / name) == digest
    source, teacher = rows(RUN / "oracle-dataset.jsonl"), read(RUN / "teacher.json")
    assert len(source) == 576 and len({r["id"] for r in source}) == 576
    parents = {
        split: {tuple(r["tokens"]) for r in source if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert [len(parents[s]) for s in parents] == [96, 24, 24]
    assert len(set.union(*parents.values())) == 144
    coordinates = [reference.reconstruct(row, teacher) for row in source]
    observations = {}
    for mechanism, seed in (("positive", 20261023), ("blind", 20261024)):
        observations[mechanism] = rows(RUN / f"observations-{mechanism}.jsonl")
        rng = np.random.default_rng(seed)
        for row, coords, obs in zip(source, coordinates, observations[mechanism]):
            assert row["id"] == obs["id"]
            weight = np.array(
                [0.05 + 0.95 * (row["tokens"][start + size - 1] / 7) ** 3 for start, size in coords]
            )
            if mechanism == "blind":
                weight *= row["compatible"]
            np.testing.assert_allclose(weight, obs["weights"], rtol=1e-14, atol=1e-14)
            probability = np.array(row["target"]) * weight
            probability /= probability.sum()
            draws = rng.choice(70, size=256, p=probability)
            np.testing.assert_array_equal(draws, obs["draws"])
            for budget in (32, 256):
                np.testing.assert_array_equal(
                    np.bincount(draws[:budget], minlength=70), obs[f"counts{budget}"]
                )
    summary = read(RUN / "summary.json")
    assert len(summary["conditions"]) == 60 and not summary["failures"]
    expected = {
        (m, n, method, seed)
        for m in launch["mechanisms"]
        for n in launch["counts"]
        for method in launch["methods"]
        for seed in launch["seeds"]
    }
    actual = {
        (r["config"]["mechanism"], r["config"]["budget"], r["config"]["method"], r["seed"])
        for r in summary["conditions"]
    }
    assert actual == expected
    metric_sets, table = 0, []
    for report in summary["conditions"]:
        config, seed = report["config"], report["seed"]
        path = RUN / f"{config['name']}-seed{seed}"
        assert read(path / "summary.json") == report
        assert all(report["gradient_seen"].values()) and len(report["gradient_seen"]) == 6
        assert all(change > 0 for change in report["parameter_max_change"].values())
        assert report["updates"] == 1440
        assert (
            report["fit_dense_span_scores"]
            == report["fit_observed_count_cells_accessed"]
            == 1612800
        )
        expected_draws = 0 if config["method"] in ("ce", "l2") else 1440 * 16 * 4 * config["draws"]
        assert report["sampled_draws"] == expected_draws
        history = read(path / "history.json")
        assert [r["step"] for r in history] == list(range(120, 1441, 120))
        selected = min(history, key=lambda r: r["validation_observed_count_ce"])
        assert selected["step"] == report["selected_step"]
        predictions = np.load(path / "predictions.npz")
        for split in ("train", "validation", "test"):
            subset = [r for r in source if r["split"] == split]
            obs = [
                r
                for r, original in zip(observations[config["mechanism"]], source)
                if original["split"] == split
            ]
            q = predictions[split]
            assert q.shape == (len(subset), 70)
            result = metrics(subset, obs, q, config)
            compare(report["metrics"][split], result)
            metric_sets += 1
            if split == "validation":
                np.testing.assert_allclose(
                    result["observed_count_ce"], selected["validation_observed_count_ce"], atol=2e-6
                )
            for context in range(4):
                mask = np.array([r["context_id"] == context for r in subset])
                contextual = metrics(
                    [r for r, keep in zip(subset, mask) if keep],
                    [r for r, keep in zip(obs, mask) if keep],
                    q[mask],
                    config,
                )
                compare(report["metrics"][split]["contexts"][str(context)], contextual)
                metric_sets += 1
            table.append(
                {
                    "condition": config["name"],
                    "seed": seed,
                    "split": split,
                    "selected_step": report["selected_step"],
                    "seconds": report["seconds_including_validation"],
                    "draws": report["sampled_draws"],
                    **{k: v for k, v in result.items() if not isinstance(v, dict)},
                    **{
                        f"{kind}_{k}": v
                        for kind in ("latent", "observed")
                        for k, v in result[kind].items()
                    },
                }
            )
    for key, splits in summary["controls"].items():
        mechanism, budget = key.split("-n")
        for split, controls in splits.items():
            subset = [r for r in source if r["split"] == split]
            obs = [
                r
                for r, original in zip(observations[mechanism], source)
                if original["split"] == split
            ]
            for label, recorded in controls.items():
                q = (
                    np.full((len(subset), 70), 1 / 70)
                    if label == "uniform"
                    else np.array([r["target"] for r in subset])
                )
                compare(recorded, metrics(subset, obs, q, {"budget": int(budget), "method": "ce"}))
                metric_sets += 1
    for seed in launch["seeds"]:
        assert len({r["initial_hash"] for r in summary["conditions"] if r["seed"] == seed}) == 1
    with (FAMILY / "results/metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    means = []
    for name in sorted({r["condition"] for r in table}):
        selected = [r for r in table if r["condition"] == name and r["split"] == "test"]
        means.append(
            {
                "condition": name,
                **{
                    k: float(np.mean([r[k] for r in selected]))
                    for k in selected[0]
                    if k not in ("condition", "split", "seed")
                },
            }
        )
    reference.dump(FAMILY / "results/condition-means.json", means)
    receipt = {
        "status": "passed",
        "artifact_hashes": len(manifest),
        "teacher_rows": len(source),
        "observation_rows": 2 * len(source),
        "recomputed_metric_sets": metric_sets,
        "fits": 60,
        "trainer_imported": False,
        "test_model_rescored": False,
    }
    reference.dump(FAMILY / "results/independent-verification.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
