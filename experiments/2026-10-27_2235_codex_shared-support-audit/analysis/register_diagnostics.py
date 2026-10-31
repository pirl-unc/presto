"""Posthoc uncertainty of statistical register weights; no register truth is available."""

from pathlib import Path
import gzip, json, math
import numpy as np

F = Path(__file__).resolve().parents[1]
PRIOR = F.parent / "2026-10-27_2100_codex_transfer-improvements"


def load(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def main():
    summaries = []
    paired = []
    records = []
    for variant in ("interaction-natural-joint", "interaction-II", "interaction-I"):
        for split in ("validation", "test"):
            seeds = {}
            for seed in (42, 43):
                prefix = (
                    "identical_weights_expanded_routing-"
                    if variant == "interaction-I"
                    else ""
                )
                rows = [
                    r
                    for r in load(
                        PRIOR
                        / "results"
                        / f"{variant}-s{seed}"
                        / f"{prefix}{split}-predictions.jsonl.gz"
                    )
                    if r["mhc_classes"] == ["II"]
                ]
                values = []
                for r in rows:
                    reg = r["diagnostics"]["candidates"][0]["registers"][0]
                    w = np.asarray(reg["weights"])
                    starts = reg["starts"]
                    assert len(w) == len(starts) and abs(w.sum() - 1) < 1e-5
                    if len(w) <= 1:
                        continue
                    entropy = float(-(w * np.log(w.clip(1e-30))).sum())
                    np.testing.assert_allclose(entropy, reg["entropy"], atol=2e-6)
                    values.append(
                        dict(
                            condition=f"{variant}-s{seed}",
                            split=split,
                            observation_id=r["observation_id"],
                            n_registers=len(w),
                            normalized_entropy=entropy / math.log(len(w)),
                            maximum_weight=float(w.max()),
                            maximum_vs_uniform=float(w.max() * len(w)),
                            top_start=starts[int(w.argmax())],
                            target=r["target"],
                            prediction=r["prediction"],
                            qualifier=r["qualifier"],
                        )
                    )
                seeds[seed] = {r["observation_id"]: r for r in values}
                records.extend(values)
                summaries.append(
                    dict(
                        variant=variant,
                        seed=seed,
                        split=split,
                        n_multiregister=len(values),
                        mean_normalized_entropy=float(
                            np.mean([r["normalized_entropy"] for r in values])
                        ),
                        median_maximum_weight=float(
                            np.median([r["maximum_weight"] for r in values])
                        ),
                        median_maximum_vs_uniform=float(
                            np.median([r["maximum_vs_uniform"] for r in values])
                        ),
                        fraction_maximum_above_half=float(
                            np.mean([r["maximum_weight"] > 0.5 for r in values])
                        ),
                    )
                )
            assert set(seeds[42]) == set(seeds[43])
            ids = list(seeds[42])
            paired.append(
                dict(
                    variant=variant,
                    split=split,
                    n=len(ids),
                    top_start_agreement=float(
                        np.mean(
                            [
                                seeds[42][k]["top_start"] == seeds[43][k]["top_start"]
                                for k in ids
                            ]
                        )
                    ),
                    uniform_random_agreement_reference=float(
                        np.mean([1 / seeds[42][k]["n_registers"] for k in ids])
                    ),
                )
            )
    result = dict(
        scope="posthoc saved-output audit; no new inference; no measured register assignments; weights are statistical, not physical occupancies",
        summary=summaries,
        paired_seed_agreement=paired,
        records=records,
    )
    (F / "results/register-diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(dict(summary=summaries, paired=paired), indent=2))


if __name__ == "__main__":
    main()
