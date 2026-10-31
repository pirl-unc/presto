"""Seal prospective gates using validation only, before test inspection."""

from pathlib import Path
import hashlib, json, time
import numpy as np

F = Path(__file__).resolve().parents[1]
VARIANTS = (
    "small-ic50",
    "wide-ic50",
    "wide-six-natural",
    "wide-six-class-balanced",
    "wide-six-primary-matched",
    "wide-six-primary-shuffled",
)
CANDIDATES = VARIANTS[2:5]


def main():
    dest = F / "results/validation-decision.json"
    if dest.exists():
        raise ValueError("validation decision already sealed")
    reports = {}
    scores = {}
    parameters = {}
    hashes = {}
    for v in VARIANTS:
        scores[v] = {}
        for seed in (42, 43):
            name = f"{v}-s{seed}"
            path = F / "results" / name / "report.json"
            report = json.loads(path.read_text())
            reports[name] = report
            m = report["common_ic50"]["validation"]["ic50"]["by_class"]
            scores[v][str(seed)] = {
                c: {k: m[c][k] for k in ("spearman", "rmse", "n")} for c in ("I", "II")
            }
            parameters[v] = report["parameters"]
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = scores["wide-ic50"]
    comparisons = {}
    eligible = []
    for v in CANDIDATES:
        x = scores[v]
        deltas = {
            c: float(
                np.mean(
                    [
                        x[str(s)][c]["spearman"] - baseline[str(s)][c]["spearman"]
                        for s in (42, 43)
                    ]
                )
            )
            for c in ("I", "II")
        }
        ratios = {
            c: float(
                np.mean([x[str(s)][c]["rmse"] for s in (42, 43)])
                / np.mean([baseline[str(s)][c]["rmse"] for s in (42, 43)])
            )
            for c in ("I", "II")
        }
        seed_deltas = {
            str(s): float(
                np.mean(
                    [
                        x[str(s)][c]["spearman"] - baseline[str(s)][c]["spearman"]
                        for c in ("I", "II")
                    ]
                )
            )
            for s in (42, 43)
        }
        gates = dict(
            macro_gain=float(np.mean(list(deltas.values()))) >= 0.01,
            no_class_rank_drop=all(d >= -0.005 for d in deltas.values()),
            no_class_rmse_regression=all(r <= 1.02 for r in ratios.values()),
            both_seeds_macro_improve=all(d > 0 for d in seed_deltas.values()),
            each_class_improves_in_some_seed=all(
                any(
                    x[str(s)][c]["spearman"] > baseline[str(s)][c]["spearman"]
                    for s in (42, 43)
                )
                for c in ("I", "II")
            ),
        )
        macro = float(
            np.mean([x[str(s)][c]["spearman"] for s in (42, 43) for c in ("I", "II")])
        )
        comparisons[v] = dict(
            class_spearman_deltas=deltas,
            class_rmse_ratios=ratios,
            seed_macro_deltas=seed_deltas,
            gates=gates,
            eligible=all(gates.values()),
            mean_macro_spearman=macro,
            parameters=parameters[v],
        )
        if all(gates.values()):
            eligible.append(v)
    preferred = "wide-ic50"
    if eligible:
        best = max(comparisons[v]["mean_macro_spearman"] for v in eligible)
        tied = [
            v for v in eligible if best - comparisons[v]["mean_macro_spearman"] <= 0.005
        ]
        preferred = min(
            tied,
            key=lambda v: (
                np.mean(
                    [
                        reports[f"{v}-s{s}"]["resources"]["wall_seconds"]
                        for s in (42, 43)
                    ]
                ),
                v,
            ),
        )
    decision = dict(
        status="sealed",
        test_used_for_decision=False,
        sealed_unix=time.time(),
        comparison="new held-out IC50 validation from prior TRAIN only; exact values; two seeds; prospective plan gates; fixed wide-ic50 comparator",
        preferred_variant=preferred,
        scores=scores,
        comparisons=comparisons,
        report_sha256=hashes,
        note="Shuffled synthetic auxiliary control and small-IC50 capacity control are never promotion candidates. A winner is specific to this new development cohort; no automatic replacement of the prior full-cohort recipe.",
    )
    dest.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
