"""Training identifiability and deterministic reporting of analytic rank ties."""

import copy
import gzip
import json
import math
from collections import Counter

import numpy as np
from scipy.stats import pearsonr, spearmanr

from verify import AA, FAMILY, feature, forward, read_groups


def canonical(theta, g, variant, a, b, t, ref):
    if variant == "constant":
        rate = math.exp(theta[0])
        boundaries = int(a > 0) + int(b < len(g["sequence"]))
        return (
            boundaries
            * (math.log(-math.expm1(-rate * t)) - math.log(-math.expm1(-rate * ref)))
            - (t - ref) * (b - a - 1) * rate
        ) / math.log(10)
    seq = g["sequence"]
    context = float(g["x"][0, -1]) if variant == "sequence_context" else 0.0

    def rate(pair):
        terms = [theta[0]]
        for offset, aa in ((0, pair[0]), (19, pair[1])):
            idx = AA.index(aa)
            if idx:
                terms.append(theta[offset + idx])
        if variant == "sequence_context":
            terms.append(theta[-1] * context)
        return math.exp(math.fsum(terms))

    boundaries = []
    if a > 0:
        boundaries.append(seq[a - 1 : a + 1])
    if b < len(seq):
        boundaries.append(seq[b - 1 : b + 1])
    cut_terms = [
        math.log(-math.expm1(-rate(p) * t)) - math.log(-math.expm1(-rate(p) * ref))
        for p in sorted(boundaries)
    ]
    counts = Counter(seq[i : i + 2] for i in range(a, b - 1))
    survival = (t - ref) * math.fsum(
        count * rate(p) for p, count in sorted(counts.items())
    )
    return (math.fsum(cut_terms) - survival) / math.log(10)


def main():
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    groups = read_groups(cfg)
    fits = json.loads((FAMILY / "results/fits.json").read_text())
    raw = json.loads((FAMILY / "results/metrics.json").read_text())
    reported = copy.deepcopy(raw)
    rank_changes = []
    maximum = 0.0
    maps = {}
    for variant in cfg["variants"]:
        theta = np.array(fits[variant]["coefficients"])
        for g in groups:
            for curve, t, ref, signal, reference in g["rows"]:
                maps[variant, curve["curve_id"], t] = canonical(
                    theta,
                    g,
                    variant,
                    curve["start_0based"],
                    curve["end_0based"],
                    t,
                    ref,
                )
    collected = {}
    with (
        gzip.open(FAMILY / "results/predictions.jsonl.gz", "rt") as src,
        gzip.open(FAMILY / "results/canonical-rank-predictions.jsonl.gz", "wt") as dest,
    ):
        for line in src:
            r = json.loads(line)
            p = maps.get((r["variant"], r["curve_id"], r["time"]), r["prediction"])
            maximum = max(maximum, abs(p - r["prediction"]))
            collected.setdefault((r["variant"], r["split"]), []).append(
                (p, r["target"])
            )
            dest.write(
                json.dumps(
                    dict(
                        variant=r["variant"],
                        curve_id=r["curve_id"],
                        time=r["time"],
                        prediction=p,
                        raw_difference=p - r["prediction"],
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    assert maximum < 1e-10
    for (variant, split), values in collected.items():
        p, y = np.array(values).T
        if variant == "no_change":
            continue
        value = float(spearmanr(p, y).statistic)
        reported[variant][split]["spearman"] = value
        rank_changes.append(
            dict(
                variant=variant,
                split=split,
                raw_spearman=raw[variant][split]["spearman"],
                canonical_spearman=value,
            )
        )
    (FAMILY / "results/reported-metrics.json").write_text(
        json.dumps(reported, indent=2, sort_keys=True) + "\n"
    )
    (FAMILY / "results/rank-reporting.json").write_text(
        json.dumps(
            dict(
                maximum_prediction_difference=maximum,
                changes=rank_changes,
                only_reported_spearman_changed=True,
                raw_fits_predictions_metrics_retained=True,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    diagnostics = {}
    for variant in cfg["variants"]:
        theta = np.array(fits[variant]["coefficients"])
        information = np.zeros((len(theta), len(theta)))
        rates_report = {}
        for g in groups:
            if g["split"] != "train":
                continue
            x = feature(g, variant)
            rates = np.exp(x @ theta)
            a, b, t, ref = g["a"], g["b"], g["t"], g["ref"]
            n = len(rates)
            prefix = np.vstack(
                (np.zeros((1, len(theta))), np.cumsum(rates[:, None] * x, axis=0))
            )
            jac = -(t - ref)[:, None] * (prefix[b - 1] - prefix[a])
            for idx, mask in ((a - 1, a > 0), (b - 1, b <= n)):
                safe = np.clip(idx, 0, n - 1)
                ht = rates[safe] * t
                hr = rates[safe] * ref
                d = np.where(
                    mask,
                    ht * np.exp(-ht) / (-np.expm1(-ht))
                    - hr * np.exp(-hr) / (-np.expm1(-hr)),
                    0.0,
                )
                jac += d[:, None] * x[safe]
            jac /= math.log(10)
            information += jac.T @ (g["w"][:, None] * jac)
            rates_report[g["condition"]] = dict(
                minimum=float(rates.min()),
                median=float(np.median(rates)),
                maximum=float(rates.max()),
            )
        eigen = np.linalg.eigvalsh(information)
        assert eigen.min() > -1e-10
        profiles = []
        for delta in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
            altered = theta.copy()
            altered[0] += delta
            mse = sum(
                float(g["w"] @ (forward(altered, g, feature(g, variant)) - g["y"]) ** 2)
                for g in groups
                if g["split"] == "train"
            )
            profiles.append(dict(intercept_shift=delta, training_mse=mse))
        diagnostics[variant] = dict(
            unregularized_weighted_JtJ_eigenvalues=eigen.tolist(),
            numerical_rank_relative_1e10=int((eigen > eigen.max() * 1e-10).sum()),
            condition_number=float(eigen.max() / eigen.min()),
            rate_per_hour_by_condition=rates_report,
            training_intercept_profile=profiles,
            statistical_confidence_interpretation=False,
        )
    (FAMILY / "results/training-identifiability.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    print(
        "Canonical rank correction maximum prediction difference", maximum, flush=True
    )
    print(
        {
            k: (v["numerical_rank_relative_1e10"], v["condition_number"])
            for k, v in diagnostics.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
