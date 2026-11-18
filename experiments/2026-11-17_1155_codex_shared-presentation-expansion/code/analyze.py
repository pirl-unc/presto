"""Independent metrics from canonical decoded per-example validation predictions."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

E = Path(__file__).resolve().parents[1]


def records(path, model, population=None):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            r = json.loads(line)
            o = r["observation"]
            p = o["provenance"]
            ep = r["endpoint"]
            assert len(set(r["mhc_classes"])) == 1, "one class, possibly multiple MHC candidates"
            pop = population or (
                "affinity"
                if ep != "elution_observation"
                else "expanded"
                if p.get("source_family")
                else "apc"
                if r["source"] == "hitlist_native_ms"
                else "broad"
            )
            yield {
                "model": model,
                "population": pop,
                "id": r["observation_id"],
                "endpoint": ep,
                "class": r["mhc_classes"][0],
                "group": r["group_id"],
                "pair": p.get("pair_id"),
                "pmid": p.get("pmid") or p.get("study_id"),
                "condition": p.get("condition_id") or p.get("sample_id"),
                "qualifier": r["qualifier"],
                "target": r["target"],
                "prediction": r["prediction"],
                "loss": r["loss"],
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preferred")
    p.add_argument("--bootstrap-cluster", choices=["protein", "condition"], default="protein")
    p.add_argument("--bootstrap-only", action="store_true")
    args = p.parse_args()
    out = E / "results/analysis"
    out.mkdir(exist_ok=True)
    if args.bootstrap_only:
        if not args.preferred:
            p.error("--bootstrap-only requires --preferred")
        bootstrap(
            pd.read_parquet(out / "predictions.parquet"),
            args.preferred,
            out,
            args.bootstrap_cluster,
        )
        return
    rows = []
    for path in [
        *list((E / "results/fits").glob("*/raw/validation-predictions.jsonl")),
        *list((E / "results/fits").glob("*/raw/validation-predictions.jsonl.gz")),
    ]:
        if path.parents[1].name.endswith(("-n12", "-n8", "-n16")):
            continue
        rows.extend(records(path, path.parents[1].name))
    for path in [
        *list((E / "results/evaluation").glob("*/*-predictions.jsonl")),
        *list((E / "results/evaluation").glob("*/*-predictions.jsonl.gz")),
    ]:
        population = path.name.split("-")[0]
        rows.extend(
            records(path, path.parent.name, None if population == "validation" else population)
        )
    matched = E / "results/matching-pairs.json"
    if matched.exists():
        for r in json.loads(matched.read_text()):
            for side, label in [("positive", 1.0), ("negative", 0.0)]:
                for model, score in [
                    ("netmhc_ba", -np.log10(r[side + "_ba_nm"])),
                    ("netmhc_el", r[side + "_el"]),
                ]:
                    rows.append(
                        {
                            "model": model,
                            "population": "matched",
                            "id": r[side + "_id"],
                            "endpoint": "elution_observation",
                            "class": r["class"],
                            "group": r["protein"],
                            "pair": r["pair_id"],
                            "pmid": r["pmid"],
                            "condition": r["condition"],
                            "qualifier": "=",
                            "target": label,
                            "prediction": score,
                            "loss": float("nan"),
                        }
                    )
    flurry = E / "results/mhcflurry/predictions.jsonl"
    if flurry.exists():
        metadata = {
            r["id"]: r for r in rows if r["model"] == "netmhc_ba" and r["population"] == "matched"
        }
        for line in flurry.open():
            r = json.loads(line)
            rows.append(
                {
                    **metadata[r["id"]],
                    "model": "mhcflurry_" + r["mode"],
                    "prediction": r["presentation_score"],
                    "loss": float("nan"),
                }
            )
    d = pd.DataFrame(rows)
    ancestry = E / "results/fixed-validation-ancestor-exposure.json"
    if ancestry.exists():
        exposures = json.loads(ancestry.read_text())["affinity-validation.jsonl"]["exposed"]
        related = {row["id"] for row in exposures}
        clean_affinity = d[(d.population == "affinity") & ~d.id.isin(related)].copy()
        clean_affinity["population"] = "affinity_ancestry_screened"
        d = pd.concat([d, clean_affinity], ignore_index=True)
    annotation = E / "results/matching-pairs-annotated.json"
    if annotation.exists():
        retained = {
            r["pair_id"]
            for r in json.loads(annotation.read_text())
            if not r["shared_positive_core8"]
        }
        distinct = d[(d.population == "matched") & d.pair.isin(retained)].copy()
        distinct["population"] = "matched_no_shared8"
        d = pd.concat([d, distinct], ignore_index=True)
    overlap = E / "results/mhcflurry/training-overlap.json"
    if overlap.exists():
        retained = set(json.loads(overlap.read_text())["retained_pairs"])
        screened = d[
            (d.population == "matched") & (d["class"] == "I") & d.pair.isin(retained)
        ].copy()
        screened["population"] = "matched_mhcflurry_screened"
        d = pd.concat([d, screened], ignore_index=True)
    d.to_parquet(out / "predictions.parquet", index=False)
    metrics = []
    for (model, pop, ep), group in d.groupby(["model", "population", "endpoint"]):
        for cls, g in [("pooled", group), *list(group.groupby("class"))]:
            record = {
                "model": model,
                "population": pop,
                "endpoint": ep,
                "class": cls,
                "n": len(g),
                "loss": float(g.loss.mean()),
            }
            if ep == "elution_observation":
                y = g.target.to_numpy()
                s = g.prediction.to_numpy()
                record.update(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s))
                if pop in ("apc", "expanded") or pop.startswith("matched"):
                    pairs = g.pivot(index="pair", columns="target", values="prediction").dropna()
                    delta = pairs[1.0] - pairs[0.0]
                    record.update(
                        pairs=len(pairs),
                        pair_win=float(((delta > 0) + 0.5 * (delta == 0)).mean()),
                    )
            else:
                exact = g[g.qualifier == "="]
                y = exact.target.to_numpy()
                s = exact.prediction.to_numpy()
                record["exact_n"] = len(exact)
                if len(exact) > 1:
                    record.update(
                        spearman=float(spearmanr(y, s).statistic),
                        pearson=float(pearsonr(y, s).statistic),
                        rmse_log10_nm=float(np.sqrt(np.mean((s - y) ** 2))),
                    )
                    binary = y <= np.log10(500)
                    prediction = s <= np.log10(500)
                    record.update(
                        accuracy=accuracy_score(binary, prediction),
                        balanced_accuracy=balanced_accuracy_score(binary, prediction)
                        if len(set(binary)) > 1
                        else None,
                        precision=precision_score(binary, prediction, zero_division=0),
                        recall=recall_score(binary, prediction, zero_division=0),
                        f1=f1_score(binary, prediction, zero_division=0),
                        binding_auroc=roc_auc_score(binary, -s) if len(set(binary)) > 1 else None,
                        binding_auprc=average_precision_score(binary, -s) if binary.any() else None,
                    )
            metrics.append(record)
    m = pd.DataFrame(metrics)
    m.to_csv(out / "metrics.csv", index=False)
    study_metrics(d, out)
    print(
        m[
            ((m.endpoint == "ic50") | (m.endpoint == "elution_observation"))
            & (m["class"] != "pooled")
        ]
        .dropna(axis=1, how="all")
        .to_string(index=False)
    )
    if args.preferred:
        bootstrap(d, args.preferred, out, args.bootstrap_cluster)


def study_metrics(d, out):
    rows = []
    elution = d[(d.endpoint == "elution_observation") & d.pmid.notna()]
    for (model, pop, cls, pmid), g in elution.groupby(["model", "population", "class", "pmid"]):
        if len(set(g.target)) < 2:
            continue
        r = {
            "model": model,
            "population": pop,
            "class": cls,
            "pmid": pmid,
            "rows": len(g),
            "auroc": roc_auc_score(g.target, g.prediction),
            "auprc": average_precision_score(g.target, g.prediction),
            "loss": float(g.loss.mean()),
        }
        if g.pair.notna().all():
            paired = g.pivot(index="pair", columns="target", values="prediction").dropna()
            r.update(
                pairs=len(paired),
                pair_win=float(((paired[1] > paired[0]) + 0.5 * (paired[1] == paired[0])).mean()),
            )
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(out / "study-metrics.csv", index=False)
    s[s.rows >= 20].groupby(["model", "population", "class"])[
        ["auroc", "auprc", "loss", "pair_win"]
    ].mean().to_csv(out / "equal-study-means-min20rows.csv")


def weighted_auc_draws(target, scores, row_weights):
    """Exact AUROC of expanded rows using multiplicities and half-credit ties."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    y = np.asarray(target)[order]
    weights = row_weights[:, order]
    positive = np.add.reduceat(weights * y, starts, axis=1)
    negative = np.add.reduceat(weights * (1 - y), starts, axis=1)
    return (positive * (np.cumsum(negative, axis=1) - 0.5 * negative)).sum(axis=1) / (
        positive.sum(axis=1) * negative.sum(axis=1)
    )


def complete_pairs(a):
    """A retained source pair ID alone does not establish a paired cohort."""
    if a.empty or a.pair.isna().any():
        return False
    counts = a.groupby("pair").target.agg(["count", "sum", "nunique"])
    return bool(((counts["count"] == 2) & (counts["sum"] == 1) & (counts["nunique"] == 2)).all())


def protein_groups_available(a):
    return bool(
        a.group.notna().all() and not a.group.astype(str).str.startswith("study-component:").any()
    )


def paired_draws(a, cluster_draws, cluster_names):
    """Match the original paired cluster bootstrap without expanding rows."""
    lookup = {name: i for i, name in enumerate(cluster_names)}
    counts = np.stack([np.bincount(draw, minlength=len(cluster_names)) for draw in cluster_draws])
    row_weights = counts[:, [lookup[v] for v in a.bootstrap_group]]
    auc_delta = weighted_auc_draws(a.target, a.prediction_real, row_weights) - weighted_auc_draws(
        a.target, a.prediction_other, row_weights
    )
    if not complete_pairs(a):
        return np.column_stack([auc_delta, np.full(len(auc_delta), np.nan)])
    pairs = a.pivot(
        index="pair", columns="target", values=["prediction_real", "prediction_other"]
    ).dropna()
    groups = a.pivot(index="pair", columns="target", values="bootstrap_group").loc[pairs.index]
    assert (groups[0.0] == groups[1.0]).all()
    pair_weights = counts[:, [lookup[v] for v in groups[1.0]]]
    wins = {
        k: ((pairs[k][1.0] > pairs[k][0.0]) + 0.5 * (pairs[k][1.0] == pairs[k][0.0])).to_numpy()
        for k in ("prediction_real", "prediction_other")
    }
    pair_delta = (pair_weights * (wins["prediction_real"] - wins["prediction_other"])).sum(
        axis=1
    ) / pair_weights.sum(axis=1)
    return np.column_stack([auc_delta, pair_delta])


def bootstrap(d, preferred, out, cluster):
    result = []
    skipped = []
    for pop in (
        "apc",
        "broad",
        "expanded",
        "matched",
        "matched_no_shared8",
        "matched_mhcflurry_screened",
    ):
        sub = d[d.population == pop]
        for cls in ("pooled", "I", "II"):
            g = sub if cls == "pooled" else sub[sub["class"] == cls]
            real = g[g.model == preferred]
            if real.empty:
                continue
            if cluster == "protein" and not protein_groups_available(real):
                skipped.append(
                    {
                        "population": pop,
                        "class": cls,
                        "reason": (
                            "split groups are study components or missing, not protein families"
                        ),
                    }
                )
                continue
            for comparator, other in g.groupby("model"):
                if comparator == preferred:
                    continue
                a = real.merge(
                    other[["id", "prediction"]],
                    on="id",
                    suffixes=("_real", "_other"),
                    validate="one_to_one",
                )
                if len(a) != len(real):
                    continue
                a["bootstrap_group"] = (
                    a["group"]
                    if cluster == "protein"
                    else a.pmid.fillna("<unknown-study>").astype(str)
                    + "|"
                    + a.condition.fillna("<unknown-condition>").astype(str)
                )
                cluster_names = sorted(a.bootstrap_group.unique())
                clusters = [v.index.to_numpy() for _, v in a.groupby("bootstrap_group")]
                paired = complete_pairs(a)
                if paired:
                    assert a.pair.notna().all()
                    pairs = a.pivot(
                        index="pair",
                        columns="target",
                        values=["prediction_real", "prediction_other"],
                    ).dropna()
                    wins = {
                        k: ((pairs[k][1] > pairs[k][0]) + 0.5 * (pairs[k][1] == pairs[k][0]))
                        for k in ("prediction_real", "prediction_other")
                    }
                if len(clusters) < 2:
                    skipped.append(
                        {
                            "population": pop,
                            "class": cls,
                            "comparator": comparator,
                            "reason": "fewer than two clusters",
                        }
                    )
                    continue
                # Stable when additional models are collected; comparisons on the
                # same examples use exactly the same cluster multiplicities.
                key = json.dumps([20261116, pop, cls, cluster, cluster_names])
                rng = np.random.default_rng(
                    int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                )
                draws = rng.integers(len(clusters), size=(1000, len(clusters)))
                samples = np.concatenate(
                    [
                        paired_draws(a, draws[i : i + 32], cluster_names)
                        for i in range(0, len(draws), 32)
                    ]
                )
                for j, metric in enumerate(("auroc", "pair_win")):
                    if j == 1 and not paired:
                        continue
                    finite = np.isfinite(samples[:, j])
                    assert finite.sum() >= 950, (pop, cls, comparator, metric, finite.sum())
                    delta = (
                        (
                            roc_auc_score(a.target, a.prediction_real)
                            - roc_auc_score(a.target, a.prediction_other)
                        )
                        if j == 0
                        else float((wins["prediction_real"] - wins["prediction_other"]).mean())
                    )
                    result.append(
                        {
                            "population": pop,
                            "class": cls,
                            "model": preferred,
                            "comparator": comparator,
                            "metric": metric,
                            "difference": delta,
                            "ci95": np.quantile(samples[finite, j], [0.025, 0.975]).tolist(),
                            "clusters": len(clusters),
                            "finite_bootstrap_draws": int(finite.sum()),
                            "complete_pair_cohort": paired,
                            "bootstrap": (
                                f"1000 paired {cluster} clusters, conditional on selected "
                                "fitted models and development cohort"
                            ),
                        }
                    )
    (
        out
        / ("paired-bootstrap.json" if cluster == "protein" else "paired-bootstrap-condition.json")
    ).write_text(json.dumps(result, indent=2))
    (out / f"paired-bootstrap-{preferred}-{cluster}.json").write_text(json.dumps(result, indent=2))
    (out / f"bootstrap-exclusions-{preferred}-{cluster}.json").write_text(
        json.dumps(skipped, indent=2)
    )


if __name__ == "__main__":
    main()
