"""Close the frozen comparison using independently verified saved predictions."""

from collections import defaultdict
import json
import math
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from verify import F, PINS, UPLOAD, csv_write, load, read, write

VARIANTS = ("ic50-exact", "ic50-bounds", "multi-assay", "multi-assay-wide")
LABELS = ("IC50 exact", "+ IC50 bounds", "Six outputs", "Six outputs, wider")


def average(values):
    values = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def regression(rows):
    exact = [r for r in rows if r["qualifier"] == "="]
    result = dict(n_all=len(rows), n=len(exact))
    if not exact:
        return result
    y = np.array([r["target"] for r in exact])
    p = np.array([r["prediction"] for r in exact])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = (
            float(spearmanr(y, p).statistic)
            if len(y) > 1 and np.ptp(y) and np.ptp(p)
            else None
        )
    result.update(
        spearman=rho,
        rmse=float(np.sqrt(np.mean((p - y) ** 2))),
        mean_error=float(np.mean(p - y)),
        prediction_sd=float(np.std(p)),
        target_sd=float(np.std(y)),
    )
    return result


def training_domains(condition):
    peptides, grooves, studies, cores = set(), set(), set(), set()
    counts = defaultdict(int)
    for spec in PINS["manifests"][condition]["sources"]:
        with (UPLOAD / "inputs" / spec["path"]).open() as stream:
            for line in stream:
                r = json.loads(line)
                if r["split"] != "train":
                    continue
                m = r["mhc"][0]
                peptide = r["peptide"]
                peptides.add(peptide)
                grooves.add((m["groove1"], m["groove2"]))
                study = r["provenance"]["raw_record"].get("reference_iri")
                if study:
                    studies.add(study)
                cores.update(peptide[i : i + 9] for i in range(len(peptide) - 8))
                counts[(r["endpoint"], m["mhc_class"], r["qualifier"])] += 1
    return peptides, grooves, studies, cores, counts


def length_bin(n):
    for upper, label in (
        (8, "7–8"),
        (9, "9"),
        (10, "10"),
        (14, "11–14"),
        (20, "15–20"),
    ):
        if n <= upper:
            return label
    return ">20"


def failure_audit():
    groups, registers, register_agreements, support = [], [], [], []
    for variant in VARIANTS:
        train_peptides, train_grooves, train_studies, train_cores, train_counts = (
            training_domains(f"{variant}-s42")
        )
        register_by_seed = {}
        for seed in (42, 43):
            name = f"{variant}-s{seed}"
            directory = F / "results" / name
            report = read(directory / "report.json")
            resolved = read(directory / "resolved-config.json")
            verified = read(F / "results" / f"{name}-independent-verification.json")
            for endpoint in resolved["model"]["endpoints"]:
                for cls in ("I", "II"):
                    vals = verified["metrics"]["validation"]["endpoints"][endpoint][
                        "by_class"
                    ][cls]
                    tests = verified["metrics"]["test"]["endpoints"][endpoint][
                        "by_class"
                    ][cls]
                    support.append(
                        dict(
                            condition=name,
                            endpoint=endpoint,
                            mhc_class=cls,
                            train_rows=sum(
                                n
                                for (e, c, q), n in train_counts.items()
                                if (e, c) == (endpoint, cls)
                            ),
                            exact_train_rows=train_counts.get((endpoint, cls, "="), 0),
                            optimizer_label_exposures=report[
                                "endpoint_class_examples"
                            ].get(endpoint + ":" + cls, 0),
                            validation_rows=vals["n_all"],
                            validation_exact=vals["n"],
                            test_rows=tests["n_all"],
                            test_exact=tests["n"],
                            scope="missing validation"
                            if not vals["n_all"]
                            else "sparse validation (<100 exact)"
                            if vals["n"] < 100
                            else "development holdout only",
                        )
                    )
            for split in ("validation", "test"):
                rows = load(directory / f"{split}-predictions.jsonl.gz")
                subsets = defaultdict(list)
                regs = defaultdict(list)
                tops = {}
                for r in rows:
                    assert r["peptide"] not in train_peptides
                    mhc = r["observation"]["mhc"][0]
                    endpoint, cls = r["endpoint"], mhc["mhc_class"]
                    study = r["observation"]["provenance"]["raw_record"].get(
                        "reference_iri"
                    )
                    peptide = r["peptide"]
                    domains = {
                        "length": length_bin(len(peptide)),
                        "target_log10_bin": str(math.floor(r["target"])),
                        "groove_seen": str(
                            (mhc["groove1"], mhc["groove2"]) in train_grooves
                        ),
                        "study_seen": "missing"
                        if not study
                        else str(study in train_studies),
                        "any_9mer_seen": str(
                            any(
                                peptide[i : i + 9] in train_cores
                                for i in range(len(peptide) - 8)
                            )
                        ),
                        "qualifier": r["qualifier"],
                    }
                    for kind, value in domains.items():
                        subsets[(endpoint, cls, kind, value)].append(r)
                    if cls == "II":
                        reg = r["diagnostics"]["candidates"][0]["registers"][0]
                        w = np.asarray(reg["weights"])
                        if len(w) > 1:
                            entropy = float(
                                -np.sum(w * np.log(w.clip(1e-30))) / np.log(len(w))
                            )
                            regs[endpoint].append(
                                (entropy, float(w.max()), float(w.max() * len(w)))
                            )
                            tops[r["observation_id"]] = (
                                endpoint,
                                reg["starts"][int(w.argmax())],
                                len(w),
                            )
                register_by_seed[(seed, split)] = tops
                for (endpoint, cls, kind, value), rr in subsets.items():
                    m = regression(rr)
                    if m["n_all"] >= 10:
                        groups.append(
                            dict(
                                condition=name,
                                split=split,
                                endpoint=endpoint,
                                mhc_class=cls,
                                stratum=kind,
                                group=value,
                                **m,
                            )
                        )
                for endpoint, values in regs.items():
                    a = np.asarray(values)
                    registers.append(
                        dict(
                            condition=name,
                            split=split,
                            endpoint=endpoint,
                            n=len(a),
                            mean_normalized_entropy=float(a[:, 0].mean()),
                            median_maximum_weight=float(np.median(a[:, 1])),
                            median_maximum_vs_uniform=float(np.median(a[:, 2])),
                            fraction_maximum_above_half=float((a[:, 1] > 0.5).mean()),
                        )
                    )
        for split in ("validation", "test"):
            a, b = register_by_seed[(42, split)], register_by_seed[(43, split)]
            assert set(a) == set(b)
            for endpoint in sorted({v[0] for v in a.values()}):
                ids = [k for k, v in a.items() if v[0] == endpoint]
                register_agreements.append(
                    dict(
                        variant=variant,
                        split=split,
                        endpoint=endpoint,
                        n=len(ids),
                        top_start_agreement=float(
                            np.mean([a[k][1] == b[k][1] for k in ids])
                        ),
                        uniform_random_reference=float(
                            np.mean([1 / a[k][2] for k in ids])
                        ),
                    )
                )
    csv_write(F / "results/failure-strata.csv", groups)
    csv_write(F / "results/output-class-support.csv", support)
    write(
        F / "results/register-diagnostics.json",
        dict(
            summary=registers,
            agreement=register_agreements,
            scope="Saved statistical weights only; no measured register truth or physical occupancy claim.",
        ),
    )
    return groups


def paired_bootstrap():
    """Same peptide clusters for paired models, mean effect over the two seeds."""
    result = []
    rng = np.random.default_rng(20261027)
    draws = 1000
    for split in ("validation", "test"):
        predictions = {}
        for v in VARIANTS:
            for seed in (42, 43):
                rr = load(
                    F
                    / "results"
                    / f"{v}-s{seed}"
                    / f"common-ic50-{split}-predictions.jsonl.gz"
                )
                predictions[(v, seed)] = {
                    r["observation_id"]: r for r in rr if r["qualifier"] == "="
                }
        ref = predictions[(VARIANTS[0], 42)]
        assert all(set(x) == set(ref) for x in predictions.values())
        for cls in ("I", "II"):
            ids = sorted(k for k, r in ref.items() if r["mhc_classes"] == [cls])
            y = np.array([ref[k]["target"] for k in ids])
            clusters = defaultdict(list)
            for i, k in enumerate(ids):
                clusters[ref[k]["peptide"]].append(i)
            members = list(clusters.values())
            p = {
                key: np.array([rr[k]["prediction"] for k in ids])
                for key, rr in predictions.items()
            }
            differences = {v: [] for v in VARIANTS[1:]}
            for _ in range(draws):
                indices = np.concatenate(
                    [members[i] for i in rng.integers(0, len(members), len(members))]
                )
                yy = y[indices]
                base = [
                    (
                        float(spearmanr(yy, p[(VARIANTS[0], s)][indices]).statistic),
                        float(
                            np.sqrt(np.mean((yy - p[(VARIANTS[0], s)][indices]) ** 2))
                        ),
                    )
                    for s in (42, 43)
                ]
                for v in VARIANTS[1:]:
                    scores = [
                        (
                            float(spearmanr(yy, p[(v, s)][indices]).statistic),
                            float(np.sqrt(np.mean((yy - p[(v, s)][indices]) ** 2))),
                        )
                        for s in (42, 43)
                    ]
                    differences[v].append(
                        np.mean(np.asarray(scores) - np.asarray(base), axis=0)
                    )
            for v, d in differences.items():
                a = np.asarray(d)
                result.append(
                    dict(
                        variant=v,
                        baseline=VARIANTS[0],
                        split=split,
                        mhc_class=cls,
                        exact_rows=len(ids),
                        peptide_clusters=len(members),
                        bootstrap_draws=draws,
                        spearman_delta_95_percentile=np.quantile(
                            a[:, 0], [0.025, 0.975]
                        ).tolist(),
                        rmse_delta_95_percentile=np.quantile(
                            a[:, 1], [0.025, 0.975]
                        ).tolist(),
                    )
                )
    write(
        F / "results/paired-bootstrap.json",
        dict(
            results=result,
            scope="Paired peptide-cluster bootstrap of mean seed effects, conditional on existing development splits; not independent studies or uncertainty over new training seeds.",
        ),
    )


def main():
    assert (
        read(F / "results/validation-decision.json")["test_used_for_decision"] is False
    )
    assert read(F / "results/verification-summary.json")["status"] == "passed"
    reports = {
        n: read(F / "results" / f"{n}-independent-verification.json")
        for n in PINS["conditions"]
    }
    raw = {n: read(F / "results" / n / "report.json") for n in PINS["conditions"]}
    primary = []
    outputs = []
    for v in VARIANTS:
        for split in ("validation", "test"):
            for cls in ("I", "II"):
                rr = [reports[f"{v}-s{s}"]["common_ic50"][split][cls] for s in (42, 43)]
                primary.append(
                    dict(
                        variant=v,
                        split=split,
                        mhc_class=cls,
                        n=rr[0]["n"],
                        **{
                            k: average([r.get(k) for r in rr])
                            for k in (
                                "spearman",
                                "pearson",
                                "rmse",
                                "loss",
                                "censor_satisfaction",
                                "prediction_sd",
                                "target_sd",
                            )
                        },
                        **{
                            "binding_" + k: average(
                                [r["binding_at_500_nM"].get(k) for r in rr]
                            )
                            for k in (
                                "accuracy",
                                "balanced_accuracy",
                                "precision",
                                "recall",
                                "f1",
                                "auroc",
                                "auprc",
                            )
                        },
                    )
                )
                endpoints = reports[f"{v}-s42"]["metrics"][split]["endpoints"]
                for endpoint in endpoints:
                    ee = [
                        reports[f"{v}-s{s}"]["metrics"][split]["endpoints"][endpoint]
                        for s in (42, 43)
                    ]
                    mm = [e["by_class"][cls] for e in ee]
                    outputs.append(
                        dict(
                            variant=v,
                            split=split,
                            endpoint=endpoint,
                            mhc_class=cls,
                            n=mm[0]["n"],
                            n_all=mm[0]["n_all"],
                            **{
                                k: average([m.get(k) for m in mm])
                                for k in (
                                    "spearman",
                                    "pearson",
                                    "rmse",
                                    "loss",
                                    "censor_satisfaction",
                                )
                            },
                            training_mean_rmse=average(
                                [
                                    e["mean_baseline_by_class"][cls].get("rmse")
                                    for e in ee
                                ]
                            ),
                        )
                    )
    csv_write(F / "results/common-ic50-means.csv", primary)
    csv_write(F / "results/all-output-means.csv", outputs)
    selection = read(F / "results/validation-decision.json")
    write(
        F / "results/comparison-summary.json",
        dict(
            primary=primary,
            outputs=outputs,
            preferred_variant=selection["preferred_variant"],
            selected_steps={n: r["selected_steps"] for n, r in raw.items()},
            hardware={n: r["runtime"]["nvidia_smi"] for n, r in raw.items()},
            resources={n: r["resources"] for n, r in raw.items()},
        ),
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for j, cls in enumerate(("I", "II")):
        for i, metric in enumerate(("spearman", "rmse")):
            ax = axs[i, j]
            for split, offset, color in (
                ("validation", -0.07, "#21618c"),
                ("test", 0.07, "#a14628"),
            ):
                means = []
                low = []
                high = []
                for v in VARIANTS:
                    a = [
                        reports[f"{v}-s{s}"]["common_ic50"][split][cls][metric]
                        for s in (42, 43)
                    ]
                    mean = float(np.mean(a))
                    means.append(mean)
                    low.append(mean - min(a))
                    high.append(max(a) - mean)
                ax.errorbar(
                    np.arange(4) + offset,
                    means,
                    yerr=[low, high],
                    fmt="o-",
                    color=color,
                    label=split,
                    capsize=4,
                )
            ax.set_xticks(np.arange(4), LABELS, rotation=14, ha="right")
            ax.set_title(
                f"MHC-{cls} · {'Spearman ρ (higher better)' if metric == 'spearman' else 'RMSE log₁₀(nM) (lower better)'}"
            )
            ax.grid(axis="y", alpha=0.18)
            if i == 0 and j == 0:
                ax.legend(frameon=False)
    fig.suptitle(
        "Shared-v5 scale comparison · original IC50 development holdouts\nTwo-seed means; whiskers show the two seeds, not confidence intervals",
        fontsize=13,
    )
    fig.savefig(F / "results/ic50-comparison.png", dpi=180)
    fig.savefig(F / "results/ic50-comparison.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    colors = ("#21618c", "#a14628", "#277451", "#7a5195")
    for ax, cls in zip(axs, ("I", "II")):
        for v, label, color in zip(VARIANTS, LABELS, colors):
            for seed in (42, 43):
                h = read(F / "results" / f"{v}-s{seed}" / "history.json")
                ax.plot(
                    [x["steps"] for x in h],
                    [x["validation"]["ic50"]["by_class"][cls]["spearman"] for x in h],
                    color=color,
                    alpha=0.85 if seed == 42 else 0.5,
                    ls="-" if seed == 42 else "--",
                    label=label if seed == 42 else None,
                )
        ax.set(
            title=f"MHC-{cls} validation rank",
            xlabel="Optimizer updates",
            ylabel="Spearman ρ",
        )
        ax.grid(alpha=0.15)
    axs[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Same update budget; six-output fits receive fewer IC50 labels")
    fig.savefig(F / "results/training-curves.png", dpi=180)
    fig.savefig(F / "results/training-curves.pdf")
    plt.close(fig)
    failure_audit()
    paired_bootstrap()
    print(
        json.dumps(
            dict(preferred=selection["preferred_variant"], primary=primary), indent=2
        )
    )


if __name__ == "__main__":
    main()
