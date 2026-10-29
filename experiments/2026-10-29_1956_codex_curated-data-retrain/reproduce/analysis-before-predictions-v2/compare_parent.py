"""Paired fixed-holdout comparisons and source-identified label sensitivity."""

from pathlib import Path
import csv
import gzip
import hashlib
import json
import time

from verify_predictions import groups, metrics

F = Path(__file__).resolve().parents[1]
P = F.parent / "2026-10-28_1540_codex_temporal-multi-output"
KEYS = ("split", "era", "endpoint", "mhc_class", "population", "study", "assay_method")
SCORES = ("spearman", "pearson", "rmse", "auroc", "auprc", "censor_satisfaction")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_rows(before, after):
    """Require identical evaluation strata before comparing the two selected fits."""
    left = {tuple(row[k] for k in KEYS): row for row in before}
    right = {tuple(row[k] for k in KEYS): row for row in after}
    assert len(left) == len(before) and len(right) == len(after)
    assert left.keys() == right.keys()
    result = []
    for key in sorted(left):
        a, b = left[key], right[key]
        for count in ("n", "n_exact", "n_censored", "positives", "negatives"):
            assert a.get(count) == b.get(count), (key, count)
        row = dict(zip(KEYS, key))
        row.update(n=a["n"], n_exact=a.get("n_exact"))
        for metric in SCORES:
            x, y = a.get(metric), b.get(metric)
            row["parent_" + metric] = x
            row["curated_" + metric] = y
            row["delta_" + metric] = y - x if x is not None and y is not None else None
        result.append(row)
    return result


def closeout():
    """Publish completed evidence only after every independent reporting gate passes."""
    for name, status in (
        ("reporting-receipt", "completed"),
        ("prediction-verification", "passed"),
        ("cohort-composition-verification", "passed"),
        ("parent-comparison", "passed"),
    ):
        assert read(F / "results" / (name + ".json"))["status"] == status
    selection = read(F / "results/validation-decision.json")
    assert selection["status"] == "frozen" and selection["test_used_for_selection"] is False
    rows = read(F / "results/stratified-metrics.json")
    lines = [
        "## Verified retrain results",
        "",
        "Both complete weighted-epoch fits, independent target/loss verification, "
        "cohort census, serving replay and paired held-out observation checks passed.",
        "Validation-selected condition: `"
        + selection["winner"]
        + "` ("
        + selection["criterion"]
        + ").",
        "Selection scores: "
        + "; ".join(f"{k} = {v:.6f}" for k, v in selection["scores"].items())
        + ".",
        "All results remain repeated fixed-holdout development evidence; no model is "
        "automatically promoted and no mechanistic qualification follows.",
        "",
        "| Condition | Split / era | Output / class / population | n (exact for IC50) "
        "| Spearman | RMSE log10(nM) | AUROC | AUPRC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "—" if value is None else f"{value:.4f}"

    for row in sorted(rows, key=lambda r: (r["condition"], *(r[k] for k in KEYS))):
        if not (
            (row["endpoint"] == "ic50" and row["study"] == "all")
            or (
                row["endpoint"] == "elution_observation"
                and row["population"] in {"synthetic_same_protein", "source_reported_outcomes"}
            )
        ):
            continue
        population = row["population"]
        if row["study"] != "all":
            population += " / " + row["study"]
        label = " / ".join((row["endpoint"], row["mhc_class"], population))
        count = row["n_exact"] if row["endpoint"] == "ic50" else row["n"]
        values = " | ".join(number(row.get(k)) for k in ("spearman", "rmse", "auroc", "auprc"))
        lines.append(
            f"| {row['condition']} | {row['split']} / {row['era']} | {label} | {count} | {values} |"
        )
    lines.extend(["", "Missing metrics indicate unsupported or single-label evaluation cells.", ""])
    for condition in sorted(selection["scores"]):
        report = read(F / "results" / condition / "report.json")
        assert report["all_training_rows_exposed"] is True
        assert report["serving_replay"]["status"] == "passed"
        resources, hardware = report["resources"], report["hardware"]
        lines.append(
            f"- `{condition}`: selected step {report['selected_steps']:,}; completed "
            f"{report['summary']['steps']:,} updates and exposed "
            f"{report['unique_training_rows_seen']:,} unique TRAIN rows. "
            f"Requested H100!, observed {hardware['actual_gpu']} "
            f"({hardware['total_gpu_bytes']:,} device bytes); wall "
            f"{resources['wall_seconds']:.1f} s, optimizer {resources['optimizer_seconds']:.1f} s, "
            f"peak allocated/reserved {resources['peak_gpu_allocated'] / 1e9:.2f} / "
            f"{resources['peak_gpu_reserved'] / 1e9:.2f} GB."
        )
    lines.extend(
        [
            "",
            "The full report preserves every output/class/era/study, exact and bounded "
            "loss, Pearson, 500 nM classification, uncertainty, source-reported outcomes, "
            "parameter/feature coverage and register diagnostics. The paired comparison "
            "checks identical held-out observations and separately excludes the three "
            "source-identified suspect generated test labels for both model generations.",
            "",
        ]
    )
    text = "\n".join(lines)
    readme = F / "README.md"
    original = readme.read_text().split("\n## Verified retrain results", 1)[0]
    original = original.replace(
        "Status: two H100! fits submitted; source preparation passed. "
        "No new predictive result yet.",
        "Status: both fits completed; held-out metrics and paired comparisons "
        "independently verified.",
    )
    original = original.replace("The running continuation", "The completed continuation")
    original = original.replace(
        "Runtime/peak memory remain pending.",
        "Final runtime and peak memory are recorded below.",
    )
    original = original.replace(
        "No improvement is\nclaimed until measured; "
        "final scientific writeup and promotion remain pending.",
        "The results below record measured differences; model promotion and scientific "
        "qualification require separate review.",
    )
    readme.write_text(
        original.rstrip()
        + "\n\n"
        + text
        + "[Full report](results/summary.md) · [Paired comparison](results/parent-comparison.md)\n"
    )
    registry = F.parent / "experiment_log.md"
    document = registry.read_text()
    start = document.index("## " + F.name)
    next_entry = document.find("\n## ", start + 3)
    end = len(document) if next_entry < 0 else next_entry
    entry = document[start:end]
    entry = entry.split("\n### Verified retrain results", 1)[0]
    entry = entry.replace("— frozen cohort, fits running", "— completed, independently verified")
    entry = entry.replace(
        "Actual hardware/runtime and held-out metrics pending. No new winner or claim of\n"
        "  improvement. Collection/verification/reporting continuation and live comparison\n"
        "  are running; complete artifact/metric closeout remains required.",
        "Actual hardware, runtime, validation selection and held-out metrics are recorded\n"
        "  below after all verification/reporting gates passed. No automatic model promotion.",
    )
    entry += "\n" + text.replace("## Verified retrain results", "### Verified retrain results", 1)
    entry += f"\n[Complete report]({F.name}/results/summary.md) · "
    entry += f"[Paired parent comparison]({F.name}/results/parent-comparison.md).\n"
    registry.write_text(document[:start] + entry + document[end:])
    receipt = dict(
        status="completed",
        completed_unix=time.time(),
        selected_condition=selection["winner"],
        automatic_promotion=False,
        analyzer_sha256=sha(Path(__file__)),
        artifacts={
            name: sha(F / "results" / name)
            for name in (
                "parent-comparison.json",
                "prediction-verification.json",
                "reporting-receipt.json",
                "stratified-metrics.json",
            )
        },
    )
    (F / "results/closeout.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    assert read(F / "results/prediction-verification.json")["status"] == "passed"
    assert read(P / "results/prediction-verification.json")["status"] == "passed"
    parent = read(P / "results/stratified-metrics.json")
    current = read(F / "results/stratified-metrics.json")
    comparisons, sensitivity, prediction_hashes = [], [], {}
    audit = read(F / "results/blacklist-audit.json")
    flagged = {
        r["observation_id"]
        for r in audit["intersections"]
        if r["generated"] and r["split"] == "test"
    }
    assert len(flagged) == 3
    for seed in (42, 43):
        old_name, new_name = "wide-multi-s" + str(seed), "curated-multi-s" + str(seed)
        comparisons.extend(
            dict(seed=seed, **r)
            for r in paired_rows(
                [r for r in parent if r["condition"] == old_name],
                [r for r in current if r["condition"] == new_name],
            )
        )
        for split in ("validation", "test"):
            old_observations = None
            for family, name in ((P, old_name), (F, new_name)):
                folder = family / "results" / name
                collection = read(folder / "collection.json")
                spec = collection["predictions"][split]
                path = Path(spec["path"])
                assert sha(path) == spec["sha256"]
                prediction_hashes[family.name + ":" + name + ":" + split] = spec["sha256"]
                with gzip.open(path, "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                observations = {
                    r["observation_id"]: json.dumps(r["observation"], sort_keys=True) for r in rows
                }
                assert len(observations) == len(rows)
                if old_observations is None:
                    old_observations = observations
                else:
                    assert observations == old_observations
                if split != "test":
                    continue
                assert flagged <= observations.keys()
                affected = {
                    key: subset
                    for key, subset in groups(rows).items()
                    if any(r["observation_id"] in flagged for r in subset)
                }
                for key, subset in affected.items():
                    era, endpoint, cls, population, study, method = key
                    assert endpoint == "elution_observation"
                    retained = [r for r in subset if r["observation_id"] not in flagged]
                    sensitivity.append(
                        dict(
                            family=family.name,
                            condition=name,
                            seed=seed,
                            split="test",
                            era=era,
                            endpoint=endpoint,
                            mhc_class=cls,
                            population=population,
                            study=study,
                            assay_method=method,
                            excluded_rows=len(subset) - len(retained),
                            original=metrics(subset, 1.0),
                            excluding_source_identified_suspect_negatives=metrics(retained, 1.0),
                        )
                    )
    result = dict(
        status="passed",
        heldout_observations_identical=True,
        interpretation="Repeated fixed-holdout development comparison; no test-based selection",
        exposure="Same complete weighted epoch policy; recomputed quotas and update budget",
        metrics_note="Positive delta helps correlation/AUC; negative delta helps RMSE",
        paired=comparisons,
        suspect_label_sensitivity=sensitivity,
        prediction_hashes=prediction_hashes,
        blacklist_audit_sha256=sha(F / "results/blacklist-audit.json"),
    )
    (F / "results/parent-comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (F / "results/parent-comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    lines = [
        "# Data-curation retrain versus parent",
        "",
        result["interpretation"] + ".",
        "Same held-out observations verified by ID and all observation fields.",
        "",
        "| Seed | Split / era | Output / class / population | Exact n or binary n "
        "| Parent | Curated | Delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["study"] != "all":
            continue
        metric = (
            "auprc"
            if row["endpoint"] in {"elution_observation", "qualitative_binding"}
            else "spearman"
        )
        if row["parent_" + metric] is None:
            continue
        label = " / ".join(row[k] for k in ("endpoint", "mhc_class", "population"))
        values = [f"{row[prefix + metric]:.4f}" for prefix in ("parent_", "curated_", "delta_")]
        lines.append(
            f"| {row['seed']} | {row['split']} / {row['era']} | {label} ({metric}) | "
            f"{row['n_exact'] if row['n_exact'] is not None else row['n']} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "All assay, class and study strata, including RMSE, Pearson, AUROC and"
            " censor satisfaction, are in `parent-comparison.csv`/`.json`. Three source-identified"
            " suspect generated test negatives are preserved in the main comparison and excluded"
            " in the separately recorded sensitivity analysis for both model generations.",
            "",
        ]
    )
    (F / "results/parent-comparison.md").write_text("\n".join(lines))
    closeout()
    print("Paired parent comparison and source-identified label sensitivity verified", flush=True)


if __name__ == "__main__":
    main()
