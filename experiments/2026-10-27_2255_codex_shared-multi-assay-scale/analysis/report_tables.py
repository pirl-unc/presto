"""Render review tables from independently verified closure artifacts."""

from pathlib import Path
import json

F = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def number(value, places=4):
    return "—" if value is None else f"{value:.{places}f}"


def main():
    comparison = read(F / "results/comparison-summary.json")
    assert read(F / "results/verification-summary.json")["status"] == "passed"
    lines = [
        "# Independently verified comparison tables",
        "",
        "Two-seed means of individual-model metrics, not ensemble predictions.",
        "IC50 comparisons use identical original held-out observation IDs.",
        "RMSE is log10(nM), except half-life in log10(minutes).",
        "",
        "## Original IC50 cohort",
        "",
        "| Condition | I validation ρ / RMSE | I test ρ / RMSE | II validation ρ / RMSE | II test ρ / RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    primary = {
        (r["variant"], r["split"], r["mhc_class"]): r for r in comparison["primary"]
    }
    variants = list(dict.fromkeys(r["variant"] for r in comparison["primary"]))
    for variant in variants:
        cells = []
        for cls in ("I", "II"):
            for split in ("validation", "test"):
                r = primary[variant, split, cls]
                cells.append(f"{number(r['spearman'])} / {number(r['rmse'])}")
        lines.append("| " + " | ".join([variant] + cells) + " |")
    lines += [
        "",
        "## Six-output results",
        "",
        "n counts exact observations. All bounds contribute to loss; classification,",
        "Pearson and mean-baseline metrics are in the machine-readable tables.",
        "Tiny or missing strata do not support generalization claims.",
        "",
        "| Condition | Output | Class | Validation n | Validation ρ / RMSE | Test n | Test ρ / RMSE | Test train-mean RMSE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    outputs = {
        (r["variant"], r["endpoint"], r["mhc_class"], r["split"]): r
        for r in comparison["outputs"]
    }
    for variant in ("multi-assay", "multi-assay-wide"):
        for endpoint in (
            "ic50",
            "kd",
            "ec50",
            "kd_via_ic50",
            "kd_via_ec50",
            "half_life",
        ):
            for cls in ("I", "II"):
                val = outputs[variant, endpoint, cls, "validation"]
                test = outputs[variant, endpoint, cls, "test"]
                lines.append(
                    f"| {variant} | {endpoint} | {cls} | {val['n']} | "
                    f"{number(val['spearman'])} / {number(val['rmse'])} | {test['n']} | "
                    f"{number(test['spearman'])} / {number(test['rmse'])} | "
                    f"{number(test['training_mean_rmse'])} |"
                )
    lines += [
        "",
        "## Runtime and selected checkpoint",
        "",
        "All fits completed 6,000 updates. Requested H100!; actual NVIDIA H100 80GB HBM3.",
        "Wall time excludes queue, image build, archive creation and download.",
        "RSS is sampled at updates/end; CUDA peak is the allocator counter.",
        "",
        "| Condition | Selected update | Wall seconds | Sampled RSS GiB | CUDA allocated peak GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, step in comparison["selected_steps"].items():
        resources = comparison["resources"][name]
        lines.append(
            f"| {name} | {step} | {resources['wall_seconds']:.1f} | "
            f"{resources['peak_rss'] / 2**30:.2f} | "
            f"{resources['peak_gpu_allocated'] / 2**30:.2f} |"
        )
    (F / "results/report-tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
