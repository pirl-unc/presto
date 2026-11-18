"""Render collected selected-checkpoint evidence, without choosing a winner."""

import json
from pathlib import Path

import pandas as pd

E = Path(__file__).resolve().parents[1]


def table(frame):
    def value(item):
        if pd.isna(item):
            return "—"
        if isinstance(item, float):
            return f"{item:.4f}"
        return str(item).replace("|", "\\|")

    rows = [
        list(frame.columns),
        ["---"] * len(frame.columns),
        *frame.itertuples(index=False, name=None),
    ]
    return "\n".join("| " + " | ".join(value(v) for v in row) + " |" for row in rows)


def main():
    out = E / "results/analysis"
    metrics = pd.read_csv(out / "metrics.csv")
    selection = []
    for path in sorted((E / "results/fits").glob("*-n6000/receipt.json")):
        folder = path.parent
        receipt = json.loads(path.read_text())
        exposure = json.loads((folder / "actual-training-exposure.json").read_text())
        config = json.loads((folder / "manifest.json").read_text())
        hardware = json.loads((folder / "hardware.json").read_text())
        selection.append(
            {
                "model": folder.name,
                "selected_step": exposure["selected_steps"],
                "executed_step": receipt["steps"],
                "coverage": receipt["coverage"],
                "wall_minutes": receipt["runtime"]["total_wall_seconds"] / 60,
                "training_minutes": receipt["runtime"]["training_and_evaluation_seconds"] / 60,
                "gpu": hardware["name"],
                "memory_bytes": hardware["memory_bytes"],
                "pathway": config["model"].get("elution_pathway_conditioning", False),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
        )
    for path in sorted((E / "results/evaluation").glob("*-terminal/receipt.json")):
        receipt = json.loads(path.read_text())
        folder = E / "results/fits" / receipt["parent_run"]
        fit = json.loads((folder / "receipt.json").read_text())
        config = json.loads((folder / "manifest.json").read_text())
        selection.append(
            {
                "model": receipt["run"],
                "selected_step": 6000,
                "executed_step": 6000,
                "coverage": "terminal coverage superset",
                "wall_minutes": fit["runtime"]["total_wall_seconds"] / 60,
                "training_minutes": fit["runtime"]["training_and_evaluation_seconds"] / 60,
                "gpu": receipt["hardware"]["name"],
                "memory_bytes": receipt["hardware"]["memory_bytes"],
                "pathway": config["model"].get("elution_pathway_conditioning", False),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
        )
    selected = pd.DataFrame(selection)
    selected.to_csv(out / "selected-runs.csv", index=False)
    present = metrics[
        (metrics.endpoint == "elution_observation")
        & metrics.population.isin(["apc", "broad", "expanded", "matched"])
        & (metrics["class"] != "pooled")
    ].copy()
    present["cohort_class"] = present.population + " " + present["class"]
    aucs = present.pivot(index="model", columns="cohort_class", values="auroc").reset_index()
    aucs.to_csv(out / "presentation-comparison.csv", index=False)
    guard = []
    for population in ["affinity", "affinity_ancestry_screened"]:
        sub = metrics[
            (metrics.endpoint == "ic50")
            & (metrics.population == population)
            & (metrics["class"] != "pooled")
        ]
        for _, row in sub.iterrows():
            ref = sub[(sub.model == "previous-default") & (sub["class"] == row["class"])].iloc[0]
            ds = row.spearman - ref.spearman
            dr = row.rmse_log10_nm - ref.rmse_log10_nm
            guard.append(
                {
                    "model": row.model,
                    "population": population,
                    "class": row["class"],
                    "exact_n": int(row.exact_n),
                    "spearman": row.spearman,
                    "rmse_log10_nm": row.rmse_log10_nm,
                    "delta_spearman": ds,
                    "delta_rmse": dr,
                    "operational_guard_passed": ds >= -0.02 and dr <= 0.03,
                }
            )
    guards = pd.DataFrame(guard)
    guards.to_csv(out / "affinity-retention.csv", index=False)
    body = [
        "# Collected selected-checkpoint comparison",
        "",
        "Development validation only. TEST remains sealed. Unmarked model IDs use the checkpoint "
        "selected by the registered equal-class/equal-study presentation BCE. Models marked "
        "terminal use the fixed6000-update endpoint, never an intermediate AUROC peak. "
        "For square-root recipes this comparison was registered before terminal outcomes; "
        "original-recipe terminal results are retrospective matched-update controls. "
        "Ongoing or uncollected runs are absent. This report does not promote a checkpoint.",
        "",
        "## Presentation AUROC",
        "",
        table(aucs),
        "",
        "Old APC and broad cohorts, newly protected validation, and affinity-matched pairs are "
        "different evaluation populations. Compare columns separately. Full AUPRC, BCE, pair "
        "wins and per-study results are in metrics.csv and study-metrics.csv.",
        "",
        "## Selected checkpoints and runtime",
        "",
        table(selected.drop(columns=["checkpoint_sha256", "gpu", "memory_bytes"])),
        "",
        "Every collected predictive fit requested H100!; observed hardware and checkpoint hashes "
        "are retained in selected-runs.csv. Update budgets match; "
        "wall times differ with candidate sets.",
        "",
        "## Measured IC50 retention",
        "",
        table(guards),
        "",
        "The screening guard is Spearman decline no greater than0.02 and log10(nM) RMSE increase "
        "no greater than0.03 against the previous default. It is an operational screen, not a "
        "formal non-inferiority result. The ancestry-screened subset removes346 historically "
        "related affinity rows; no exact affinity validation matches were found.",
        "",
        "## Uncertainty",
        "",
        "Paired bootstrap files compare identical examples using protein-family "
        "or sample/condition "
        "clusters. They condition on selected fitted models and this reused development cohort. "
        "They do not include training-seed variation or correct adaptive model selection. "
        "The broad class-I cohort has only three study-component split groups; no protein-cluster "
        "interval or incomplete-cohort pair-win estimate is reported for it. "
        "Cluster counts, finite draws and skipped contrasts are explicit "
        "in the analysis artifacts.",
        "",
    ]
    (out / "report.md").write_text("\n".join(body))
    print(out / "report.md")


if __name__ == "__main__":
    main()
