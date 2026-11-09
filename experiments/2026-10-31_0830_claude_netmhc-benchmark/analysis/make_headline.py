"""Collate per-dump pooled tables into one headline CSV and a markdown table.

Usage: python make_headline.py <experiment_dir>
Pooled values are means of per-study metrics within class x detector; never across
detectors or classes. CI = study-level cluster bootstrap (few studies => wide/degenerate).
"""

import json
import sys
from pathlib import Path

import pandas as pd

exp = Path(sys.argv[1])
res = exp / "results"
manifest = json.loads((exp / "reproduce" / "launch.json").read_text())
rows = []
for entry in manifest["dumps"]:
    pooled = pd.read_csv(res / f"{entry['label']}__pooled.csv")
    pooled.insert(0, "label", entry["label"])
    pooled.insert(1, "family", entry["family"])
    pooled.insert(2, "seed", entry["seed"])
    pooled.insert(3, "split", entry["split"])
    rows.append(pooled)
allp = pd.concat(rows, ignore_index=True)
allp.to_csv(res / "headline_pooled_all.csv", index=False)

lines = []
order = ["presto", "netmhcpan41", "netmhciipan43", "mhcflurry", "train_mean"]
for label, group in allp.groupby("label", sort=False):
    meta = group.iloc[0]
    tag = "DEVELOPMENTAL (validation)" if meta["split"] == "validation" else "held-out test"
    lines += [f"### {label} ({tag})", ""]
    lines += [
        "| class | detector | method | studies | rows | Spearman [95% CI] | Fisher-z rho "
        "| Pearson | RMSE log10 | AUROC<=500nM | bal.acc | Spearman gap vs Presto [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (cls, det), g in group.groupby(["mhc_class", "detector"], sort=True):
        for method in order:
            m = g[g["method"] == method]
            if m.empty:
                continue

            def val(metric, field="mean_of_study", m=m):
                r = m[m["metric"] == metric]
                v = r[field].iloc[0] if len(r) else float("nan")
                return v

            sp = m[m["metric"] == "spearman"].iloc[0]
            ci = f"{sp['mean_of_study']:.3f} [{sp['ci_lo']:.3f}, {sp['ci_hi']:.3f}]"
            gap = (
                f"{sp['diff_vs_presto']:+.3f} [{sp['diff_ci_lo']:+.3f}, {sp['diff_ci_hi']:+.3f}]"
                if pd.notna(sp["diff_vs_presto"])
                else ""
            )
            if pd.isna(sp["mean_of_study"]):
                ci = "undefined (constant)"
            lines.append(
                f"| {cls} | {det} | {method} | {int(sp['n_studies'])} | {int(sp['n_rows'])} "
                f"| {ci} | {val('spearman', 'fisher_z_weighted'):.3f} | {val('pearson'):.3f} "
                f"| {val('rmse'):.3f} | {val('auroc'):.3f} | {val('balanced_accuracy'):.3f} | {gap} |"
            )
    lines.append("")
(res / "headline_pooled_all.md").write_text("\n".join(lines))
print("\n".join(lines))
