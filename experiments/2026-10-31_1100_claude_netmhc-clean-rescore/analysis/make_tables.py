"""Render markdown tables from results/*.csv. Usage: python make_tables.py <experiment_dir>."""

import sys
from pathlib import Path

import pandas as pd

res = Path(sys.argv[1]) / "results"
HEAD = ["direct-s42_test", "direct-s43_test", "curated-multi-s43_test"]
fmt = lambda x, p=3: "" if pd.isna(x) else f"{x:.{p}f}"  # noqa: E731
lines = []

frac = pd.read_csv(res / "seen_fraction_by_study.csv")
f = frac[(frac["label"] == "direct-s43_test") & (frac["rows"] == "exact_ba")]
t = f.groupby(["mhc_class", "tool", "variant"])[["n", "n_seen"]].sum().reset_index()
t["frac"] = t["n_seen"] / t["n"]
lines += ["## Seen fraction of held-out exact-BA test rows (identical rows in every family)", "",
          "| class | tool | variant | rows | seen | fraction |", "|---|---|---|---|---|---|"]  # fmt: skip
for r in t.itertuples():
    lines.append(f"| {r.mhc_class} | {r.tool} | {r.variant} | {r.n} | {r.n_seen} | {r.frac:.3f} |")

lines += ["", "## Seen fraction per study (pair variant, direct-s43_test, exact BA)", "",
          "| class | tool | detector | study | n | seen | fraction |", "|---|---|---|---|---|---|---|"]  # fmt: skip
for r in f[f["variant"] == "pair"].itertuples():
    lines.append(f"| {r.mhc_class} | {r.tool} | {r.detector} | {r.study_id} | {r.n} | "
                 f"{r.n_seen} | {r.frac_seen:.3f} |")  # fmt: skip

gap = pd.read_csv(res / "gap_before_after.csv")
for label in HEAD:
    for variant in ("pair", "peptide", "nested"):
        g = gap[(gap["label"] == label) & (gap["variant"] == variant) & (gap["scope"] != "common")]
        lines += ["", f"## Tool-minus-Presto gap, {label}, variant={variant}, own clean subset", "",
                  "Gap > 0 favours the tool for Spearman/Pearson/AUROC; for RMSE gap < 0 favours the tool. "
                  "`full` = all PR #149 studies; `matched` = full restricted to the studies surviving in the clean cell; "
                  "`clean` = rows the tool has not seen.", "",
                  "| class | detector | tool | metric | full gap (studies/rows) | matched gap | clean gap [CI] | clean studies/rows | tool / Presto on clean | status |",
                  "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in g.sort_values(["mhc_class", "detector", "tool", "metric"]).itertuples():
            ci = (f"{fmt(r.clean_gap)} [{fmt(r.clean_gap_lo)}, {fmt(r.clean_gap_hi)}]"
                  if pd.notna(r.clean_gap) else "n/a")  # fmt: skip
            cs = (f"{int(r.clean_n_studies)}/{int(r.clean_n_rows)}"
                  if pd.notna(r.clean_n_studies) else "0/0")  # fmt: skip
            lines.append(
                f"| {r.mhc_class} | {r.detector} | {r.tool} | {r.metric} | {fmt(r.full_gap)} "
                f"({int(r.full_n_studies)}/{int(r.full_n_rows)}) | {fmt(r.fullmatch_gap)} | {ci} | "
                f"{cs} | {fmt(r.clean_tool)} / {fmt(r.clean_presto)} | {r.reportable} |"
            )  # fmt: skip
(res / "headline_clean.md").write_text("\n".join(lines) + "\n")
