#!/usr/bin/env python3
"""Collect uncollected modal_runs into the experiment registry.

For each experiment family, this script:
1. Detects the data format of each modal_run directory
2. Normalizes data to the aggregate_summary_runs expected format
3. Runs aggregation and generates plots/tables
4. Writes a README.md and reproduce/launch.json
5. Appends an entry to experiments/experiment_log.md
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reuse functions from aggregate_summary_runs
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from aggregate_summary_runs import (
    TEST_METRICS,
    jsonable,
    plot_metric_grid,
    plot_metric_ranking,
    plot_probe_heatmap,
    plot_training_curves,
    summarize_run,
    write_json,
)

ROOT = Path(__file__).resolve().parent.parent
MODAL_RUNS = ROOT / "modal_runs"
EXPERIMENTS = ROOT / "experiments"

# ---------------------------------------------------------------------------
# Format detection & loading
# ---------------------------------------------------------------------------


def resolve_run_dir(path: Path) -> Path:
    """Handle nesting: if path contains a subdir with same name, descend."""
    if not path.is_dir():
        return path
    subdir = path / path.name
    if subdir.is_dir():
        return subdir
    # Check pull/<name>/ pattern (some runs have data in pull/ subdir)
    pull_subdir = path / "pull" / path.name
    if pull_subdir.is_dir():
        return pull_subdir
    # Single data-bearing subdir
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        c = subdirs[0]
        if (c / "summary.json").exists() or (c / "config.json").exists():
            return c
    # Check if a pull/ dir has a single data-bearing subdir
    pull_dir = path / "pull"
    if pull_dir.is_dir():
        pull_subs = [d for d in pull_dir.iterdir() if d.is_dir()]
        if len(pull_subs) == 1:
            c = pull_subs[0]
            if (c / "summary.json").exists() or (c / "config.json").exists():
                return c
    return path


def detect_format(path: Path) -> str:
    """Return 'A', 'B', 'C', 'logs_only', or 'unknown'."""
    summary = path / "summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(data, list):
                return "C"
            return "A"
    if (path / "config.json").exists():
        return "B"
    if any(path.glob("*.log")):
        return "logs_only"
    return "unknown"


def _rename_probe_cols(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    if "kd_nM" in df.columns and "ic50_nM" not in df.columns:
        renames["kd_nM"] = "ic50_nM"
    if "kd_log10" in df.columns and "ic50_log10" not in df.columns:
        renames["kd_log10"] = "ic50_log10"
    return df.rename(columns=renames) if renames else df


def _load_probe_csv(path: Path) -> pd.DataFrame:
    probe_path = path / "probe_affinity_over_epochs.csv"
    if probe_path.exists() and probe_path.stat().st_size > 0:
        df = pd.read_csv(probe_path)
        return _rename_probe_cols(df)
    return pd.DataFrame()


def load_format_a(path: Path) -> dict:
    """Load Format A: summary.json with epochs/epoch_summaries + probe CSV."""
    summary = json.loads((path / "summary.json").read_text())
    config = summary.get("config", {})
    epochs = summary.get("epochs", summary.get("epoch_summaries", []))
    epoch_df = pd.DataFrame(epochs)
    if not epoch_df.empty and "epoch" in epoch_df.columns:
        epoch_df["epoch"] = epoch_df["epoch"].astype(int)
    probe_df = _load_probe_csv(path)
    return {
        "config": config,
        "epoch_df": epoch_df,
        "probe_df": probe_df,
        "test_metrics": summary.get("test_metrics", {}),
        "raw_summary": summary,
    }


def load_format_b(path: Path) -> dict:
    """Load Format B: config.json + metrics.jsonl + probe CSV."""
    config = json.loads((path / "config.json").read_text())

    epoch_df = pd.DataFrame()
    metrics_jsonl = path / "metrics.jsonl"
    if metrics_jsonl.exists() and metrics_jsonl.stat().st_size > 0:
        rows = []
        with metrics_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if rows:
            mdf = pd.DataFrame(rows)
            # JSONL format: each row has step, split, and all metrics as columns
            if "step" in mdf.columns and "split" in mdf.columns:
                val_rows = mdf[mdf["split"] == "val"].copy()
                train_rows = mdf[mdf["split"] == "train"].copy()
                if not val_rows.empty:
                    epoch_records = []
                    for step in sorted(val_rows["step"].unique()):
                        rec = {"epoch": int(step)}
                        vr = val_rows[val_rows["step"] == step].iloc[0]
                        if "loss" in vr.index:
                            rec["val_loss"] = float(vr["loss"])
                        tr = train_rows[train_rows["step"] == step]
                        if not tr.empty and "loss" in tr.columns:
                            rec["train_loss"] = float(tr.iloc[0]["loss"])
                        epoch_records.append(rec)
                    epoch_df = pd.DataFrame(epoch_records)

    probe_df = _load_probe_csv(path)
    return {
        "config": config,
        "epoch_df": epoch_df,
        "probe_df": probe_df,
        "test_metrics": {},
        "raw_summary": {"config": config},
    }


def load_format_c(path: Path) -> list[dict]:
    """Load Format C: sweep summary.json (list of candidates)."""
    candidates = json.loads((path / "summary.json").read_text())
    results = []
    for cand in candidates:
        if cand.get("status") == "failed":
            continue
        label = cand.get("candidate", "unknown")
        config = {
            "label": label,
            "d_model": cand.get("d_model"),
            "n_layers": cand.get("n_layers"),
            "n_heads": cand.get("n_heads"),
            "n_params": cand.get("n_params"),
        }
        epoch_rows = []
        if cand.get("val_best_loss") is not None:
            epoch_rows.append({
                "epoch": 1,
                "val_loss": cand.get("val_best_loss"),
                "train_loss": cand.get("train_best_loss"),
            })
        epoch_df = pd.DataFrame(epoch_rows)

        # Try per-candidate probe CSV
        probe_df = pd.DataFrame()
        for f in sorted(path.glob(f"*{label}*probe_affinity*.csv")):
            try:
                probe_df = _rename_probe_cols(pd.read_csv(f))
            except Exception:
                pass
            break

        results.append({
            "config": config,
            "epoch_df": epoch_df,
            "probe_df": probe_df,
            "test_metrics": {},
            "raw_summary": cand,
            "run_name": label,
        })
    return results


# ---------------------------------------------------------------------------
# Normalization: write to aggregate_summary_runs expected format
# ---------------------------------------------------------------------------


def normalize_to_aggregate(run_data: dict, out_dir: Path, run_name: str) -> None:
    """Write summary.json + probes.jsonl for aggregate_summary_runs consumption."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config = run_data["config"].copy()
    if "label" not in config:
        config["label"] = run_name

    epoch_summaries = []
    edf = run_data["epoch_df"]
    if not edf.empty:
        for _, row in edf.iterrows():
            entry = {}
            for col in edf.columns:
                val = row[col]
                if pd.notna(val):
                    try:
                        entry[col] = int(val) if col == "epoch" else float(val)
                    except (ValueError, TypeError):
                        entry[col] = val
            epoch_summaries.append(entry)

    summary = {"config": config, "epoch_summaries": epoch_summaries}
    if run_data.get("test_metrics"):
        summary["test_metrics"] = run_data["test_metrics"]

    write_json(out_dir / "summary.json", summary)

    pdf = run_data["probe_df"]
    if not pdf.empty:
        with (out_dir / "probes.jsonl").open("w") as fh:
            for _, row in pdf.iterrows():
                rec = {}
                for col in pdf.columns:
                    val = row[col]
                    if pd.notna(val):
                        if col == "epoch":
                            rec[col] = int(val)
                        elif isinstance(val, (int, float, np.integer, np.floating)):
                            rec[col] = float(val)
                        else:
                            rec[col] = str(val)
                fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Aggregation + plotting (reuses aggregate_summary_runs logic)
# ---------------------------------------------------------------------------


def run_dirs_list(root: Path) -> list[Path]:
    """Find all run dirs with summary.json under root."""
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists())


def aggregate_and_plot(exp_dir: Path) -> pd.DataFrame:
    """Run aggregation and plotting, return summary_df."""
    runs_root = exp_dir / "results" / "runs"
    output_dir = exp_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, epoch_frames, probe_frames = [], [], []
    for rd in run_dirs_list(runs_root):
        row, edf, pdf = summarize_run(rd)
        rows.append(row)
        if not edf.empty:
            epoch_frames.append(edf)
        if not pdf.empty:
            probe_frames.append(pdf)

    if not rows:
        print(f"  WARNING: no valid runs found in {runs_root}")
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows)
    sort_metric = "test_spearman" if "test_spearman" in summary_df else "label"
    summary_df["display_label"] = summary_df["label"]
    dup = summary_df["label"].duplicated(keep=False)
    summary_df.loc[dup, "display_label"] = summary_df.loc[dup].apply(
        lambda r: f"{r['label']} [{r['run_id']}]", axis=1
    )
    summary_df = summary_df.sort_values(sort_metric, ascending=(sort_metric == "label"))

    epoch_df = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()
    probe_df = pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame()
    if not epoch_df.empty:
        epoch_df["display_label"] = epoch_df["run_id"].map(summary_df.set_index("run_id")["display_label"])
    if not probe_df.empty:
        probe_df["display_label"] = probe_df["run_id"].map(summary_df.set_index("run_id")["display_label"])

    summary_df.to_csv(output_dir / "condition_summary.csv", index=False)
    epoch_df.to_csv(output_dir / "epoch_summary.csv", index=False)
    probe_df.to_csv(output_dir / "final_probe_predictions.csv", index=False)
    write_json(output_dir / "condition_summary.json", summary_df.to_dict(orient="records"))

    try:
        plot_metric_ranking(summary_df, output_dir / "test_spearman_ranking.png")
    except Exception as e:
        print(f"  plot_metric_ranking skipped: {e}")
    try:
        plot_metric_grid(summary_df, output_dir / "test_metric_grid.png")
    except Exception as e:
        print(f"  plot_metric_grid skipped: {e}")
    try:
        plot_training_curves(epoch_df, output_dir / "training_curves.png")
    except Exception as e:
        print(f"  plot_training_curves skipped: {e}")
    try:
        plot_probe_heatmap(probe_df, output_dir / "final_probe_heatmap.png")
    except Exception as e:
        print(f"  plot_probe_heatmap skipped: {e}")

    return summary_df


# ---------------------------------------------------------------------------
# README + launch.json generation
# ---------------------------------------------------------------------------


def write_readme(exp_dir: Path, spec: dict, summary_df: pd.DataFrame) -> None:
    """Write experiment README.md."""
    lines = [
        f"# {spec['title']}",
        "",
        f"**EXP ID**: {spec['exp_id']}",
        f"**Date**: {spec['date']}",
        f"**Agent**: {spec['agent']}",
        "",
        "## Overview",
        "",
        spec["description"],
        "",
        "## Dataset & Training",
        "",
        spec.get("training_notes", "See individual run configs for details."),
        "",
        "## Source Modal Runs",
        "",
    ]
    for d in spec["modal_runs"]:
        lines.append(f"- `modal_runs/{d}/`")
    lines.append("")

    if not summary_df.empty:
        lines.append("## Conditions")
        lines.append("")

        # Build conditions table
        cols = ["label"]
        for c in ["final_epoch", "best_val_loss", "best_val_spearman",
                   "test_spearman", "test_auroc", "test_f1", "test_rmse_log10"]:
            if c in summary_df.columns:
                cols.append(c)

        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines.append(header)
        lines.append(sep)
        for _, row in summary_df.iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v) if pd.notna(v) else "")
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # Plots
    lines.append("## Plots")
    lines.append("")
    for png in ["training_curves.png", "final_probe_heatmap.png",
                "test_spearman_ranking.png", "test_metric_grid.png"]:
        if (exp_dir / "results" / png).exists():
            lines.append(f"![{png}](results/{png})")
            lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Condition summary: `results/condition_summary.csv`")
    lines.append(f"- Epoch summary: `results/epoch_summary.csv`")
    lines.append(f"- Probe predictions: `results/final_probe_predictions.csv`")
    lines.append(f"- Reproduce: `reproduce/launch.json`")
    lines.append("")

    (exp_dir / "README.md").write_text("\n".join(lines))


def write_launch_json(exp_dir: Path, spec: dict) -> None:
    """Write reproduce/launch.json pointing to raw modal_runs."""
    repro = exp_dir / "reproduce"
    repro.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": spec["exp_id"],
        "title": spec["title"],
        "date": spec["date"],
        "agent": spec["agent"],
        "modal_run_dirs": [f"modal_runs/{d}" for d in spec["modal_runs"]],
        "collection_timestamp": datetime.now().isoformat(),
        "note": "These runs were collected retrospectively from modal_runs/ into the experiment registry.",
    }
    write_json(repro / "launch.json", payload)


# ---------------------------------------------------------------------------
# Scorepath special handler: find all summary.json recursively
# ---------------------------------------------------------------------------


def find_scorepath_runs(base: Path) -> list[tuple[str, Path]]:
    """Find all runs in the scorepath_bench dir (nested in e00X-live/ and pulls/)."""
    runs = []
    for sj in sorted(base.rglob("summary.json")):
        rd = sj.parent
        # Use the immediate parent dir name as the run name
        runs.append((rd.name, rd))
    return runs


# ---------------------------------------------------------------------------
# Experiment family definitions
# ---------------------------------------------------------------------------

FAMILIES = [
    {
        "exp_id": "EXP-21",
        "slug": "a0201-a2402-target-bakeoff",
        "title": "A*02:01/A*24:02 Target Encoding & Probe Systematic Comparison",
        "date": "2026-03-08",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Systematic comparison of target encoding modes (direct, IC50-exact, numeric-synth), "
                       "probe configurations (probe-only, shared-probe, peptide-rank, allele-rank), "
                       "and training variants (balanced, contrastive, warmstart, synthetics) on the "
                       "2-allele HLA-A*02:01/A*24:02 panel.",
        "training_notes": "2-allele panel (A*02:01, A*24:02). 10 epochs, batch 512 (direct/numeric) or 128 (IC50-exact). "
                          "GrooveTransformerModel, seed 42. Various measurement profiles and assay modes. "
                          "Some runs warm-started from mhc-pretrain-20260308b.",
        "format": "A",
        "modal_runs": [
            "a0201-a2402-direct-balanced-20260308a",
            "a0201-a2402-direct-balanced-contrastive-20260308b",
            "a0201-a2402-direct-balanced-peptide-only-20260308d",
            "a0201-a2402-direct-balanced-peptide-rank-20260308c",
            "a0201-a2402-ic50-exact-allrows-ic50only-20260308h",
            "a0201-a2402-ic50-exact-allrows-ic50only-warmstart-20260308i",
            "a0201-a2402-ic50-exact-warmstart-allele-rank-20260308b",
            "a0201-a2402-ic50-exact-warmstart-peptide-rank-20260308b",
            "a0201-a2402-ic50-exact-warmstart-synth-20260308b",
            "a0201-a2402-ic50-probe-only-20260308e",
            "a0201-a2402-ic50-probe-peptide-rank-20260308e",
            "a0201-a2402-ic50-shared-probe-only-20260308f",
            "a0201-a2402-ic50-shared-probe-peptide-rank-20260308f",
            "a0201-a2402-numeric-synth-balanced-20260308a",
            "a0201-a2402-numeric-synth-balanced-contrastive-20260308b",
            "a0201-a2402-numeric-synth-balanced-peptide-only-20260308d",
            "a0201-a2402-numeric-synth-balanced-peptide-rank-20260308c",
        ],
    },
    {
        "exp_id": "EXP-22",
        "slug": "class1-panel-quant",
        "title": "7-Allele Class-I Panel Warmstart Variants",
        "date": "2026-03-08",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Three 7-allele panel training variants: IC50-exact warmstart, "
                       "IC50-exact warmstart with synthetics + peptide ranking, and "
                       "quantitative affinity warmstart.",
        "training_notes": "7-allele class-I panel (A*02:01, A*03:01, A*11:01, A*01:01, A*24:02, B*07:02, B*44:02). "
                          "12 epochs, batch 140, GrooveTransformerModel, warm-start from mhc-pretrain-20260308b. "
                          "IC50-exact or quantitative affinity measurement profiles.",
        "format": "A",
        "modal_runs": [
            "class1-panel-ic50-exact-warmstart-20260308a",
            "class1-panel-ic50-exact-warmstart-synth-peprank-20260309a",
            "class1-quant-affinity-warmstart-20260308b",
        ],
    },
    {
        "exp_id": "EXP-23",
        "slug": "presto-7allele-broad",
        "title": "Broad 7-Allele Presto Baseline",
        "date": "2026-03-10",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Broad 7-allele baseline using full Presto model (not groove-only) "
                       "with numeric_no_qualitative measurement profile.",
        "training_notes": "7-allele class-I panel, numeric_no_qualitative, qualifier_filter=all. "
                          "Full Presto model with 4.8M params. Warm-start from mhc-pretrain-20260308b.",
        "format": "A",
        "modal_runs": [
            "presto-7allele-broad",
        ],
    },
    {
        "exp_id": "EXP-24",
        "slug": "arch-dimension-sweep",
        "title": "Architecture Dimension Sweep (d_model x layers x heads)",
        "date": "2026-02-26",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Sweep over d_model, n_layers, and n_heads to find optimal "
                       "model dimensions. Two sweep runs with different batch sizes.",
        "training_notes": "Sweep of d_model={224,...}, n_layers={4,5}, n_heads={4,8}. "
                          "20M tokens per candidate, batch 64. Ranking by val_drop_per_epoch.",
        "format": "C",
        "modal_runs": [
            "sweep20m-live-20260226-20260226T171152Z",
            "sweep20m-snfull-bs64-20260226-20260226T193011Z",
        ],
    },
    {
        "exp_id": "EXP-25",
        "slug": "scorepath-bench",
        "title": "Score-Path Comparative Benchmark",
        "date": "2026-03-09",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Comparative benchmark of different score-path configurations "
                       "(e004 vs e006) with and without peptide ranking.",
        "training_notes": "Various score-path configurations on 2-allele panel. "
                          "See individual run summaries for details.",
        "format": "scorepath",
        "modal_runs": [
            "scorepath_bench",
        ],
    },
    {
        "exp_id": "EXP-26",
        "slug": "diagnostic-runs",
        "title": "Diagnostic & Repair Training Runs",
        "date": "2026-03-01",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Diagnostic runs for debugging training issues: reservoir sampling, "
                       "motif scanning, repair iterations, and small-GPU diagnostics.",
        "training_notes": "Diagnostic profile: max_binding=40K, d_model=128, n_layers=2, n_heads=4. "
                          "1-3 epochs, batch 256. Various cap_sampling strategies.",
        "format": "B",
        "modal_runs": [
            "diaggpu-small-20260228",
            "diag-max40k-resv-motif-20260301b",
            "diag-max40k-resv-motif-sw-20260301e",
            "diag-repair-20260306a",
            "diag-repair-20260306a_refetch",
            "diag-repair-20260306c",
        ],
    },
    {
        "exp_id": "EXP-27",
        "slug": "training-trajectories",
        "title": "Training Trajectory Studies",
        "date": "2026-03-05",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Training trajectory analysis with small/filtered data subsets "
                       "to study convergence patterns and loss dynamics.",
        "training_notes": "Canary profile with max_batches=80, max_val_batches=20. "
                          "6 epochs (small) or 1 epoch (full filtered). d_model=128, n_layers=2, n_heads=4.",
        "format": "B",
        "modal_runs": [
            "traj-small-20260305T090347",
            "traj-small-20260305T090854-filt",
            "traj-1ep-20260305T115512-filt",
        ],
    },
    {
        "exp_id": "EXP-28",
        "slug": "refactor-verification",
        "title": "Refactoring Behavior Verification",
        "date": "2026-03-07",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Verification runs after code refactoring to ensure training "
                       "behavior was preserved across commits.",
        "training_notes": "1 epoch each on full profile, batch 128. d_model=128, n_layers=2, n_heads=4. "
                          "Each run tagged with a git commit hash.",
        "format": "B",
        "modal_runs": [
            "refactor-e1-57725c3a",
            "refactor-e1-cc80425a",
            "refactor-e1-d428f09a",
            "refactor-e1-dabcd3da",
        ],
    },
    {
        "exp_id": "EXP-29",
        "slug": "m1-regression-check",
        "title": "Legacy M1 Regression Check",
        "date": "2026-03-10",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Quick and full regression check of the legacy_m1 baseline "
                       "to verify probe discrimination was preserved.",
        "training_notes": "7-allele panel, IC50-exact, warmstart. "
                          "Quick (3 epoch) and full (12 epoch) runs. GrooveTransformerModel.",
        "format": "A",
        "modal_runs": [
            "regcheck",
        ],
    },
    {
        "exp_id": "EXP-30",
        "slug": "mhc-pretraining",
        "title": "MHC Sequence Pretraining Baseline",
        "date": "2026-03-08",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "MHC sequence pretraining on 54K sequences for chain_type, species, "
                       "and class classification. Used as warm-start checkpoint for downstream runs.",
        "training_notes": "54,419 MHC sequences, 48,977 train / 5,442 val. 1 epoch, batch 192. "
                          "d_model=128, n_layers=2, n_heads=4. Classification targets: chain_type, species, class.",
        "format": "A",
        "modal_runs": [
            "mhc-pretrain-20260308b",
        ],
    },
    {
        "exp_id": "EXP-31",
        "slug": "iedb-2k-baseline",
        "title": "Earliest IEDB 2K Training Run",
        "date": "2026-02-16",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "First training run on IEDB data with 2K row cap per assay family. "
                       "Small model (d_model=64, 1 layer, 2 heads) with 10 epochs.",
        "training_notes": "IEDB data, max 2K rows per assay family. 10 epochs, batch 256, lr=1e-4. "
                          "d_model=64, n_layers=1, n_heads=2. Earliest successful training run.",
        "format": "B",
        "modal_runs": [
            "iedb-2k-10ep-20260216i",
        ],
    },
    {
        "exp_id": "EXP-32",
        "slug": "early-baselines",
        "title": "Early Foundation Training Runs",
        "date": "2026-02-26",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Early full-scale and probe-tracking training runs that established "
                       "baseline behavior before the diagnostic/groove refactoring.",
        "training_notes": "Full profile, various batch sizes. d_model=128, n_layers=2, n_heads=4. "
                          "Limited metrics data (empty metrics files in some runs).",
        "format": "B",
        "modal_runs": [
            "full-bs128-fastprobe-perflive-20260228",
            "unified-probe10-20260226b",
        ],
    },
    {
        "exp_id": "EXP-33",
        "slug": "memory-probe",
        "title": "GPU Memory OOM Boundary Exploration",
        "date": "2026-02-26",
        "agent": "Claude Code (claude-opus-4-6)",
        "description": "Batch-size memory profiling to find the OOM boundary. "
                       "Tested batch sizes from 64 to 192.",
        "training_notes": "Memory profiling only (no training metrics). Tested batch sizes: "
                          "64, 96, 112, 128, 160, 176, 180, 192.",
        "format": "logs_only",
        "modal_runs": [
            "memory_probe_20260226",
        ],
    },
]


# ---------------------------------------------------------------------------
# Process one family
# ---------------------------------------------------------------------------


def process_family(spec: dict) -> pd.DataFrame:
    """Process one experiment family and return condition summary."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    dirname = f"{spec['date']}_{timestamp.split('_')[1]}_claude_{spec['slug']}"
    exp_dir = EXPERIMENTS / dirname
    runs_dir = exp_dir / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {spec['exp_id']}: {spec['title']}")
    print(f"  → {exp_dir.name}")

    fmt = spec["format"]

    if fmt == "logs_only":
        # Just copy log references and write README
        (exp_dir / "results").mkdir(parents=True, exist_ok=True)
        for mr_name in spec["modal_runs"]:
            mr_path = MODAL_RUNS / mr_name
            if mr_path.is_dir():
                # Summarize log files
                logs = sorted(mr_path.glob("*.log"))
                log_summary = []
                for log in logs:
                    log_summary.append({
                        "name": log.name,
                        "size_bytes": log.stat().st_size,
                    })
                write_json(runs_dir / f"{mr_name}_logs.json", log_summary)

        write_launch_json(exp_dir, spec)
        write_readme(exp_dir, spec, pd.DataFrame())
        print(f"  Logs-only family collected.")
        return pd.DataFrame()

    if fmt == "scorepath":
        # Special handling for scorepath_bench nested structure
        base = MODAL_RUNS / spec["modal_runs"][0]
        for run_name, run_path in find_scorepath_runs(base):
            print(f"  Loading scorepath run: {run_name}")
            data = load_format_a(run_path)
            normalize_to_aggregate(data, runs_dir / run_name, run_name)

    elif fmt == "C":
        for mr_name in spec["modal_runs"]:
            mr_path = MODAL_RUNS / mr_name
            resolved = resolve_run_dir(mr_path)
            if not resolved.is_dir():
                print(f"  SKIP (not found): {mr_name}")
                continue
            candidates = load_format_c(resolved)
            print(f"  Loaded sweep {mr_name}: {len(candidates)} candidates")
            for cand in candidates:
                rn = cand.get("run_name", "unknown")
                normalize_to_aggregate(cand, runs_dir / rn, rn)

    elif fmt == "A" and len(spec["modal_runs"]) == 1 and spec["modal_runs"][0] == "regcheck":
        # regcheck has subdirectories with summary.json directly
        base = MODAL_RUNS / "regcheck"
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and (sub / "summary.json").exists():
                print(f"  Loading regcheck run: {sub.name}")
                data = load_format_a(sub)
                normalize_to_aggregate(data, runs_dir / sub.name, sub.name)

    elif fmt == "A" and len(spec["modal_runs"]) == 1 and spec["modal_runs"][0] == "presto-7allele-broad":
        # presto-7allele-broad has a dated subdirectory
        base = MODAL_RUNS / "presto-7allele-broad"
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and (sub / "summary.json").exists():
                print(f"  Loading 7allele run: {sub.name}")
                data = load_format_a(sub)
                normalize_to_aggregate(data, runs_dir / sub.name, sub.name)
        # If no subdirs found, try the dir itself
        if not list(runs_dir.iterdir()):
            resolved = resolve_run_dir(base)
            if (resolved / "summary.json").exists():
                data = load_format_a(resolved)
                normalize_to_aggregate(data, runs_dir / base.name, base.name)

    else:
        # Standard A or B format
        for mr_name in spec["modal_runs"]:
            mr_path = MODAL_RUNS / mr_name
            if not mr_path.is_dir():
                print(f"  SKIP (not found): {mr_name}")
                continue
            resolved = resolve_run_dir(mr_path)
            detected = detect_format(resolved)
            print(f"  Loading {mr_name} (format={detected})")

            if detected == "A":
                data = load_format_a(resolved)
            elif detected == "B":
                data = load_format_b(resolved)
            else:
                print(f"  SKIP (unknown format): {mr_name}")
                continue

            normalize_to_aggregate(data, runs_dir / mr_name, mr_name)

    # Aggregate and plot
    summary_df = aggregate_and_plot(exp_dir)
    write_launch_json(exp_dir, spec)
    write_readme(exp_dir, spec, summary_df)

    n_runs = len(list(runs_dir.iterdir())) if runs_dir.exists() else 0
    n_plots = len(list((exp_dir / "results").glob("*.png")))
    print(f"  Done: {n_runs} runs, {n_plots} plots")
    return summary_df


# ---------------------------------------------------------------------------
# Experiment log entry generation
# ---------------------------------------------------------------------------


def generate_log_entry(spec: dict, summary_df: pd.DataFrame, exp_dirname: str) -> str:
    """Generate a markdown experiment log entry."""
    lines = [
        f"## {spec['exp_id']}: {spec['title']}",
        "",
        f"- **Date**: {spec['date']}",
        f"- **Agent**: {spec['agent']}",
        f"- **Dir**: [{exp_dirname}]({exp_dirname}/)",
        f"- **Dataset**: {spec['training_notes'].split('.')[0]}",
        f"- **Training**: {'. '.join(spec['training_notes'].split('.')[1:3]).strip()}",
    ]

    if not summary_df.empty:
        lines.append(f"- **Conditions tested**: {len(summary_df)} runs")
        lines.append("")

        # Build a compact table
        cols = ["label"]
        col_headers = ["Condition"]
        for c, h in [
            ("final_epoch", "Epochs"),
            ("best_val_loss", "Best Val Loss"),
            ("best_val_spearman", "Best Val Spr"),
            ("test_spearman", "Test Spr"),
            ("test_auroc", "Test AUROC"),
            ("test_f1", "Test F1"),
            ("test_rmse_log10", "Test RMSE"),
        ]:
            if c in summary_df.columns and summary_df[c].notna().any():
                cols.append(c)
                col_headers.append(h)

        header = "| " + " | ".join(col_headers) + " |"
        sep = "| " + " | ".join("---" for _ in col_headers) + " |"
        lines.append(header)
        lines.append(sep)
        for _, row in summary_df.head(10).iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float) and pd.notna(v):
                    vals.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
                elif pd.notna(v):
                    vals.append(str(v))
                else:
                    vals.append("")
            lines.append("| " + " | ".join(vals) + " |")

        if len(summary_df) > 10:
            lines.append(f"| ... ({len(summary_df) - 10} more) | | |")
        lines.append("")
    else:
        lines.append("")

    # Winner
    if not summary_df.empty and "best_val_loss" in summary_df.columns:
        best = summary_df.iloc[0]
        lines.append(f"- **Best condition**: `{best.get('label', 'N/A')}`")

    lines.append(f"- **Artifact paths**:")
    lines.append(f"  - experiment dir: `experiments/{exp_dirname}/`")
    lines.append(f"  - condition summary: `experiments/{exp_dirname}/results/condition_summary.csv`")

    lines.append(f"- **Takeaway**: {spec['description'].split('.')[0]}.")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Collecting uncollected modal_runs into experiment registry...")
    print(f"Modal runs dir: {MODAL_RUNS}")
    print(f"Experiments dir: {EXPERIMENTS}")

    log_entries = []
    results = {}

    for spec in FAMILIES:
        summary_df = process_family(spec)
        # Find the directory we just created
        matching = sorted(EXPERIMENTS.glob(f"{spec['date']}*_claude_{spec['slug']}"))
        exp_dirname = matching[-1].name if matching else f"{spec['date']}_0000_claude_{spec['slug']}"

        entry = generate_log_entry(spec, summary_df, exp_dirname)
        log_entries.append(entry)
        results[spec["exp_id"]] = {
            "dirname": exp_dirname,
            "n_runs": len(summary_df) if not summary_df.empty else 0,
        }

    # Append all entries to experiment_log.md
    log_path = EXPERIMENTS / "experiment_log.md"
    separator = "\n\n---\n\n"
    new_section = separator.join([
        "## Retrospective Collection (EXP-21 through EXP-33)\n\n"
        "The following experiments were collected retrospectively from `modal_runs/` "
        "directories that had not been registered in the experiment registry. "
        "Each entry below points to a newly created experiment directory with "
        "normalized data, plots, and reproducibility metadata.\n",
    ] + log_entries)

    with log_path.open("a") as f:
        f.write("\n\n" + new_section)

    print("\n" + "=" * 60)
    print("Collection complete!")
    print(f"Created {len(FAMILIES)} experiment directories")
    print(f"Appended {len(log_entries)} entries to experiment_log.md")
    for exp_id, info in results.items():
        print(f"  {exp_id}: {info['dirname']} ({info['n_runs']} runs)")


if __name__ == "__main__":
    main()
