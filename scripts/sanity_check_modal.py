#!/usr/bin/env python
"""DAG sanity check: ~500 mini-batch training run on Modal A100.

Verifies the model learns pMHC binding motifs after architectural changes.
Uses SLLQHLIGL with HLA-A*02:01 (known binder) vs HLA-A*24:02 (non-binder)
as the primary discrimination probe, evaluated every epoch (~50 batches).

Usage:
    modal run presto/scripts/sanity_check_modal.py

The run produces:
  - probe_affinity_over_epochs.csv  (KD, binding_prob per allele per epoch)
  - probe_affinity_over_epochs.png  (trajectory plot)
  - metrics.csv                     (full training metrics)
  - A PASS/FAIL verdict on A0201 vs A2402 discrimination

Expected result: after ~500 batches the A0201 binding_prob should exceed
A2402 binding_prob, indicating the model has learned allele-specific
binding preferences.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import modal

DEFAULT_GPU = os.environ.get("PRESTO_MODAL_GPU", "A100")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("PRESTO_MODAL_TIMEOUT_SECONDS", 3600))

LOCAL_IMAGE_IGNORE = [
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "artifacts/**",
    "modal_runs/**",
    "site/**",
    "build/**",
    "data/merged_deduped.tsv",
    "data/merged_assays/**",
    "data/mhc_index.csv",
    "presto.pt",
    "*.tar.gz",
    "data/10x/**",
    "data/iedb/**",
    "data/imgt/**",
    "data/ipd_mhc/**",
    "data/mcpas/**",
    "data/stcrdab/**",
    "data/vdjdb/**",
]


def _build_image() -> modal.Image:
    source_mode = os.environ.get("PRESTO_MODAL_SOURCE", "local").lower()
    image = modal.Image.debian_slim(python_version="3.11")
    if source_mode == "git":
        repo_url = os.environ.get(
            "PRESTO_MODAL_REPO_URL", "https://github.com/escalante-bio/presto.git"
        )
        repo_ref = os.environ.get("PRESTO_MODAL_REPO_REF", "main")
        image = image.apt_install("git").run_commands(
            f"git clone --depth 1 --branch {repo_ref} {repo_url} /opt/presto",
        )
    else:
        image = image.add_local_dir(
            ".",
            remote_path="/opt/presto",
            copy=True,
            ignore=LOCAL_IMAGE_IGNORE,
        )
    return image.run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install /opt/presto",
    )


image = _build_image()
app = modal.App("presto-sanity-check", image=image)
checkpoints_volume = modal.Volume.from_name("presto-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("presto-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Sanity check configuration
# ---------------------------------------------------------------------------
# Target: ~500 mini-batches with probe eval every ~50 batches.
# 10 epochs * ~50 batches/epoch = ~500 batches.
# With batch_size=64 and ~3200 training samples we get ~50 batches/epoch.
SANITY_CHECK_ARGS = [
    "--profile", "diagnostic",
    "--epochs", "10",
    "--batch_size", "64",
    "--d_model", "128",
    "--n_layers", "2",
    "--n_heads", "4",
    # Data caps: ~3200 total samples → ~50 batches/epoch
    "--max-binding", "1500",
    "--max-elution", "1000",
    "--max-tcell", "500",
    "--max-processing", "200",
    "--max-kinetics", "0",
    "--max-stability", "0",
    "--max-vdjdb", "0",
    "--cap-sampling", "head",
    # Probe: SLLQHLIGL A0201 vs A2402
    "--track-probe-affinity",
    "--probe-peptide", "SLLQHLIGL",
    "--probe-alleles", "HLA-A*02:01,HLA-A*24:02",
    "--no-track-probe-motif-scan",
    # Skip expensive diagnostics for speed
    "--no-track-pmhc-flow",
    "--no-track-output-latent-stats",
    # Reproducibility
    "--seed", "42",
    "--filter-unresolved-mhc",
    "--strict-mhc-resolution",
    "--balanced-batches",
    "--lr", "1e-4",
    "--weight_decay", "0.01",
]


def _detect_repo_root() -> Path:
    candidates = [
        Path("/Users/iskander/code/presto"),
        Path("/root/presto"),
        Path("/opt/presto"),
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "scripts").exists():
            return candidate
    return Path("/opt/presto")


def _try_parse_float(raw: str) -> Optional[float]:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _load_probe_results(metrics_csv: Path) -> Dict[int, Dict[str, float]]:
    """Extract probe metrics by epoch from metrics.csv."""
    results: Dict[int, Dict[str, float]] = {}
    if not metrics_csv.exists():
        return results
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split", "")).strip()
            if split != "probe":
                continue
            step = _try_parse_float(str(row.get("step", "")))
            metric = str(row.get("metric", "")).strip()
            value = _try_parse_float(str(row.get("value", "")))
            if step is None or value is None or not metric:
                continue
            results.setdefault(int(step), {})[metric] = value
    return results


def _emit_metric_updates(
    metrics_csv: Optional[Path],
    emitted_steps: Set[int],
) -> None:
    if metrics_csv is None or not metrics_csv.exists():
        return
    by_step: Dict[int, Dict[str, Dict[str, float]]] = {}
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_raw = _try_parse_float(str(row.get("step", "")))
            if step_raw is None:
                continue
            step = int(step_raw)
            split = str(row.get("split", "")).strip()
            metric = str(row.get("metric", "")).strip()
            value = _try_parse_float(str(row.get("value", "")))
            if not split or not metric or value is None:
                continue
            by_step.setdefault(step, {}).setdefault(split, {})[metric] = value
    for step in sorted(by_step):
        if step in emitted_steps:
            continue
        step_block = by_step[step]
        has_val = "val" in step_block and "loss" in step_block["val"]
        if not has_val:
            continue
        payload: Dict[str, Any] = {"event": "sanity_epoch", "epoch": step}
        if "train" in step_block and "loss" in step_block["train"]:
            payload["train_loss"] = step_block["train"]["loss"]
        if has_val:
            payload["val_loss"] = step_block["val"]["loss"]
        probe = step_block.get("probe", {})
        for key, value in probe.items():
            if "a0201" in key or "a2402" in key:
                payload[key] = value
        print("MODAL_METRIC " + json.dumps(payload, sort_keys=True))
        emitted_steps.add(step)


def _evaluate_verdict(metrics_csv: Path) -> Dict[str, Any]:
    """Check whether A0201 vs A2402 discrimination was learned."""
    probe = _load_probe_results(metrics_csv)
    if not probe:
        return {"pass": False, "reason": "no probe metrics found"}

    last_epoch = max(probe.keys())
    last = probe[last_epoch]

    a0201_prob = last.get("probe_sllqhligl_hla_a_02_01_binding_prob")
    a2402_prob = last.get("probe_sllqhligl_hla_a_24_02_binding_prob")
    delta_kd = last.get("probe_sllqhligl_a0201_minus_a2402_kd_log10")
    delta_prob = last.get("probe_sllqhligl_a0201_minus_a2402_binding_prob")

    verdict: Dict[str, Any] = {
        "last_epoch": last_epoch,
        "a0201_binding_prob": a0201_prob,
        "a2402_binding_prob": a2402_prob,
        "delta_kd_log10": delta_kd,
        "delta_binding_prob": delta_prob,
    }

    # SLLQHLIGL is a known HLA-A*02:01 binder.
    # Pass criteria: A0201 binding_prob > A2402 binding_prob (any margin).
    if a0201_prob is not None and a2402_prob is not None:
        verdict["pass"] = a0201_prob > a2402_prob
        if verdict["pass"]:
            verdict["reason"] = (
                f"A0201 binding_prob ({a0201_prob:.4f}) > "
                f"A2402 ({a2402_prob:.4f}) — motif discrimination learned"
            )
        else:
            verdict["reason"] = (
                f"A0201 binding_prob ({a0201_prob:.4f}) <= "
                f"A2402 ({a2402_prob:.4f}) — no discrimination yet"
            )
    else:
        verdict["pass"] = False
        verdict["reason"] = "probe binding_prob values missing"

    return verdict


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/data": data_volume,
    },
)
def sanity_check(
    data_dir: str = "/data",
    extra_args: str = "",
) -> Dict[str, Any]:
    """Run ~500 mini-batch sanity check and evaluate SLLQHLIGL probe.

    Returns dict with training metrics, probe trajectory, and pass/fail verdict.
    """
    run_id = "sanity-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("/checkpoints") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "presto.pt"

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    repo_root = _detect_repo_root()
    env["PRESTO_REPO_ROOT"] = str(repo_root)
    parent = str(repo_root.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{parent}:{existing}" if existing else parent
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Build data (MHC index + merged TSV)
    default_index = str(Path(data_dir) / "mhc_index.csv")
    default_merged = str(Path(data_dir) / "merged_deduped.tsv")

    resolved_args = list(SANITY_CHECK_ARGS)
    if extra_args:
        resolved_args += [a for a in extra_args.split(" ") if a]

    resolved_args += ["--run-dir", str(run_dir)]
    resolved_args += ["--checkpoint", str(checkpoint)]

    if Path(default_index).exists():
        resolved_args += ["--index-csv", default_index]
    if Path(default_merged).exists():
        resolved_args += ["--merged-tsv", default_merged]

    if not Path(default_index).exists() or not Path(default_merged).exists():
        from presto.scripts.train_modal import _prepare_iedb_data
        index_csv = _prepare_iedb_data(data_dir, env)
        resolved_args += ["--index-csv", index_csv]

        merged_tsv = str(Path(data_dir) / "merged_deduped.tsv")
        merge_cmd = [
            "python", "-m", "presto", "data", "merge",
            "--datadir", data_dir, "--quiet",
        ]
        subprocess.run(merge_cmd, cwd=str(repo_root), env=env, check=True)
        resolved_args += ["--merged-tsv", merged_tsv]

    cmd = [
        "python", "-m", "presto", "train", "unified",
    ] + resolved_args

    print(f"=== Sanity Check: {run_id} ===")
    print(f"Command: {' '.join(cmd)}")
    print()

    metrics_csv = run_dir / "metrics.csv"
    emitted_steps: Set[int] = set()

    proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env)
    while proc.poll() is None:
        _emit_metric_updates(metrics_csv, emitted_steps)
        time.sleep(15.0)
    _emit_metric_updates(metrics_csv, emitted_steps)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    checkpoints_volume.commit()

    # Evaluate probe verdict
    verdict = _evaluate_verdict(metrics_csv)

    print()
    print("=" * 60)
    print("SANITY CHECK VERDICT")
    print("=" * 60)
    print(json.dumps(verdict, indent=2))
    if verdict.get("pass"):
        print("\n  PASS — model learns allele-specific binding discrimination")
    else:
        print(f"\n  FAIL — {verdict.get('reason', 'unknown')}")
    print("=" * 60)

    # Probe trajectory
    probe_results = _load_probe_results(metrics_csv)
    trajectory = []
    for epoch in sorted(probe_results):
        row = {"epoch": epoch}
        row.update(probe_results[epoch])
        trajectory.append(row)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "verdict": verdict,
        "probe_trajectory": trajectory,
        "download_hint": f"modal volume get presto-checkpoints {run_id}/ ./sanity_{run_id}/",
    }


@app.local_entrypoint()
def main():
    result = sanity_check.remote()
    print()
    print(json.dumps(result, indent=2, default=str))
