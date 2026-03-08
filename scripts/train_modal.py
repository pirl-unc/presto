#!/usr/bin/env python
"""Run long Presto training jobs on Modal and persist checkpoints.

The remote function runs one Presto train command (unified or synthetic),
stores checkpoints in a Modal volume,
and returns checkpoint metadata.

Example:
    modal run presto.scripts.train_modal --mode unified --epochs 200
"""

from __future__ import annotations

import csv
import json
import os
import sys
import subprocess
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import modal


DEFAULT_REPO_URL = os.environ.get("PRESTO_MODAL_REPO_URL", "https://github.com/escalante-bio/presto.git")
DEFAULT_REPO_REF = os.environ.get("PRESTO_MODAL_REPO_REF", "main")
DEFAULT_SOURCE_MODE = os.environ.get("PRESTO_MODAL_SOURCE", "local").lower()
DEFAULT_GPU = os.environ.get("PRESTO_MODAL_GPU", "A100")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("PRESTO_MODAL_TIMEOUT_SECONDS", 24 * 60 * 60))
LOCAL_IMAGE_IGNORE = [
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    # Keep local modal source uploads lightweight.
    "artifacts/**",
    "modal_runs/**",
    "site/**",
    "build/**",
    "data/merged_deduped.tsv",
    "data/merged_assays/**",
    "data/mhc_index.csv",
    "presto.pt",
    "*.tar.gz",
    # Exclude heavyweight raw data payloads from image upload.
    "data/10x/**",
    "data/iedb/**",
    "data/imgt/**",
    "data/ipd_mhc/**",
    "data/mcpas/**",
    "data/stcrdab/**",
    "data/vdjdb/**",
]


def _build_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.11")
    if DEFAULT_SOURCE_MODE == "git":
        image = image.apt_install("git").run_commands(
            f"git clone --depth 1 --branch {DEFAULT_REPO_REF} {DEFAULT_REPO_URL} /opt/presto",
        )
    else:
        # Default to local source packaging to avoid Git auth issues in builders.
        image = image.add_local_dir(
            ".",
            remote_path="/opt/presto",
            copy=True,
            ignore=LOCAL_IMAGE_IGNORE,
        )
    return image.run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install /opt/presto matplotlib",
    )


image = _build_image()
app = modal.App("presto-train", image=image)
checkpoints_volume = modal.Volume.from_name("presto-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("presto-data", create_if_missing=True)


@dataclass
class TrainSpec:
    mode: str
    epochs: int
    run_id: str
    checkpoint_name: str
    data_dir: str
    extra_args: Optional[List[str]] = None


def _build_train_command(spec: TrainSpec) -> List[str]:
    run_dir = Path("/checkpoints") / spec.run_id
    checkpoint = run_dir / spec.checkpoint_name

    if spec.mode in {"unified", "iedb"}:
        cmd = [
            "python",
            "-m",
            "presto",
            "train",
            "unified",
            "--epochs",
            str(spec.epochs),
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            spec.data_dir,
        ]
    else:
        cmd = [
            "python",
            "-m",
            "presto",
            "train",
            spec.mode,
            "--epochs",
            str(spec.epochs),
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint),
        ]

    if spec.mode == "synthetic":
        cmd += [
            "--data_dir",
            spec.data_dir,
        ]

    if spec.extra_args:
        cmd += list(spec.extra_args)

    return cmd


def _try_parse_float(raw: str) -> Optional[float]:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _load_metrics_by_step(metrics_csv: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Parse RunLogger metrics into step->split->metric map."""
    by_step: Dict[int, Dict[str, Dict[str, float]]] = {}
    if not metrics_csv.exists() or metrics_csv.stat().st_size == 0:
        return by_step

    with metrics_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step_raw = row.get("step")
            split = str(row.get("split", "")).strip()
            metric = str(row.get("metric", "")).strip()
            value = _try_parse_float(str(row.get("value", "")).strip())
            if step_raw is None or not split or not metric or value is None:
                continue
            step_val = _try_parse_float(step_raw)
            if step_val is None:
                continue
            step = int(step_val)
            by_step.setdefault(step, {}).setdefault(split, {})[metric] = float(value)
    return by_step


def _emit_structured_metric_updates(
    *,
    metrics_csv: Optional[Path],
    emitted_steps: Set[int],
    candidate_tag: Optional[str],
    force: bool = False,
) -> None:
    """Emit structured epoch-level metric logs for Modal observability."""
    if metrics_csv is None:
        return

    by_step = _load_metrics_by_step(metrics_csv)
    for step in sorted(by_step):
        if step in emitted_steps:
            continue
        step_block = by_step[step]
        # Prefer logging once val-loss is present; force can emit trailing data.
        has_val_loss = "val" in step_block and "loss" in step_block["val"]
        if not has_val_loss and not force:
            continue

        payload: Dict[str, Any] = {
            "event": "epoch_metrics",
            "epoch": step,
        }
        if candidate_tag:
            payload["candidate"] = candidate_tag
        if "train" in step_block and "loss" in step_block["train"]:
            payload["train_loss"] = float(step_block["train"]["loss"])
        if "val" in step_block and "loss" in step_block["val"]:
            payload["val_loss"] = float(step_block["val"]["loss"])

        probe = step_block.get("probe", {})
        # Keep payload compact but include key control probes + delta when present.
        selected_probe: Dict[str, float] = {}
        for key, value in probe.items():
            if (
                "a0201_minus_a2402_binding_prob" in key
                or "a0201_minus_a2402_kd_log10" in key
                or key.endswith("hla_a_02_01_binding_prob")
                or key.endswith("hla_a_24_02_binding_prob")
                or key.endswith("hla_a_02_01_kd_nM")
                or key.endswith("hla_a_24_02_kd_nM")
            ):
                selected_probe[key] = float(value)
        if selected_probe:
            payload["probe"] = selected_probe

        print("MODAL_METRIC " + json.dumps(payload, sort_keys=True))
        emitted_steps.add(step)


def _run_command(
    cmd: List[str],
    env: Dict[str, str],
    *,
    metrics_csv: Optional[Path] = None,
    candidate_tag: Optional[str] = None,
    monitor_interval_sec: float = 30.0,
) -> None:
    repo_root = env.get("PRESTO_REPO_ROOT", "/opt/presto")
    emitted_steps: Set[int] = set()
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
    )
    while proc.poll() is None:
        _emit_structured_metric_updates(
            metrics_csv=metrics_csv,
            emitted_steps=emitted_steps,
            candidate_tag=candidate_tag,
            force=False,
        )
        time.sleep(max(float(monitor_interval_sec), 1.0))

    _emit_structured_metric_updates(
        metrics_csv=metrics_csv,
        emitted_steps=emitted_steps,
        candidate_tag=candidate_tag,
        force=True,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _find_flag_value(args: List[str], flag: str) -> Optional[str]:
    """Read a `--flag value` or `--flag=value` string from arg list."""
    for idx, token in enumerate(args):
        if token == flag:
            if idx + 1 < len(args):
                return args[idx + 1]
            return None
        prefix = f"{flag}="
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _detect_repo_root() -> Path:
    """Find the mounted Presto source tree inside the Modal container."""
    candidates = [
        Path("/Users/iskander/code/presto"),
        Path("/root/presto"),
        Path("/opt/presto"),
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "scripts").exists():
            return candidate
    return Path("/opt/presto")


def _prepare_iedb_data(data_dir: str, env: Dict[str, str]) -> str:
    """Download core multi-source data and build an MHC index."""
    from presto.data.downloaders import download_dataset

    data_root = Path(data_dir)
    for dataset_name in (
        "iedb_mhc_ligand",
        "iedb_tcell",
        "iedb_bcell",
        "iedb_cedar_mhc_ligand",
        "iedb_cedar_tcell",
        "iedb_cedar_bcell",
        "vdjdb",
        "10x_pbmc_10k_tcr",
        "imgt_hla",
        "ipd_mhc_nhp",
    ):
        state = download_dataset(
            dataset_name,
            data_root,
            force=False,
            agree_terms=True,
            verbose=False,
        )
        if state.status != "completed":
            raise RuntimeError(f"Failed to download {dataset_name}: {state.error}")

    iedb_dir = Path(data_dir) / "iedb"
    _materialize_zip_member(
        iedb_dir / "mhc_ligand_full_single_file.zip",
        member_suffix="mhc_ligand_full.csv",
        out_csv=iedb_dir / "iedb_mhc_ligand_full.csv",
    )
    _materialize_zip_member(
        iedb_dir / "tcell_full_v3.zip",
        member_suffix="tcell_full_v3.csv",
        out_csv=iedb_dir / "iedb_tcell_full_v3.csv",
    )
    _materialize_zip_member(
        iedb_dir / "cedar_mhc_ligand_full_single_file.zip",
        member_suffix="mhc_ligand_full.csv",
        out_csv=iedb_dir / "cedar_mhc_ligand_full.csv",
    )
    _materialize_zip_member(
        iedb_dir / "cedar_tcell_full_v3.zip",
        member_suffix="tcell_full_v3.csv",
        out_csv=iedb_dir / "cedar_tcell_full_v3.csv",
    )

    index_csv = str(Path(data_dir) / "mhc_index.csv")
    # Build index via library API to avoid CLI-version mismatch on remote images.
    from presto.data.mhc_index import build_mhc_index

    build_mhc_index(
        imgt_fasta=str(Path(data_dir) / "imgt" / "hla_prot.fasta"),
        ipd_mhc_dir=str(Path(data_dir) / "ipd_mhc" / "ipd_mhc_prot.fasta"),
        out_csv=index_csv,
        out_fasta=None,
    )
    return index_csv


def _materialize_zip_member(zip_path: Path, member_suffix: str, out_csv: Path) -> Path:
    """Extract one CSV member from a ZIP into a deterministic output path."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected ZIP not found: {zip_path}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(member_suffix.lower())]
        if not members:
            raise FileNotFoundError(f"No member ending with {member_suffix} in {zip_path}")
        member = sorted(members)[0]
        with zf.open(member, "r") as src, out_csv.open("wb") as dst:
            dst.write(src.read())
    return out_csv


def _write_head_lines(src: Path, dst: Path, max_lines: int) -> Path:
    """Write first `max_lines` lines of a text file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as in_f, dst.open("w", encoding="utf-8") as out_f:
        for i, line in enumerate(in_f):
            if i >= max_lines:
                break
            out_f.write(line)
    return dst


def _prepare_iedb_training_files(
    data_dir: str,
    max_lines: int,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Create bounded-size CSVs for IEDB/CEDAR training on very large exports."""
    iedb_dir = Path(data_dir) / "iedb"
    binding_src = iedb_dir / "iedb_mhc_ligand_full.csv"
    tcell_src = iedb_dir / "iedb_tcell_full_v3.csv"
    cedar_binding_src = iedb_dir / "cedar_mhc_ligand_full.csv"
    cedar_tcell_src = iedb_dir / "cedar_tcell_full_v3.csv"
    if not binding_src.exists():
        binding_src = iedb_dir / "mhc_ligand_full.csv"
    if not tcell_src.exists():
        tcell_src = iedb_dir / "tcell_full_v3.csv"
    binding_subset = iedb_dir / f"mhc_ligand_head_{max_lines}.csv"
    tcell_subset = iedb_dir / f"tcell_head_{max_lines}.csv"
    _write_head_lines(binding_src, binding_subset, max_lines)
    _write_head_lines(tcell_src, tcell_subset, max_lines)

    cedar_binding_subset: Optional[Path] = None
    cedar_tcell_subset: Optional[Path] = None
    if cedar_binding_src.exists():
        cedar_binding_subset = iedb_dir / f"cedar_mhc_ligand_head_{max_lines}.csv"
        _write_head_lines(cedar_binding_src, cedar_binding_subset, max_lines)
    if cedar_tcell_src.exists():
        cedar_tcell_subset = iedb_dir / f"cedar_tcell_head_{max_lines}.csv"
        _write_head_lines(cedar_tcell_src, cedar_tcell_subset, max_lines)

    return (
        str(binding_subset),
        str(tcell_subset),
        str(cedar_binding_subset) if cedar_binding_subset is not None else None,
        str(cedar_tcell_subset) if cedar_tcell_subset is not None else None,
    )


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/data": data_volume,
    },
)
def train_long_run(
    mode: str = "unified",
    epochs: int = 200,
    run_id: Optional[str] = None,
    checkpoint_name: str = "presto.pt",
    data_dir: str = "/data",
    extra_args: str = "",
) -> Dict[str, str]:
    """Run long Presto training on Modal and return checkpoint metadata."""
    if mode not in {"synthetic", "unified", "iedb"}:
        raise ValueError(
            "mode must be one of {'synthetic', 'unified', 'iedb'}"
        )
    resolved_mode = "unified" if mode == "iedb" else mode

    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if isinstance(extra_args, str):
        resolved_extra_args = [arg for arg in extra_args.split(" ") if arg]
    else:
        resolved_extra_args = list(extra_args or [])
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    repo_root = _detect_repo_root()
    env["PRESTO_REPO_ROOT"] = str(repo_root)

    # Prefer mounted source tree over installed site-packages.
    parent = str(repo_root.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{parent}:{existing_pythonpath}" if existing_pythonpath else parent
    if parent not in sys.path:
        sys.path.insert(0, parent)

    if resolved_mode == "unified":
        default_index = str(Path(data_dir) / "mhc_index.csv")
        default_merged = str(Path(data_dir) / "merged_deduped.tsv")
        index_csv = _find_flag_value(resolved_extra_args, "--index-csv")
        merged_tsv = _find_flag_value(resolved_extra_args, "--merged-tsv")

        if index_csv is None and Path(default_index).exists():
            index_csv = default_index
            resolved_extra_args += ["--index-csv", index_csv]
        if merged_tsv is None and Path(default_merged).exists():
            merged_tsv = default_merged
            resolved_extra_args += ["--merged-tsv", merged_tsv]

        if index_csv is None or merged_tsv is None:
            index_csv = _prepare_iedb_data(data_dir, env)
            if _find_flag_value(resolved_extra_args, "--index-csv") is None:
                resolved_extra_args += ["--index-csv", index_csv]

            merged_tsv = str(Path(data_dir) / "merged_deduped.tsv")
            merge_cmd = [
                "python",
                "-m",
                "presto",
                "data",
                "merge",
                "--datadir",
                data_dir,
                "--quiet",
            ]
            _run_command(merge_cmd, env)
            if _find_flag_value(resolved_extra_args, "--merged-tsv") is None:
                resolved_extra_args += ["--merged-tsv", merged_tsv]

    spec = TrainSpec(
        mode=resolved_mode,
        epochs=epochs,
        run_id=resolved_run_id,
        checkpoint_name=checkpoint_name,
        data_dir=data_dir,
        extra_args=resolved_extra_args or None,
    )

    run_dir = Path("/checkpoints") / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_train_command(spec)
    _run_command(cmd, env)
    checkpoints_volume.commit()

    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint was not created: {checkpoint_path}")

    return {
        "run_id": resolved_run_id,
        "mode": resolved_mode,
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "download_hint": (
            "modal volume get presto-checkpoints "
            f"{resolved_run_id}/{checkpoint_name} ./"
        ),
    }


def _write_sweep_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/data": data_volume,
    },
)
def sweep_20m_runs(
    target_params: int = 20_000_000,
    min_params: int = 18_000_000,
    max_params: int = 22_000_000,
    d_models: str = "160,192,224,256,288,320",
    layer_min: int = 2,
    layer_max: int = 6,
    heads: str = "4,8,10,12,16",
    max_candidates: int = 6,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    run_id_prefix: str = "sweep20m",
    data_dir: str = "/data",
    merged_tsv: str = "",
    index_csv: str = "",
    synthetic_negatives: bool = True,
    balanced_batches: bool = True,
    track_probe_affinity: bool = True,
    probe_peptide: str = "SLLQHLIGL",
    probe_alleles: str = "HLA-A*02:01,HLA-A*24:02",
    ranking_metric: str = "val_drop_per_epoch",
    monitor_interval_sec: float = 30.0,
    fail_fast: bool = False,
    extra_args: str = "",
) -> Dict[str, Any]:
    """Run a near-20M architecture sweep on Modal with structured epoch logs.

    Emits `MODAL_METRIC {...}` lines during training so progress is visible in
    Modal logs / dashboard in near-real-time.
    """
    from presto.scripts.sweep_20m_models import (
        generate_candidates,
        parse_int_list,
        read_metric_series,
        summarize_loss_series,
    )

    d_model_values = parse_int_list(d_models)
    head_values = parse_int_list(heads)
    candidates = generate_candidates(
        target_params=int(target_params),
        min_params=int(min_params),
        max_params=int(max_params),
        d_models=d_model_values,
        layer_min=int(layer_min),
        layer_max=int(layer_max),
        heads=head_values,
        max_candidates=int(max_candidates),
    )
    if not candidates:
        raise RuntimeError("No architecture candidates found in requested parameter band.")

    sweep_id = (run_id_prefix or "sweep20m").strip() + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = Path("/checkpoints") / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    repo_root = _detect_repo_root()
    env["PRESTO_REPO_ROOT"] = str(repo_root)

    parent = str(repo_root.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{parent}:{existing_pythonpath}" if existing_pythonpath else parent
    if parent not in sys.path:
        sys.path.insert(0, parent)

    merged_path = str(merged_tsv or "").strip()
    index_path = str(index_csv or "").strip()
    if not merged_path:
        default_merged = Path(data_dir) / "merged_deduped.tsv"
        if default_merged.exists():
            merged_path = str(default_merged)
    if not index_path:
        default_index = Path(data_dir) / "mhc_index.csv"
        if default_index.exists():
            index_path = str(default_index)

    if not merged_path or not index_path:
        resolved_index = _prepare_iedb_data(data_dir, env)
        if not index_path:
            index_path = resolved_index
        if not merged_path:
            merged_path = str(Path(data_dir) / "merged_deduped.tsv")
            merge_cmd = [
                "python",
                "-m",
                "presto",
                "data",
                "merge",
                "--datadir",
                data_dir,
                "--quiet",
            ]
            _run_command(merge_cmd, env)

    ranking_choices = {"val_drop_per_epoch", "val_speed", "train_drop_per_epoch", "train_speed"}
    if ranking_metric not in ranking_choices:
        raise ValueError(
            f"ranking_metric must be one of {sorted(ranking_choices)}, got {ranking_metric!r}"
        )
    result_rows: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_tag = candidate.tag
        candidate_dir = sweep_dir / f"{idx:02d}_{candidate_tag}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = "model.pt"
        spec = TrainSpec(
            mode="unified",
            epochs=int(epochs),
            run_id=str(candidate_dir.relative_to(Path("/checkpoints"))),
            checkpoint_name=checkpoint_name,
            data_dir=data_dir,
            extra_args=[],
        )

        cmd = _build_train_command(spec)
        cmd.extend(
            [
                "--merged-tsv",
                merged_path,
                "--index-csv",
                index_path,
                "--batch_size",
                str(int(batch_size)),
                "--lr",
                str(float(lr)),
                "--weight_decay",
                str(float(weight_decay)),
                "--d_model",
                str(int(candidate.d_model)),
                "--n_layers",
                str(int(candidate.n_layers)),
                "--n_heads",
                str(int(candidate.n_heads)),
            ]
        )
        if not balanced_batches:
            cmd.append("--no-balanced-batches")
        if not synthetic_negatives:
            cmd.extend(
                [
                    "--synthetic-pmhc-negative-ratio",
                    "0",
                    "--synthetic-class-i-no-mhc-beta-negative-ratio",
                    "0",
                    "--synthetic-processing-negative-ratio",
                    "0",
                ]
            )
        if track_probe_affinity:
            cmd.extend(
                [
                    "--track-probe-affinity",
                    "--probe-peptide",
                    str(probe_peptide),
                    "--probe-alleles",
                    str(probe_alleles),
                ]
            )
        else:
            cmd.append("--no-track-probe-affinity")
        if extra_args:
            cmd.extend([arg for arg in extra_args.split(" ") if arg])

        print(
            "MODAL_SWEEP_START "
            + json.dumps(
                {
                    "candidate": candidate_tag,
                    "index": idx,
                    "total_candidates": len(candidates),
                    "n_params": candidate.n_params,
                    "d_model": candidate.d_model,
                    "n_layers": candidate.n_layers,
                    "n_heads": candidate.n_heads,
                    "run_dir": str(candidate_dir),
                },
                sort_keys=True,
            )
        )

        metrics_csv = candidate_dir / "metrics.csv"
        status = "ok"
        return_code = 0
        error_text = ""
        t0 = time.perf_counter()
        try:
            _run_command(
                cmd,
                env,
                metrics_csv=metrics_csv,
                candidate_tag=candidate_tag,
                monitor_interval_sec=float(monitor_interval_sec),
            )
        except subprocess.CalledProcessError as exc:
            status = "failed"
            return_code = int(exc.returncode)
            error_text = str(exc)
            if fail_fast:
                print(f"Fail-fast enabled; stopping sweep after failure in {candidate_tag}")

        duration_sec = float(time.perf_counter() - t0)
        train_series = read_metric_series(metrics_csv, split="train", metric="loss")
        val_series = read_metric_series(metrics_csv, split="val", metric="loss")
        summary: Dict[str, float] = {}
        summary.update(summarize_loss_series(train_series, "train"))
        summary.update(summarize_loss_series(val_series, "val"))
        ranking_value = float(summary.get(ranking_metric, float("nan")))

        row: Dict[str, Any] = {
            "candidate": candidate_tag,
            "status": status,
            "return_code": return_code,
            "error": error_text,
            "duration_sec": duration_sec,
            "run_dir": str(candidate_dir),
            "checkpoint": str(candidate_dir / checkpoint_name),
            "d_model": candidate.d_model,
            "n_layers": candidate.n_layers,
            "n_heads": candidate.n_heads,
            "n_params": candidate.n_params,
            "ranking_metric": ranking_metric,
            "ranking_value": ranking_value,
        }
        row.update(summary)
        result_rows.append(row)
        checkpoints_volume.commit()

        print(
            "MODAL_SWEEP_RESULT "
            + json.dumps(
                {
                    "candidate": candidate_tag,
                    "status": status,
                    "duration_sec": duration_sec,
                    "ranking_metric": ranking_metric,
                    "ranking_value": ranking_value,
                    "val_first_loss": row.get("val_first_loss"),
                    "val_last_loss": row.get("val_last_loss"),
                    "val_drop_per_epoch": row.get("val_drop_per_epoch"),
                },
                sort_keys=True,
            )
        )
        if status != "ok" and fail_fast:
            break

    result_rows_sorted = sorted(
        result_rows,
        key=lambda row: (
            row.get("status") != "ok",
            -float(row.get("ranking_value", float("-inf")))
            if float(row.get("ranking_value", float("nan")))
            == float(row.get("ranking_value", float("nan")))
            else float("inf"),
        ),
    )

    summary_json = sweep_dir / "summary.json"
    summary_json.write_text(json.dumps(result_rows_sorted, indent=2), encoding="utf-8")
    summary_csv = sweep_dir / "summary.csv"
    _write_sweep_summary_csv(summary_csv, result_rows_sorted)
    checkpoints_volume.commit()

    return {
        "sweep_id": sweep_id,
        "sweep_dir": str(sweep_dir),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "ranking_metric": ranking_metric,
        "results": result_rows_sorted,
        "download_hint": f"modal volume get presto-checkpoints {sweep_id} ./",
    }


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/data": data_volume,
    },
)
def probe_training_run(
    batches: int = 500,
    batch_size: int = 128,
    data_dir: str = "/data",
    run_id: Optional[str] = None,
    extra_args: str = "",
) -> Dict[str, str]:
    """Run probe_training.py on Modal with GPU and return artifacts."""
    resolved_run_id = run_id or datetime.now(UTC).strftime("probe_%Y%m%dT%H%M%SZ")
    out_dir = Path("/checkpoints") / resolved_run_id

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    repo_root = _detect_repo_root()
    env["PRESTO_REPO_ROOT"] = str(repo_root)
    parent = str(repo_root.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{parent}:{existing_pythonpath}" if existing_pythonpath else parent

    # Ensure data is available
    merged_tsv = Path(data_dir) / "merged_deduped.tsv"
    index_csv = Path(data_dir) / "mhc_index.csv"
    if not merged_tsv.exists() or not index_csv.exists():
        _prepare_iedb_data(data_dir, env)
        merge_cmd = [
            "python", "-m", "presto", "data", "merge",
            "--datadir", data_dir, "--quiet",
        ]
        _run_command(merge_cmd, env)

    cmd = [
        "python", "-m", "presto.scripts.probe_training",
        "--data-dir", data_dir,
        "--out-dir", str(out_dir),
        "--batches", str(batches),
        "--batch-size", str(batch_size),
    ]
    if extra_args:
        cmd.extend([arg for arg in str(extra_args).split(" ") if arg])
    _run_command(cmd, env)
    checkpoints_volume.commit()

    return {
        "run_id": resolved_run_id,
        "out_dir": str(out_dir),
        "download_hint": f"modal volume get presto-checkpoints {resolved_run_id} ./",
    }


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes={
        "/checkpoints": checkpoints_volume,
        "/data": data_volume,
    },
)
def focused_binding_run(
    epochs: int = 3,
    batch_size: int = 512,
    data_dir: str = "/data",
    run_id: Optional[str] = None,
    extra_args: str = "",
) -> Dict[str, str]:
    """Run focused allele-panel binding diagnostic on Modal."""
    resolved_run_id = run_id or datetime.now(UTC).strftime("focused_binding_%Y%m%dT%H%M%SZ")
    out_dir = Path("/checkpoints") / resolved_run_id

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    repo_root = _detect_repo_root()
    env["PRESTO_REPO_ROOT"] = str(repo_root)
    parent = str(repo_root.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{parent}:{existing_pythonpath}" if existing_pythonpath else parent

    merged_tsv = Path(data_dir) / "merged_deduped.tsv"
    index_csv = Path(data_dir) / "mhc_index.csv"
    if not merged_tsv.exists() or not index_csv.exists():
        _prepare_iedb_data(data_dir, env)
        merge_cmd = [
            "python", "-m", "presto", "data", "merge",
            "--datadir", data_dir, "--quiet",
        ]
        _run_command(merge_cmd, env)

    cmd = [
        "python",
        "-m",
        "presto.scripts.focused_binding_probe",
        "--data-dir",
        data_dir,
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    if extra_args:
        cmd.extend([arg for arg in str(extra_args).split(" ") if arg])
    _run_command(cmd, env)
    checkpoints_volume.commit()

    return {
        "run_id": resolved_run_id,
        "out_dir": str(out_dir),
        "download_hint": f"modal volume get presto-checkpoints {resolved_run_id} ./",
    }


@app.local_entrypoint()
def main(
    mode: str = "unified",
    epochs: int = 200,
    batches: int = 500,
    batch_size: int = 128,
    run_id: str = "",
    checkpoint_name: str = "presto.pt",
    data_dir: str = "/data",
    extra_args: str = "",
    output_manifest: str = "modal_train_result.json",
):
    """Launch one Modal training run and save returned metadata locally.

    Use --mode probe to run the SLLQHLIGL probe training script.
    """
    if mode == "probe":
        result = probe_training_run.remote(
            batches=batches,
            batch_size=batch_size,
            data_dir=data_dir,
            run_id=run_id or None,
            extra_args=extra_args,
        )
    elif mode == "focused_binding":
        result = focused_binding_run.remote(
            epochs=epochs,
            batch_size=batch_size,
            data_dir=data_dir,
            run_id=run_id or None,
            extra_args=extra_args,
        )
    else:
        parsed_extra_args = [arg for arg in extra_args.split(" ") if arg]
        result = train_long_run.remote(
            mode=mode,
            epochs=epochs,
            run_id=run_id or None,
            checkpoint_name=checkpoint_name,
            data_dir=data_dir,
            extra_args=" ".join(parsed_extra_args),
        )
    Path(output_manifest).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
