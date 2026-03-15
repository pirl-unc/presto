#!/usr/bin/env python
"""Launch matched positional-composition bakeoff for Presto and groove controls."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from experiment_registry import default_agent_label, initialize_experiment_dir


DEFAULT_ALLELES = (
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02",
)
DEFAULT_PROBES = ("SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI")
DEFAULT_WARM_START = "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
APP_ID_PATTERN = re.compile(r"\bap-[A-Za-z0-9]+\b")

POSITION_MODES: Tuple[Tuple[str, str], ...] = (
    ("start_only", "start_only"),
    ("end_only", "end_only"),
    ("start_plus_end", "start_plus_end"),
    ("concat_start_end", "concat_start_end"),
    ("concat_start_end_frac", "concat_start_end_frac"),
    ("mlp_start_end", "mlp_start_end"),
    ("mlp_start_end_frac", "mlp_start_end_frac"),
    ("triple_baseline", "triple_baseline"),
)


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    family: str
    description: str
    extra_args: Tuple[str, ...]


def _make_designs() -> Tuple[DesignSpec, ...]:
    designs: List[DesignSpec] = []
    for idx, (pep_mode, groove_mode) in enumerate(POSITION_MODES):
        designs.append(
            DesignSpec(
                design_id=f"P{idx:02d}",
                family="presto",
                description=f"Presto P03-style, peptide={pep_mode}, groove={groove_mode}",
                extra_args=(
                    "--d-model", "128",
                    "--peptide-pos-mode", pep_mode,
                    "--groove-pos-mode", groove_mode,
                    "--binding-core-lengths", "8,9,10,11",
                    "--binding-core-refinement", "shared",
                    "--affinity-assay-mode", "legacy",
                    "--affinity-assay-residual-mode", "shared_base_segment_residual",
                    "--binding-kinetic-input-mode", "affinity_vec",
                    "--binding-direct-segment-mode", "off",
                    "--affinity-target-encoding", "log10",
                    "--max-affinity-nm", "50000",
                ),
            )
        )
        designs.append(
            DesignSpec(
                design_id=f"G{idx:02d}",
                family="groove_transformer",
                description=f"G1-style groove transformer, peptide={pep_mode}, groove={groove_mode}",
                extra_args=(
                    "--model-variant", "transformer",
                    "--embed-dim", "128",
                    "--hidden-dim", "256",
                    "--n-heads", "4",
                    "--n-layers", "2",
                    "--peptide-pos-mode", pep_mode,
                    "--groove-pos-mode", groove_mode,
                ),
            )
        )
    return tuple(designs)


DESIGNS = _make_designs()


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260311a"


def _common_args(*, alleles: Sequence[str], probes: Sequence[str]) -> List[str]:
    return [
        "--alleles", ",".join(alleles),
        "--probe-peptide", probes[0],
        "--extra-probe-peptides", ",".join(probes[1:]),
        "--measurement-profile", "numeric_no_qualitative",
        "--qualifier-filter", "all",
        "--probe-plot-frequency", "off",
        "--no-synthetic-negatives",
        "--binding-contrastive-weight", "0",
        "--binding-peptide-contrastive-weight", "0",
    ]


def _build_extra_args(
    *,
    design: DesignSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
) -> List[str]:
    args = _common_args(alleles=alleles, probes=probes)
    args.extend(["--design-id", design.design_id])
    if design.family == "presto":
        args.extend(["--affinity-loss-mode", "full", "--init-checkpoint", warm_start])
    args.extend(list(design.extra_args))
    return args


def _launch_design(
    *,
    design: DesignSpec,
    alleles: Sequence[str],
    probes: Sequence[str],
    warm_start: str,
    epochs: int,
    presto_batch_size: int,
    groove_batch_size: int,
    prefix: str,
    out_dir: Path,
    launch_timeout_s: float,
    launch_retries: int,
    launch_backoff_s: float,
) -> Dict[str, Any]:
    run_id = _run_id(prefix, design.design_id)
    batch_size = groove_batch_size if design.family == "groove_transformer" else presto_batch_size
    extra_args = _build_extra_args(
        design=design,
        alleles=alleles,
        probes=probes,
        warm_start=warm_start,
    )
    modal_target = (
        "scripts/train_modal.py::groove_baseline_run"
        if design.family == "groove_transformer"
        else "scripts/train_modal.py::focused_binding_run"
    )
    cmd = [
        "modal",
        "run",
        "--detach",
        modal_target,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--run-id",
        run_id,
        "--extra-args",
        " ".join(extra_args),
    ]
    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = max(int(launch_retries), 1)
    errors: List[str] = []
    output = ""
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=float(launch_timeout_s),
                check=True,
            )
            output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            log_path.write_text(output, encoding="utf-8")
            match = APP_ID_PATTERN.search(output)
            if match is None:
                raise RuntimeError(
                    f"Detached Modal launch for {run_id} did not emit an app id:\n{output}"
                )
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
            partial = ""
            if isinstance(exc, subprocess.TimeoutExpired):
                partial = "\n".join(
                    str(part, "utf-8", errors="replace") if isinstance(part, bytes) else str(part)
                    for part in (exc.stdout, exc.stderr)
                    if part
                )
            elif isinstance(exc, subprocess.CalledProcessError):
                partial = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
            elif isinstance(exc, RuntimeError):
                partial = str(exc)
            if partial:
                log_path.write_text(partial, encoding="utf-8")
            errors.append(f"[attempt {attempt}] {type(exc).__name__}: {partial or exc}")
            if attempt == attempts:
                raise RuntimeError(
                    f"Detached Modal launch failed for {run_id} after {attempts} attempts:\n"
                    + "\n\n".join(errors)
                ) from exc
            time.sleep(float(launch_backoff_s) * attempt)
    else:
        raise RuntimeError(f"Detached Modal launch failed for {run_id} without attempts")

    match = APP_ID_PATTERN.search(output)
    if match is None:
        raise RuntimeError(f"Detached Modal launch for {run_id} did not emit an app id:\n{output}")
    return {
        "run_id": run_id,
        "design_id": design.design_id,
        "family": design.family,
        "description": design.description,
        "app_id": match.group(0),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "extra_args": extra_args,
        "command": cmd,
    }


def _write_variants(out_dir: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Positional Composition Variants",
        "",
        "| design | family | app id | run id | description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        lines.append(
            f"| `{run['design_id']}` | `{run['family']}` | `{run['app_id']}` | `{run['run_id']}` | {run['description']} |"
        )
    (out_dir / "variants.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_designs(design_ids: Sequence[str]) -> Tuple[DesignSpec, ...]:
    if not design_ids:
        return DESIGNS
    wanted = {str(design_id).strip().upper() for design_id in design_ids if str(design_id).strip()}
    selected = tuple(design for design in DESIGNS if design.design_id.upper() in wanted)
    missing = sorted(wanted.difference({design.design_id.upper() for design in selected}))
    if missing:
        raise SystemExit(f"Unknown design ids: {', '.join(missing)}")
    return selected


def _merge_runs(existing: Sequence[Mapping[str, Any]], new_runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for run in existing:
        merged[str(run["design_id"])] = dict(run)
    for run in new_runs:
        merged[str(run["design_id"])] = dict(run)
    return [merged[key] for key in sorted(merged.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch matched positional-composition bakeoff.")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--slug", type=str, default="positional-composition-round1")
    parser.add_argument("--title", type=str, default="Positional Composition Round 1")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--presto-batch-size", type=int, default=140)
    parser.add_argument("--groove-batch-size", type=int, default=256)
    parser.add_argument("--prefix", type=str, default="poscomp-r1")
    parser.add_argument("--launch-timeout-s", type=float, default=180.0)
    parser.add_argument("--launch-retries", type=int, default=3)
    parser.add_argument("--launch-backoff-s", type=float, default=20.0)
    parser.add_argument("--design-ids", type=str, default="")
    args = parser.parse_args()

    alleles = tuple(part.strip() for part in str(args.alleles).split(",") if part.strip())
    probes = tuple(part.strip() for part in str(args.probes).split(",") if part.strip())
    selected = _selected_designs(tuple(part.strip() for part in str(args.design_ids).split(",") if part.strip()))

    metadata = {
        "dataset_contract": {
            "source": "data/merged_deduped.tsv",
            "alleles": list(alleles),
            "measurement_profile": "numeric_no_qualitative",
            "qualifier_filter": "all",
            "included_assays": ["IC50", "KD", "KD (~IC50)", "KD (~EC50)", "EC50"],
            "excluded_assays": ["qualitative binding"],
            "expected_split": {"train_rows": 32855, "val_rows": 8194},
        },
        "training": {
            "epochs": int(args.epochs),
            "presto_batch_size": int(args.presto_batch_size),
            "groove_batch_size": int(args.groove_batch_size),
            "warm_start": str(args.warm_start),
            "synthetics": False,
            "ranking": False,
        },
        "tested": [
            {"design_id": d.design_id, "family": d.family, "description": d.description}
            for d in selected
        ],
    }
    out_dir = initialize_experiment_dir(
        out_dir=str(args.out_dir),
        slug=str(args.slug),
        title=str(args.title),
        source_script="scripts/benchmark_positional_composition.py",
        agent_label=str(args.agent_label),
        metadata=metadata,
    )

    runs = [
        _launch_design(
            design=design,
            alleles=alleles,
            probes=probes,
            warm_start=str(args.warm_start),
            epochs=int(args.epochs),
            presto_batch_size=int(args.presto_batch_size),
            groove_batch_size=int(args.groove_batch_size),
            prefix=str(args.prefix),
            out_dir=out_dir,
            launch_timeout_s=float(args.launch_timeout_s),
            launch_retries=int(args.launch_retries),
            launch_backoff_s=float(args.launch_backoff_s),
        )
        for design in selected
    ]
    manifest_path = out_dir / "manifest.json"
    existing_runs: List[Mapping[str, Any]] = []
    if manifest_path.exists():
        existing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_runs = list(existing_payload.get("runs", []))
    merged_runs = _merge_runs(existing_runs, runs)
    manifest_path.write_text(
        json.dumps({"runs": merged_runs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_variants(out_dir, merged_runs)


if __name__ == "__main__":
    main()
