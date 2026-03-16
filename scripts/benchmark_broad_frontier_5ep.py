#!/usr/bin/env python
"""Launch the 5-epoch broad-contract frontier bakeoff on Modal."""

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


@dataclass(frozen=True)
class DesignSpec:
    design_id: str
    family: str  # presto | groove
    description: str
    extra_args: Tuple[str, ...]
    batch_size: int
    warm_start: bool = False


def _presto_common(*, d_model: int, peptide_pos: str, groove_pos: str, residual: str, kinetic_input: str = "affinity_vec") -> Tuple[str, ...]:
    return (
        "--d-model", str(d_model),
        "--peptide-pos-mode", peptide_pos,
        "--groove-pos-mode", groove_pos,
        "--binding-core-lengths", "8,9,10,11",
        "--binding-core-refinement", "shared",
        "--binding-kinetic-input-mode", kinetic_input,
        "--affinity-target-encoding", "log10",
        "--max-affinity-nm", "50000",
        "--affinity-assay-residual-mode", residual,
    )


def _groove_common(*, model_variant: str = "transformer", embed_dim: int = 128, hidden_dim: int = 256, n_layers: int = 2, n_heads: int = 4, peptide_pos: str = "triple_baseline", groove_pos: str = "triple_baseline") -> Tuple[str, ...]:
    args: List[str] = [
        "--model-variant", model_variant,
        "--embed-dim", str(embed_dim),
        "--hidden-dim", str(hidden_dim),
        "--peptide-pos-mode", peptide_pos,
        "--groove-pos-mode", groove_pos,
    ]
    if model_variant == "transformer":
        args.extend(["--n-layers", str(n_layers), "--n-heads", str(n_heads)])
    return tuple(args)


DESIGNS: Tuple[DesignSpec, ...] = (
    # Directness round-2 entries recorded in the canonical log ranking.
    DesignSpec("DP00", "presto", "Directness P00 legacy triple/triple", _presto_common(d_model=256, peptide_pos="triple", groove_pos="triple", residual="legacy"), 140, True),
    DesignSpec("DP01", "presto", "Directness P01 shared_base_segment_residual triple/triple", _presto_common(d_model=256, peptide_pos="triple", groove_pos="triple", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("DP05", "presto", "Directness P05 shared_base_segment_residual triple_plus_abs/triple", _presto_common(d_model=256, peptide_pos="triple_plus_abs", groove_pos="triple", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("DG1", "groove", "Directness G1 groove transformer", _groove_common(model_variant="transformer", embed_dim=128, hidden_dim=256, n_layers=2, n_heads=4, peptide_pos="triple_baseline", groove_pos="triple_baseline"), 256, False),
    # Positional composition P00..P07
    DesignSpec("PP00", "presto", "Positional P00 start_only", _presto_common(d_model=128, peptide_pos="start_only", groove_pos="start_only", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("PP01", "presto", "Positional P01 end_only", _presto_common(d_model=128, peptide_pos="end_only", groove_pos="end_only", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("PP02", "presto", "Positional P02 start_plus_end", _presto_common(d_model=128, peptide_pos="start_plus_end", groove_pos="start_plus_end", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("PP03", "presto", "Positional P03 concat(start,end)", _presto_common(d_model=128, peptide_pos="concat_start_end", groove_pos="concat_start_end", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("PP04", "presto", "Positional P04 concat(start,end,frac)", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("PP05", "presto", "Positional P05 mlp(concat(start,end))", _presto_common(d_model=128, peptide_pos="mlp_start_end", groove_pos="mlp_start_end", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("PP06", "presto", "Positional P06 mlp(concat(start,end,frac))", _presto_common(d_model=128, peptide_pos="mlp_start_end_frac", groove_pos="mlp_start_end_frac", residual="shared_base_segment_residual"), 140, True),
    DesignSpec("PP07", "presto", "Positional P07 triple_baseline", _presto_common(d_model=128, peptide_pos="triple_baseline", groove_pos="triple_baseline", residual="shared_base_segment_residual"), 140, True),
    # Positional groove controls G00..G07
    DesignSpec("PG00", "groove", "Positional G00 start_only", _groove_common(peptide_pos="start_only", groove_pos="start_only"), 256, False),
    DesignSpec("PG01", "groove", "Positional G01 end_only", _groove_common(peptide_pos="end_only", groove_pos="end_only"), 256, False),
    DesignSpec("PG02", "groove", "Positional G02 start_plus_end", _groove_common(peptide_pos="start_plus_end", groove_pos="start_plus_end"), 256, False),
    DesignSpec("PG03", "groove", "Positional G03 concat(start,end)", _groove_common(peptide_pos="concat_start_end", groove_pos="concat_start_end"), 256, False),
    DesignSpec("PG04", "groove", "Positional G04 concat(start,end,frac)", _groove_common(peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac"), 256, False),
    DesignSpec("PG05", "groove", "Positional G05 mlp(concat(start,end))", _groove_common(peptide_pos="mlp_start_end", groove_pos="mlp_start_end"), 256, False),
    DesignSpec("PG06", "groove", "Positional G06 mlp(concat(start,end,frac))", _groove_common(peptide_pos="mlp_start_end_frac", groove_pos="mlp_start_end_frac"), 256, False),
    DesignSpec("PG07", "groove", "Positional G07 triple_baseline", _groove_common(peptide_pos="triple_baseline", groove_pos="triple_baseline"), 256, False),
    # Assay-head A00..A07 on top of P04 positional base.
    DesignSpec("A00", "presto", "Assay A00 pooled_single_output merged_kd", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="pooled_single_output") + ("--kd-grouping-mode", "merged_kd"), 140, True),
    DesignSpec("A01", "presto", "Assay A01 pooled_single_output split_kd_proxy", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="pooled_single_output") + ("--kd-grouping-mode", "split_kd_proxy"), 140, True),
    DesignSpec("A02", "presto", "Assay A02 shared_base_segment_residual merged_kd", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="shared_base_segment_residual") + ("--kd-grouping-mode", "merged_kd"), 140, True),
    DesignSpec("A03", "presto", "Assay A03 shared_base_segment_residual split_kd_proxy", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="shared_base_segment_residual") + ("--kd-grouping-mode", "split_kd_proxy"), 140, True),
    DesignSpec("A04", "presto", "Assay A04 factorized_context merged_kd", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="shared_base_factorized_context_residual") + ("--kd-grouping-mode", "merged_kd"), 140, True),
    DesignSpec("A05", "presto", "Assay A05 factorized_context split_kd_proxy", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="shared_base_factorized_context_residual") + ("--kd-grouping-mode", "split_kd_proxy"), 140, True),
    DesignSpec("A06", "presto", "Assay A06 factorized_context_plus_segment merged_kd", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="shared_base_factorized_context_plus_segment_residual") + ("--kd-grouping-mode", "merged_kd"), 140, True),
    DesignSpec("A07", "presto", "Assay A07 factorized_context_plus_segment split_kd_proxy", _presto_common(d_model=128, peptide_pos="concat_start_end_frac", groove_pos="concat_start_end_frac", residual="shared_base_factorized_context_plus_segment_residual") + ("--kd-grouping-mode", "split_kd_proxy"), 140, True),
)


def _run_id(prefix: str, design_id: str) -> str:
    return f"{prefix}-{design_id.lower()}-20260312a"


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


def _build_extra_args(design: DesignSpec, alleles: Sequence[str], probes: Sequence[str], warm_start: str) -> List[str]:
    args = _common_args(alleles=alleles, probes=probes)
    args.extend(["--design-id", design.design_id])
    if design.family == "presto":
        args.extend(["--affinity-loss-mode", "assay_heads_only"])
        if design.warm_start:
            args.extend(["--init-checkpoint", warm_start])
    args.extend(list(design.extra_args))
    return args


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _write_manifest(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(json.dumps(list(runs), indent=2, sort_keys=True), encoding="utf-8")


def _write_variants(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Broad Frontier 5-epoch Variants",
        "",
        "| design | family | app id | run id | description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        lines.append(f"| `{run['design_id']}` | `{run['family']}` | `{run.get('app_id','')}` | `{run['run_id']}` | {run['description']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected(design_ids: Sequence[str]) -> Tuple[DesignSpec, ...]:
    if not design_ids:
        return DESIGNS
    wanted = {d.strip().upper() for d in design_ids if d.strip()}
    selected = tuple(d for d in DESIGNS if d.design_id.upper() in wanted)
    missing = sorted(wanted - {d.design_id.upper() for d in selected})
    if missing:
        raise SystemExit(f"Unknown design ids: {', '.join(missing)}")
    return selected


def _launch(design: DesignSpec, *, alleles: Sequence[str], probes: Sequence[str], warm_start: str, epochs: int, prefix: str, out_dir: Path, timeout_s: float, retries: int) -> Dict[str, Any]:
    run_id = _run_id(prefix, design.design_id)
    extra_args = _build_extra_args(design, alleles, probes, warm_start)
    target = "scripts/train_modal.py::groove_baseline_run" if design.family == "groove" else "scripts/train_modal.py::focused_binding_run"
    cmd = [
        "modal", "run", "--detach", target,
        "--epochs", str(epochs),
        "--batch-size", str(design.batch_size),
        "--run-id", run_id,
        "--extra-args", " ".join(extra_args),
    ]
    log_path = out_dir / "launch_logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output = ""
    errors: List[str] = []
    for attempt in range(1, retries + 1):
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                proc = subprocess.Popen(
                    cmd,
                    text=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            start = time.time()
            app_id = ""
            while True:
                if time.time() - start > timeout_s:
                    existing = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
                    raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s, output=existing)
                if log_path.exists():
                    output = log_path.read_text(encoding="utf-8", errors="replace")
                    if not app_id:
                        m = APP_ID_PATTERN.search(output)
                        if m:
                            app_id = m.group(0)
                            break
                if proc.poll() is not None and not app_id:
                    break
                time.sleep(0.5)
            output = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else output
            if not app_id:
                raise RuntimeError(f"No app id in detached output for {run_id}:\n{output}")
            return {
                "design_id": design.design_id,
                "family": design.family,
                "description": design.description,
                "run_id": run_id,
                "app_id": app_id,
                "batch_size": design.batch_size,
                "epochs": epochs,
                "extra_args": extra_args,
                "launch_log": str(log_path),
            }
        except Exception as exc:
            msg = str(exc)
            if isinstance(exc, subprocess.TimeoutExpired):
                partial = "\n".join(
                    (part.decode("utf-8", errors="replace") if isinstance(part, bytes) else str(part))
                    for part in (exc.stdout, exc.stderr) if part
                )
                msg = partial or msg
            log_path.write_text(msg, encoding="utf-8")
            errors.append(f"attempt {attempt}: {msg}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to launch {run_id}:\n" + "\n\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch 5-epoch broad frontier bakeoff")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warm-start", type=str, default=DEFAULT_WARM_START)
    parser.add_argument("--alleles", type=str, default=",".join(DEFAULT_ALLELES))
    parser.add_argument("--probes", type=str, default=",".join(DEFAULT_PROBES))
    parser.add_argument("--prefix", type=str, default="broad-frontier-r1")
    parser.add_argument("--agent-label", type=str, default=default_agent_label())
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--design", action="append", default=[])
    parser.add_argument("--launch-timeout-s", type=float, default=240.0)
    parser.add_argument("--launch-retries", type=int, default=3)
    args = parser.parse_args()

    alleles = [a.strip() for a in args.alleles.split(",") if a.strip()]
    probes = [p.strip().upper() for p in args.probes.split(",") if p.strip()]
    selected = _selected(args.design)
    out_dir = initialize_experiment_dir(
        out_dir=args.out_dir,
        slug="broad-frontier-5ep-round1",
        title="Broad Frontier 5-Epoch Bakeoff",
        source_script="scripts/benchmark_broad_frontier_5ep.py",
        agent_label=args.agent_label,
        metadata={
            "dataset_contract": {
                "alleles": alleles,
                "measurement_profile": "numeric_no_qualitative",
                "broad_numeric_families": ["IC50", "KD", "KD (~IC50)", "KD (~EC50)", "EC50"],
                "qualifier_filter": "all",
                "probe_peptides": probes,
            },
            "training": {
                "epochs": args.epochs,
                "synthetic_negatives": False,
                "ranking_losses": False,
                "warm_start": args.warm_start,
            },
            "tested": [
                {"design_id": d.design_id, "family": d.family, "description": d.description, "batch_size": d.batch_size}
                for d in selected
            ],
        },
    )
    manifest_path = out_dir / "manifest.json"
    variants_path = out_dir / "variants.md"
    launched = _load_manifest(manifest_path)
    seen = {r.get("design_id") for r in launched}
    for design in selected:
        if design.design_id in seen:
            continue
        run = _launch(design, alleles=alleles, probes=probes, warm_start=args.warm_start, epochs=args.epochs, prefix=args.prefix, out_dir=out_dir, timeout_s=args.launch_timeout_s, retries=args.launch_retries)
        launched.append(run)
        _write_manifest(manifest_path, launched)
        _write_variants(variants_path, launched)
    _write_manifest(manifest_path, launched)
    _write_variants(variants_path, launched)
    print(json.dumps({"out_dir": str(out_dir), "count": len(launched)}, indent=2))


if __name__ == "__main__":
    main()
