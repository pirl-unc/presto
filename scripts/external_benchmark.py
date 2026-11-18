#!/usr/bin/env python
"""Run NetMHCpan-4.1 / NetMHCIIpan-4.3 / MHCflurry on held-out Presto dump rows.

Usage: ``python -m presto.scripts.external_benchmark --manifest launch.json``.
The manifest lists ``{"label", "family", "seed", "split", "dump", "config"}`` entries;
see ``experiments/2026-10-31_0830_claude_netmhc-benchmark/reproduce/launch.json``.
CPU only; at most 4 concurrent tool processes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from presto.evaluation import external_benchmark as eb


def _list_alleles(command: list) -> list:
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("PLATFORM")
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True, help="experiment directory (tables, hashes)")
    parser.add_argument("--artifacts-dir", required=True, help="per-example dumps (gitignored)")
    parser.add_argument("--bundle-home", default="/Users/iskander/code/netmhc-bundle")
    parser.add_argument(
        "--mhcflurry", default="/Users/iskander/code/shared-virtual-env/bin/mhcflurry-predict"
    )
    parser.add_argument("--row-draws", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    out_dir, art = Path(args.out_dir), Path(args.artifacts_dir)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    bundle = args.bundle_home
    supported = {
        "netmhcpan41": _list_alleles([f"{bundle}/bin/netMHCpan-4.1", "-listMHC"]),
        "netmhciipan43": _list_alleles([f"{bundle}/bin/netMHCIIpan-4.3", "-list"]),
        "mhcflurry": _list_alleles([args.mhcflurry, "--list-supported-alleles"]),
    }
    for tool, names in supported.items():
        (art / f"supported_alleles_{tool}.txt").write_text("\n".join(names) + "\n")

    frames = {}
    for entry in manifest["dumps"]:
        frame = eb.extract_records(entry["dump"])
        frame, sub_log = eb.subsample_per_study(frame)
        entry["subsampled"] = sub_log
        frame = eb.annotate_eligibility(frame, supported)
        frames[entry["label"]] = frame

    # Unique eligible (tool, allele, peptide) jobs across every dump.
    jobs = {tool: set() for tool in eb.ALL_TOOLS}
    for frame in frames.values():
        ok = frame[frame["exclusion"].isna()]
        for tool in eb.ALL_TOOLS:
            sub = ok[ok[f"allele_{tool}"].notna()]
            jobs[tool].update(zip(sub[f"allele_{tool}"], sub["peptide"]))
    cache_path = art / "tool_predictions.csv.gz"
    cached = (
        pd.read_csv(cache_path)
        if cache_path.exists()
        else pd.DataFrame(
            columns=["tool", "allele", "peptide", "ic50_nM", "allele_distance_to_training"]
        )
    )
    have = set(zip(cached["tool"], cached["allele"], cached["peptide"]))
    new_parts, run_logs = [], []
    for tool in eb.ALL_TOOLS:
        todo = pd.DataFrame(
            [(a, p) for a, p in sorted(jobs[tool]) if (tool, a, p) not in have],
            columns=["allele", "peptide"],
        )
        if todo.empty:
            continue
        if tool == "mhcflurry":
            new_parts.append(eb.run_mhcflurry(todo, args.mhcflurry, str(art)))
        else:
            part, logs = eb.run_bundle_tool(tool, todo, bundle)
            new_parts.append(part)
            run_logs.extend(logs)
        print(f"{tool}: ran {len(todo)} pairs", flush=True)
    predictions = pd.concat([cached, *new_parts], ignore_index=True)
    predictions.to_csv(cache_path, index=False)

    distance = (
        predictions[predictions["tool"] != "mhcflurry"]
        .groupby(["tool", "allele"])["allele_distance_to_training"]
        .first()
        .reset_index()
    )
    distance.to_csv(out_dir / "results" / "allele_distance_to_training.csv", index=False)

    hashes = {"tool_predictions.csv.gz": eb.sha256_file(cache_path)}
    summary = {}
    for entry in manifest["dumps"]:
        label = entry["label"]
        scales = json.loads(Path(entry["config"]).read_text())["scales"]
        frame = eb.attach_tool_predictions(frames[label], predictions)
        frame = eb.add_train_mean(frame, scales)
        frame = eb.apply_tool_failures(frame)
        dump_path = art / f"predictions_{label}.csv.gz"
        frame.to_csv(dump_path, index=False)
        hashes[dump_path.name] = eb.sha256_file(dump_path)
        tables = eb.score_dump(frame, row_draws=args.row_draws, draws=args.draws)
        for name, table in tables.items():
            table.to_csv(out_dir / "results" / f"{label}__{name}.csv", index=False)
        census = tables["census"]
        summary[label] = {
            "family": entry["family"],
            "seed": entry["seed"],
            "split": entry["split"],
            "developmental": entry["split"] == "validation",
            "dump_sha256": eb.sha256_file(entry["dump"]),
            "rows_in_scope": int(len(frame)),
            "census": census.to_dict("records"),
            "subsampled": entry["subsampled"],
        }
        print(f"{label}: scored {int((frame['exclusion'].isna()).sum())} eligible rows", flush=True)

    (out_dir / "results" / "run_summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "results" / "prediction_dump_hashes.json").write_text(json.dumps(hashes, indent=1))
    (out_dir / "results" / "tool_versions.json").write_text(
        json.dumps(eb.tool_versions(bundle, args.mhcflurry), indent=1)
    )
    (out_dir / "results" / "tool_run_log.json").write_text(json.dumps(run_logs, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
