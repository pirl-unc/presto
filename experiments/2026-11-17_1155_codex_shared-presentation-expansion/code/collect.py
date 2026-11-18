"""Collect exact run files; never recursively copy a volume directory."""

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

E = Path(__file__).resolve().parents[1]
NAME = E.name


def main():
    a = argparse.ArgumentParser()
    a.add_argument(
        "--runs",
        default="pilot-balanced-s42-n6000,expanded-none-s42-n6000,expanded-state-s42-n6000",
    )
    a.add_argument("--evaluation", action="store_true")
    a.add_argument("--context-controls", action="store_true")
    a.add_argument("--root-files")
    a.add_argument(
        "--unstaged",
        action="store_true",
        help="Original integration run before local source staging",
    )
    args = a.parse_args()
    if args.evaluation and args.context_controls:
        a.error("choose either --evaluation or --context-controls")
    vol = modal.Volume.from_name("presto-checkpoints")
    jobs = []
    for run in [] if args.root_files else args.runs.split(","):
        if args.context_controls:
            names = ["receipt.json"] + [
                f"{population}-{mode}-predictions.jsonl.gz"
                for population in ("apc", "expanded", "matched")
                for mode in ("original", "unknown", "within-study-shuffle")
            ]
            folder = "context-ablation"
        elif args.evaluation:
            names = [
                "receipt.json",
                "matched-predictions.jsonl.gz",
                "matched-metrics.json",
            ]
            if run in ("parent", "previous-default"):
                names += [
                    f"{k}-{t}.{suffix}"
                    for k in ("apc", "affinity", "broad", "expanded")
                    for t, suffix in [("predictions", "jsonl.gz"), ("metrics", "json")]
                ]
            elif run.endswith("-terminal"):
                names += [
                    "checkpoint.pt",
                    "validation-predictions.jsonl.gz",
                    "validation-metrics.json",
                ]
            folder = "evaluation"
        else:
            names = [
                "receipt.json",
                "manifest.json",
                "strata.json",
                "execution-manifest.json",
                "source-staging.json",
                "hardware.json",
                "update-check.json",
                "actual-training-exposure.json",
                "validation-context-projection.json",
                "raw/selected.pt",
                "raw/summary.json",
                "raw/history.json",
                "raw/coverage.json",
                "raw/admission.json",
                "raw/resolved-config.json",
                "raw/completed.json",
                "raw/validation-predictions.jsonl.gz",
            ]
            if args.unstaged:
                names = [
                    name
                    for name in names
                    if name not in ("execution-manifest.json", "source-staging.json")
                ]
            folder = "fits"
        for name in names:
            jobs.append((f"{folder}/{run}/{name}", E / "results" / folder / run / name))
    if args.root_files:
        jobs = [(name, E / "results" / name) for name in args.root_files.split(",")]

    def get(job):
        name, path = job
        compressed = path.name.endswith("-predictions.jsonl")
        if compressed:
            path = path.with_suffix(path.suffix + ".gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        size = 0
        temp = path.with_suffix(path.suffix + ".partial")
        with gzip.open(temp, "wb", compresslevel=3) if compressed else temp.open("wb") as f:
            for block in vol.read_file(f"{NAME}/{name}"):
                f.write(block)
                h.update(block)
                size += len(block)
        if compressed:
            check = hashlib.sha256()
            with gzip.open(temp, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    check.update(block)
            assert check.hexdigest() == h.hexdigest(), name
        stored = hashlib.sha256(temp.read_bytes()).hexdigest()
        temp.replace(path)
        return {
            "path": str(path.relative_to(E)),
            "remote": f"{NAME}/{name}",
            "sha256": h.hexdigest(),
            "bytes": size,
            "stored_sha256": stored,
            "stored_bytes": path.stat().st_size,
            "compression": "gzip" if compressed else None,
            "verified_lossless": compressed,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(get, jobs))
    out = (
        E
        / "results"
        / (
            "collection-root-" + hashlib.sha256(args.root_files.encode()).hexdigest()[:12]
            if args.root_files
            else "collection-controls-" + args.runs.replace(",", "_")
            if args.context_controls
            else "collection-eval-" + args.runs.replace(",", "_")
            if args.evaluation
            else "collection-fit-" + args.runs.replace(",", "_")
        )
    )
    out = out.with_suffix(".json")
    out.write_text(json.dumps(receipts, indent=2))
    print("collected", len(receipts))


if __name__ == "__main__":
    main()
