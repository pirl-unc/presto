#!/usr/bin/env python
"""Re-score each finished run with the checkpoint it actually selected.

The runs in this family were launched before `train_iedb` reloaded the
best-validation checkpoint for the held-out pass, so their `val_summary.json`
/ `test_summary.json` / `*_predictions.csv` describe the final epoch while
carrying `best_val_loss` from a different one. `model.pt` on disk is the
selected epoch, so the correct numbers are recoverable without spending
another GPU-second.

This driver replays the held-out pass only. It calls `train_iedb.run()` with
the run's own saved `config.json`, `--epochs 0` so no training happens, and
`--checkpoint` pointing at the fetched `model.pt`; the reload added in
`train_iedb` then scores that checkpoint through the production pipeline
rather than a reimplementation of it.

Validity of doing this locally: the five frozen hitlist artifacts in
`~/.hitlist` are byte-identical to the ones the Modal runs hash-verified, and
a local load reproduces each run's split exactly (seed 42 test = 3,772 rows /
2,900 peptides, matching the run log). The local hitlist *package* is 1.56.0
against the runs' 1.55.8, which is why the split is re-derived and compared
rather than assumed -- a mismatch aborts that run instead of writing numbers
that cannot be compared.

Outputs land in `<run>/best_checkpoint_eval/`. Nothing overwrites the original
artifacts; the two sets are meant to be read side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from presto.scripts import train_iedb  # noqa: E402

#: What the replay must not inherit from the original run.
OVERRIDES = {
    "epochs": 0,
    "num_workers": 0,
    "compile": False,
}


def _split_signature(run_dir: Path) -> tuple[int, int] | None:
    """`(test_rows, test_peptides)` as the original run reported them."""
    log = run_dir / "run.log"
    if not log.is_file():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("Split: peptide-grouped"):
            try:
                test_part = line.split("test=")[1]
                rows = int(test_part.split()[0])
                peptides = int(test_part.split("/")[1].split()[0])
                return rows, peptides
            except (IndexError, ValueError):
                return None
    return None


def _replay(run_dir: Path, *, device: str) -> dict:
    config_path = run_dir / "config.json"
    checkpoint = run_dir / "model.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"{run_dir.name}: no config.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{run_dir.name}: no model.pt -- it is gitignored, so fetch the run "
            f"from the Modal volume before replaying"
        )

    config = json.loads(config_path.read_text())
    out_dir = run_dir / "best_checkpoint_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    config.update(OVERRIDES)
    config["device"] = device
    config["checkpoint"] = str(checkpoint)
    config["run_dir"] = str(out_dir)
    # The hitlist-only path is selected by an empty data dir; the original
    # value was a container path that does not exist here.
    data_dir = Path("/tmp/presto-hitlist-only")
    data_dir.mkdir(parents=True, exist_ok=True)
    config["data_dir"] = str(data_dir)
    config["index_csv"] = str(REPO_ROOT / "data" / "mhc_index.csv")

    # `train_iedb` prints the split line but writes no run.log -- the Modal
    # launcher did that. Tee stdout so the split check below has something to
    # read rather than silently finding no file and passing.
    replay_log = out_dir / "run.log"
    with replay_log.open("w", encoding="utf-8") as handle:

        class _Tee:
            def write(self, text):
                handle.write(text)
                sys.__stdout__.write(text)

            def flush(self):
                handle.flush()
                sys.__stdout__.flush()

        saved = sys.stdout
        sys.stdout = _Tee()
        try:
            train_iedb.run(Namespace(**config))
        finally:
            sys.stdout = saved

    result = {"run": run_dir.name, "out_dir": str(out_dir)}
    expected = _split_signature(run_dir)
    actual = _split_signature(out_dir)
    # Both must be readable. A missing signature means the comparison did not
    # happen, which is not the same as it having passed.
    if expected is None or actual is None:
        raise RuntimeError(
            f"{run_dir.name}: could not read the split line from "
            f"{'the original run.log' if expected is None else 'the replay'}; "
            f"refusing to report metrics whose held-out rows are unverified"
        )
    if expected != actual:
        raise RuntimeError(
            f"{run_dir.name}: replayed split {actual} does not match the "
            f"original {expected}; the recomputed metrics would describe "
            f"different held-out rows"
        )
    result["split_matches_original"] = True
    result["test_rows"], result["test_peptides"] = actual
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(EXPERIMENT_DIR / "results" / "runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", default=None, help="substring filter on run id")
    args = parser.parse_args()

    runs = sorted(p for p in Path(args.runs_dir).iterdir() if p.is_dir())
    if args.only:
        runs = [p for p in runs if args.only in p.name]
    if not runs:
        raise SystemExit(f"no runs under {args.runs_dir}")

    results = []
    for run_dir in runs:
        print(f"\n=== replaying held-out pass for {run_dir.name} ===", flush=True)
        results.append(_replay(run_dir, device=args.device))

    summary_path = Path(args.runs_dir).parent / "best_checkpoint_eval_status.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
