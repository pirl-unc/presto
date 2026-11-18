"""Read-only, timestamped snapshots of committed training histories."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import modal

E = Path(__file__).resolve().parents[1]
DEFAULT = ",".join(
    [
        f"{arm}-s{seed}-n6000"
        for seed in (42, 43)
        for arm in ("expanded-none", "expanded-state")
    ]
    + ["pilot-balanced-s42-n6000"]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=DEFAULT)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = E / "results/interim" / stamp
    folder.mkdir(parents=True)
    volume = modal.Volume.from_name("presto-checkpoints")

    def read(run):
        remote = f"{E.name}/fits/{run}/raw/history.json"

        def optional(name):
            try:
                return json.loads(
                    b"".join(volume.read_file(f"{E.name}/fits/{run}/{name}"))
                )
            except (FileNotFoundError, modal.exception.NotFoundError):
                return None

        completed = optional("receipt.json")
        failure = optional("failure.json")
        try:
            data = b"".join(volume.read_file(remote))
        except (FileNotFoundError, modal.exception.NotFoundError):
            return {
                "run": run,
                "status": "failed" if failure else "no_committed_history_yet",
                "failure": failure,
            }
        (folder / f"{run}.json").write_bytes(data)
        history = json.loads(data)
        if not history:
            return {"run": run, "status": "empty_history"}
        selected = [h for h in history if h["selected"]][-1]

        def metrics(h):
            classes = h["validation"]["elution_observation"]["by_class"]
            return {
                "steps": h["steps"],
                "auroc": {c: v["auroc"] for c, v in classes.items()},
                "rows": {c: v["n"] for c, v in classes.items()},
            }

        return {
            "run": run,
            "status": "completed" if completed else "failed" if failure else "interim",
            "completion_receipt": completed,
            "failure": failure,
            "latest": metrics(history[-1]),
            "currently_selected": metrics(selected),
            "remote": remote,
            "history_sha256": hashlib.sha256(data).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=5) as pool:
        receipts = list(pool.map(read, args.runs.split(",")))
    (folder / "summary.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps({"snapshot": str(folder), "runs": receipts}, indent=2))


if __name__ == "__main__":
    main()
