"""Collect specific native-source calls; a polling timeout is never failure."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import modal


def collect(row, output):
    path = output / (row["source_file"] + ".json.gz")
    if path.exists():
        result = json.loads(gzip.decompress(path.read_bytes()))
        assert result["receipt"]["source"]["name"] == row["source_file"]
    else:
        try:
            result = modal.FunctionCall.from_id(row["call_id"]).get(timeout=0)
        except TimeoutError:
            return {**row, "status": "pending"}
        except Exception as error:
            # Preserve remote errors for inspection; never translate one to success.
            return {
                **row,
                "status": "error",
                "error": type(error).__name__ + ": " + str(error),
            }
        payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        with gzip.open(path, "wb") as stream:
            stream.write(payload)
    return {
        **row,
        "status": "completed",
        "receipt": result["receipt"],
        "n_target_psms": len(result["queries"]["target_psms"]["rows"]),
        "n_high_psms": sum(r[2] == 3 for r in result["queries"]["target_psms"]["rows"]),
        "local_result": str(path),
        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt-dir", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.calls.read_text().splitlines()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda row: collect(row, args.output), rows))
    name = datetime.now(timezone.utc).strftime("native-collection-%Y%m%dT%H%M%SZ.json")
    (args.receipt_dir / name).write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": name,
                "completed": sum(r["status"] == "completed" for r in results),
                "pending": [
                    r["source_file"] for r in results if r["status"] == "pending"
                ],
                "errors": [r for r in results if r["status"] == "error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
