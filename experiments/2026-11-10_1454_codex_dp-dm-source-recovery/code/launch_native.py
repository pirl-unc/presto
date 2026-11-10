"""Queue the frozen public native-source recovery; two CPU workers, no GPU."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import modal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--calls", type=Path, required=True)
    args = p.parse_args()
    items = json.loads(args.manifest.read_text())
    q = json.loads(args.queries.read_text())
    fn = modal.Function.from_name("presto-dp-dm-native-source-20261110", "recover")
    with args.calls.open("x") as out:
        for item in items:
            call = fn.spawn(item, q)
            rec = {
                "call_id": call.object_id,
                "source_file": item["name"],
                "spawned_unix": time.time(),
                "queries": args.queries.name,
                "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
                "gpu": None,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
