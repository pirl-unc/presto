"""Small temporary-volume write diagnostic; no model training or cohort mutation."""

import hashlib
import itertools
import json
import tempfile
import time
from pathlib import Path


def probe(root):
    source = root / "inputs/expanded-train.jsonl"
    with source.open() as stream:
        records = list(itertools.islice(stream, 10000))
    expected = hashlib.sha256("".join(records).encode()).hexdigest()
    result = {
        "source": str(source),
        "rows": len(records),
        "sha256": expected,
        "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="io-probe-", dir=root) as temp:
        for buffer in [8192, 1048576]:
            path = Path(temp) / f"buffer-{buffer}.jsonl"
            start, cpu = time.monotonic(), time.process_time()
            with path.open("w", buffering=buffer) as stream:
                for record in records:
                    stream.write(record)
            result["runs"].append(
                {
                    "buffer_bytes": buffer,
                    "bytes": path.stat().st_size,
                    "wall_seconds": time.monotonic() - start,
                    "cpu_seconds": time.process_time() - cpu,
                }
            )
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    (root / "write-io-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import sys

    probe(Path(sys.argv[1]))
