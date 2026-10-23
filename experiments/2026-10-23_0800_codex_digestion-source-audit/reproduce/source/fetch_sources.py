"""Fetch only the pinned processed data and small original metadata assets."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


for item in json.loads((ROOT / "results/source-assets.json").read_text()):
    destination = Path(item["path"])
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--ssl-reqd",
                "--connect-timeout",
                "12",
                "--max-time",
                "600",
                "--silent",
                "--show-error",
                item["url"],
                "-o",
                str(temporary),
            ],
            check=True,
        )
        assert temporary.stat().st_size == item["bytes"]
        assert digest(temporary) == item["sha256"]
        temporary.rename(destination)
    assert destination.stat().st_size == item["bytes"]
    assert digest(destination) == item["sha256"]
    print(destination.name, "verified", flush=True)
