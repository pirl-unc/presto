"""Fetch the exact released producer through shared datacache."""

import json
from pathlib import Path

from datacache import fetch_file

producer = json.loads((Path(__file__).resolve().parents[1] / "reproduce/producer.json").read_text())
print(
    fetch_file(
        producer["wheel_url"],
        filename=producer["wheel_name"],
        subdir="presto/releases/0.1.16",
        cache_root=Path.home() / ".hitlist",
        expected_sha256=producer["wheel_sha256"],
        expected_size=producer["wheel_size_bytes"],
        show_progress=False,
    )
)
