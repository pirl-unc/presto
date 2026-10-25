"""Restore published compressed JSON artifacts without overwriting local evidence."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "reproduce/compressed-artifacts.json").read_text())
for item in manifest:
    packed = ROOT / item["compressed_path"]
    target = ROOT / item["original_path"]
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == item["compressed_sha256"]
    data = gzip.decompress(packed.read_bytes())
    assert len(data) == item["original_bytes"]
    assert hashlib.sha256(data).hexdigest() == item["original_sha256"]
    if target.exists():
        assert target.read_bytes() == data, target
        print("Already present and byte-identical:", item["original_path"])
    else:
        with target.open("xb") as stream:
            stream.write(data)
        print("Restored:", item["original_path"])
