"""Restore verified lossless result archives without replacing local artifacts."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "results/archive-manifest.json").read_text())
for entry in manifest["archives"]:
    source = (ROOT / entry["archive_path"]).resolve()
    target = (ROOT / entry["original_path"]).resolve()
    if not source.is_relative_to(ROOT) or not target.is_relative_to(ROOT):
        raise ValueError("Archive path escapes this experiment")
    compressed = source.read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == entry["archive_sha256"]
    raw = gzip.decompress(compressed)
    assert len(raw) == entry["original_size_bytes"]
    assert hashlib.sha256(raw).hexdigest() == entry["original_sha256"]
    if target.exists():
        assert target.read_bytes() == raw, f"Existing artifact differs: {target}"
        print("Verified existing", entry["original_path"])
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        print("Restored", entry["original_path"])
