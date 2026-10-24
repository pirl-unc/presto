"""Archive executed raw outputs byte-for-byte with deterministic metadata."""

import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]


def main():
    groups = [[], [], []]
    for f in sorted((FAMILY / "results").iterdir()):
        if f.name in ["qualification.json", "path-traces.json"] or f.name.startswith(
            "primary-"
        ):
            groups[0].append(f)
        elif re.match(r"actual-\d\d(?:\.|-)", f.name):
            groups[1 if int(f.name[7:9]) < 11 else 2].append(f)
    records = {}
    archives = []
    for name, group in zip(
        ["raw-numerical.tar.gz", "raw-actual-00-10.tar.gz", "raw-actual-11-20.tar.gz"],
        groups,
    ):
        output = FAMILY / "results" / name
        assert not output.exists(), "Refuse to replace immutable archive"
        members = []
        with (
            output.open("wb") as stream,
            gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as gz,
            tarfile.open(fileobj=gz, mode="w") as tar,
        ):
            for f in group:
                rel = str(f.relative_to(FAMILY))
                members.append(rel)
                records[rel] = {
                    "bytes": f.stat().st_size,
                    "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                }
                info = tar.gettarinfo(str(f), arcname=rel)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with f.open("rb") as raw:
                    tar.addfile(info, raw)
        archives.append(
            {
                "path": str(output.relative_to(FAMILY)),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "members": members,
            }
        )
        print(name, len(group), output.stat().st_size, flush=True)
    receipt = {
        "scope": "Exact raw outputs; restore under original relative paths with reproduce/unpack.py; local originals preserved",
        "files": records,
        "archives": archives,
    }
    (FAMILY / "results/raw-artifacts.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
