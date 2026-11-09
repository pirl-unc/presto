"""Archive immutable run outputs and verify every archived byte against its source."""

import hashlib
import json
from pathlib import Path
import shutil
import tarfile

FAMILY = Path(__file__).resolve().parents[1]
RUN = FAMILY / "results/run"
archive = FAMILY / "results/raw-run.tar.gz"
if archive.exists():
    raise FileExistsError(archive)
sources = {str(p.relative_to(RUN)): p for p in RUN.rglob("*") if p.is_file()}
with tarfile.open(archive, "w:gz") as stream:
    for name, path in sorted(sources.items()):
        stream.add(path, arcname=f"run/{name}", recursive=False)
with tarfile.open(archive, "r:gz") as stream:
    members = stream.getmembers()
    assert len(members) == len(sources)
    for member in members:
        name = member.name.removeprefix("run/")
        assert stream.extractfile(member).read() == sources[name].read_bytes()
for name in ("summary.json", "execution.json", "teacher.json", "artifacts.json"):
    shutil.copy2(RUN / name, FAMILY / "results" / name)
receipt = {
    "status": "passed",
    "files": len(sources),
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
}
(FAMILY / "results/package-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
