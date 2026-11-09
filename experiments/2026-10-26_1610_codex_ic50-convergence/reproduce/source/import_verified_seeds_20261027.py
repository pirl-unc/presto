"""Copy completed immutable seed evidence without touching the live source runner."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

FAMILY = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/private/tmp/presto-pr-completion-20261026/pr131/presto/experiments/"
    "2026-10-26_1610_codex_ic50-convergence"
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    receipt = FAMILY / "reproduce/verified-seeds-import-20261027.json"
    assert not receipt.exists(), "Inspect the existing receipt before reimporting."
    files = []
    for seed in ("seed42", "seed43", "seed44"):
        verified = SOURCE / "results" / f"{seed}-independent-verification.json"
        result = json.loads(verified.read_text())
        assert result["status"] == "passed" and result["scientific_screen_passed"]
        files.extend(sorted((SOURCE / "results" / seed).iterdir()))
        files.extend([verified, SOURCE / "results" / f"{seed}-strata.json"])
    before = {str(path.relative_to(SOURCE)): sha(path) for path in files}
    for name in before:
        destination = FAMILY / name
        assert not destination.exists(), name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, destination)
    for name, expected in before.items():
        assert sha(SOURCE / name) == expected, "Source changed during copy: " + name
        assert sha(FAMILY / name) == expected, "Destination mismatch: " + name
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "destination": str(FAMILY),
        "source_and_destination_sha256": before,
        "verified_before_and_after": True,
        "copied_files": len(before),
        "copied_bytes": sum(path.stat().st_size for path in files),
        "source_runner_modified": False,
        "permutation_and_family_completion": "still pending; no whole-family success claim",
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified and copied {result['copied_files']} files / {result['copied_bytes']} bytes")


if __name__ == "__main__":
    main()
