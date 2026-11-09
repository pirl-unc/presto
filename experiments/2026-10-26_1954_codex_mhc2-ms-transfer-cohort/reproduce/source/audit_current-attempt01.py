"""Replay current restriction classification and retain literal CIITA statements."""

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import resource
import signal
import shutil
import sys
import time

import hitlist
import psutil
from hitlist.curation import restriction_evidence_for_row

FAMILY = Path(__file__).resolve().parents[1]
SOURCE = Path("/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01/observations.jsonl")
SOURCE_SHA = "532977c4a5e9b6530fca36d039b73a3b832c7b15197dc7a6def0bd66191aecef"
CURRENT = Path("/private/tmp/presto-mhc2-20261026/hitlist-current")


def main():
    started = time.monotonic()
    signal.alarm(120)
    assert Path(hitlist.__file__).resolve().is_relative_to(CURRENT)
    with SOURCE.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == SOURCE_SHA
    destination = FAMILY / "results/current-source-qualification.json"
    if destination.exists():
        raise FileExistsError(destination)
    signature = inspect.signature(restriction_evidence_for_row)
    counts, statements = Counter(), Counter()
    examples = []
    with SOURCE.open() as stream:
        for ordinal, line in enumerate(stream):
            if ordinal % 4096 == 0:
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024**3
                assert psutil.virtual_memory().available >= 10 * 1024**3
                assert shutil.disk_usage(SOURCE.parent).free >= 10 * 1024**3
            row = json.loads(line)
            raw = row["provenance"]["raw_record"]
            pmid = str(raw["pmid"])
            if pmid == "33858848":
                kwargs = {
                    key: raw.get(key) for key in signature.parameters if key != "mhc_restriction"
                }
                result = restriction_evidence_for_row(raw["mhc_restriction"], **kwargs)
                counts[(raw["restriction_evidence"], result)] += 1
                if len(examples) < 3:
                    examples.append(
                        {
                            "observation_id": row["observation_id"],
                            "restriction": raw["mhc_restriction"],
                            "native_evidence": raw["restriction_evidence"],
                            "current_evidence": result,
                        }
                    )
            elif pmid == "33592498":
                statements[(raw["condition_id"], raw["assay_comments"], raw["sample_note"])] += 1
    report = {
        "status": "replayed",
        "hitlist_version": hitlist.__version__,
        "current_curation_module": inspect.getfile(restriction_evidence_for_row),
        "current_curation_module_sha256": hashlib.sha256(
            Path(inspect.getfile(restriction_evidence_for_row)).read_bytes()
        ).hexdigest(),
        "input_sha256": SOURCE_SHA,
        "atlas_classification": [
            {"native": k[0], "current": k[1], "rows": v} for k, v in counts.items()
        ],
        "examples": examples,
        "ciita_statements": [
            {"condition_id": k[0], "assay_comments": k[1], "sample_note": k[2], "rows": v}
            for k, v in statements.items()
        ],
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "argv": sys.argv,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": (
            "current classifier replay; exact source assignment audit remains "
            "required, no blanket relabeling"
        ),
    }
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(
        json.dumps(
            {
                "atlas": report["atlas_classification"],
                "ciita_statement_groups": len(statements),
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
