"""Describe archived restriction deferrals using metadata only, not outcomes."""

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import re
import sys
import time

import pyarrow.parquet as pq

F = Path(__file__).resolve().parents[1]
OLD = F.parent / "2026-10-27_2235_codex_shared-support-audit"
SOURCE = Path(
    "/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z"
)
COLUMNS = [
    "mhc_restriction",
    "mhc_class",
    "response_measured",
    "source",
    "reference_iri",
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    admission = json.loads((OLD / "results/data-admission.json").read_text())
    dispositions = OLD / "results/quantitative-source-dispositions.jsonl.gz"
    selected = defaultdict(set)
    with gzip.open(dispositions, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["reason"] == "missing_explicit_mhc_partner":
                selected[row["source_file"]].add(row["ordinal"])
    expected = admission["counts"]["deferred:missing_explicit_mhc_partner"]
    assert sum(map(len, selected.values())) == expected
    counts = {
        k: Counter()
        for k in ("class", "restriction", "assay", "source", "loci", "study")
    }
    cross = Counter()
    found = 0
    for filename, digest in admission["source_hashes"].items():
        path = SOURCE / filename
        assert sha(path) == digest
        ordinal = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, columns=COLUMNS
        ):
            for row in batch.to_pylist():
                if ordinal in selected[filename]:
                    found += 1
                    restriction = str(row["mhc_restriction"])
                    loci = (
                        "+".join(
                            sorted(
                                set(
                                    re.findall(
                                        r"(?:HLA-)?(DRA|DRB[1345]|DQA1|DQB1|DPA1|DPB1|A|B|C)(?=[* :])",
                                        restriction,
                                    )
                                )
                            )
                        )
                        or "unparsed"
                    )
                    values = dict(
                        **{"class": str(row["mhc_class"])},
                        restriction=restriction,
                        assay=str(row["response_measured"]),
                        source=str(row["source"]),
                        loci=loci,
                        study=str(row["reference_iri"]),
                    )
                    for key, value in values.items():
                        counts[key][value] += 1
                    cross[(values["class"], loci, values["assay"])] += 1
                ordinal += 1
    assert found == expected
    result = dict(
        status="passed",
        scope="Metadata-only description of previously deferred rows. No peptide or quantitative outcome column read; no inferred partner, new admission, split or fit change. Locus names are lexical labels, not molecular resolution.",
        n=found,
        columns_read=COLUMNS,
        source_hashes=admission["source_hashes"],
        dispositions_sha256=sha(dispositions),
        counts={key: dict(value.most_common()) for key, value in counts.items()},
        class_locus_assay=[
            dict(mhc_class=c, loci=g, assay=a, n=n)
            for (c, g, a), n in cross.most_common()
        ],
    )
    (F / "results/deferred-restrictions.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (F / "reproduce/deferred-restrictions.json").write_text(
        json.dumps(
            dict(
                argv=sys.argv,
                executable=sys.executable,
                source_sha256=sha(Path(__file__)),
                source_directory=str(SOURCE),
                completed_unix=time.time(),
                columns_read=COLUMNS,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                n=found,
                classes=result["counts"]["class"],
                loci=result["counts"]["loci"],
                assays=result["counts"]["assay"],
                largest_groups=result["class_locus_assay"][:12],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
