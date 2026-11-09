"""Reproduce the source-level loss of MAPTAC DM labels without imputing them."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.csv as pacsv
import yaml

HERE = Path(__file__).resolve().parents[1]
SOURCE = Path("/private/tmp/presto-mhc2-20261026/hitlist-current/hitlist/data")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    pa.set_cpu_count(1)
    studies = yaml.safe_load((SOURCE / "pmid_overrides.yaml").read_text())
    study = next(x for x in studies if x["pmid"] == 31495665)
    dm_arms = [x for x in study["ms_samples"] if "(dm" in x["sample_label"]]
    beta_restrictions = {x["mhc"].split("/")[0] for x in dm_arms}
    path = SOURCE / "supplementary/abelin_2019_maptac_class2.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        counts = Counter(row["mhc_restriction"] for row in reader)
    table = pacsv.read_csv(path)
    replay = {
        row["mhc_restriction"]: row["peptide_count"]
        for row in table.group_by("mhc_restriction", use_threads=False)
        .aggregate([("peptide", "count")])
        .to_pylist()
    }
    assert replay == counts
    summary = {
        "hitlist_commit": "13b91424295e01286db46e8e9818fcc2fe816834",
        "hitlist_version": "2.0.42",
        "import_file": {"path": str(path), "sha256": digest(path), "columns": fields},
        "curation_file_sha256": digest(SOURCE / "pmid_overrides.yaml"),
        "n_import_rows": sum(counts.values()),
        "n_rows_for_restrictions_with_both_dm_arms": sum(counts[x] for x in beta_restrictions),
        "counts_by_literal_restriction": dict(counts),
        "curated_dm_arms": dm_arms,
        "limitation": (
            "CSV lacks sample/DM fields; no original workbook recovered. Counts describe "
            "merged import rows, not recovered observations or arm sizes."
        ),
        "independent_arrow_replay_passed": True,
        "seconds": time.monotonic() - started,
        "script_sha256": digest(Path(__file__)),
    }
    dest = HERE / "results/maptac-import-review.json"
    dest.parent.mkdir(exist_ok=True)
    with dest.open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (HERE / "reproduce/source/review_sources.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "n_import_rows",
                    "n_rows_for_restrictions_with_both_dm_arms",
                    "counts_by_literal_restriction",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
