"""Audit source modification headers only; no quantitative outcomes or predictions."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("--requests", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.requests.read_text())
root = Path(__file__).resolve().parents[1]
views = pq.read_table(
    request["source_views"],
    columns=[
        "view_id",
        "source_file",
        "sheet_name",
        "source_view_role",
        "interpretation_json",
    ],
).to_pylist()
selected = []
for view in views:
    if view["source_view_role"] != "organs":
        continue
    fields = json.loads(view.pop("interpretation_json"))["original_headers"]
    selected.append(
        {
            **view,
            "modification_headers": {
                column: label
                for column, label in fields.items()
                if isinstance(label, str) and "mod" in label.lower()
            },
        }
    )
counts = Counter(
    json.dumps(v["modification_headers"], sort_keys=True) for v in selected
)
result = {
    "n_views": len(selected),
    "header_patterns": dict(counts),
    "views": selected,
    "quantitative_outcomes_queried": False,
}
with (root / "results/modification-header-audit.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps({"n_views": len(selected), "header_patterns": dict(counts)}))
