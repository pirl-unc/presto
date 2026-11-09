"""Compact quantity coverage, including numeric values embedded in Treatment."""

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / "data/mapp2025_source/2026-10-24_2130"


def main():
    start = time.monotonic()
    path = EXP / "results/workbook-audit.json"
    payload = path.read_bytes()
    document = json.loads(payload)
    pooled = next(s for s in document["sheets"] if s["sheet"] == "MAPP database")
    pattern_counts = Counter()
    labels = Counter()
    totals = Counter()
    repeated = 0
    pattern_regex = re.compile(r"(?:^|;)([^;:]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    for value, weight in pooled["condition_rows"].items():
        parts = [p.rsplit(":", 1) for p in value.split(";")]
        pairs = [(label, float(number)) for label, number in parts]
        alternate = [(m.group(1), float(m.group(2))) for m in pattern_regex.finditer(value)]
        assert pairs == alternate and all(math.isfinite(n) for _, n in pairs)
        names = tuple(label for label, _ in pairs)
        repeated += weight * (len(names) != len(set(names)))
        pattern_counts[names] += weight
        for label, number in pairs:
            labels[label] += weight
            totals["numeric_entries"] += weight
            totals["positive" if number > 0 else "zero" if number == 0 else "negative"] += weight
    pooled["embedded_quantity_summary"] = {
        **dict(totals),
        "rows_with_repeated_labels": repeated,
        "label_occurrences": dict(labels),
        "distinct_label_patterns": len(pattern_counts),
        "patterns": [{"ordered_labels": list(k), "rows": v} for k, v in pattern_counts.items()],
        "interpretation": "Numeric tokens with unresolved measurement and run scope; not admitted training values",
        "independent_split_vs_regex_agreement": True,
    }
    for sheet in document["sheets"]:
        sheet["distinct_treatment_strings"] = len(sheet.pop("condition_rows"))
    original = RAW / "workbook-audit-expanded.json"
    assert not original.exists() or original.read_bytes() == payload
    original.write_bytes(payload)
    document["expanded_audit"] = {
        "path": str(original),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    document["summary_seconds"] = time.monotonic() - start
    (EXP / "results/summary.json").write_text(json.dumps(document, indent=2) + "\n")
    print(
        json.dumps(
            {
                "embedded_numbers": dict(totals),
                "repeated_label_rows": repeated,
                "label_patterns": len(pattern_counts),
                "seconds": document["summary_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
