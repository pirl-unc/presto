"""Independent original-XML recipe admission; no Hitlist imports."""

from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import resource
import time
import traceback

from xml_reference import read_xml, canonical, numeric, colno

R = Path(__file__).resolve().parents[1]
C = json.loads((R / "reproduce/launch.json").read_text())


def main(report):
    assert json.loads((R / "results/admission-execution.json").read_text())["passed"]
    package = Path(C["recipe_output"])
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["source_package_manifest_sha256"] == C["source_manifest_sha256"]
    for name, expected in manifest["files"].items():
        data = (package / name).read_bytes()
        assert (
            len(data) == expected["size_bytes"]
            and hashlib.sha256(data).hexdigest() == expected["sha256"]
        )
    tables = json.loads((package / "source_tables.json").read_text())
    books = {
        n: next(
            iter(
                read_xml(Path(C["original_assets"]) / f"supplementary-data-{n}.xlsx")[
                    0
                ].values()
            )
        )
        for n in (1, 2, 4)
    }

    def v(n, coord):
        return books[n].get(coord, {}).get("value")

    panel = {
        canonical(v(2, f"A{row}"), panel=True): (v(2, f"A{row}"), v(2, f"C{row}"))
        for row in range(5, 35)
    }
    amounts = {
        (v(1, f"A{row}"), v(1, f"{col}5")): v(1, f"{col}{row}")
        for row in range(6, 11)
        for col in "BCDEF"
    }
    rank = sorted(
        panel,
        key=lambda seq: hashlib.sha256(
            (C["recipe_id"] + ":" + seq).encode()
        ).hexdigest(),
    )
    folds = {seq: i % 5 for i, seq in enumerate(rank)}
    source_reporters = {}
    mode = None
    for row in range(1, 75):
        label = v(4, f"A{row}")
        if isinstance(label, str) and label.lower() in ("un-normalized", "normalized"):
            mode = label.lower()
        if isinstance(label, str) and "." in label and mode == "un-normalized":
            source_reporters[label] = row
    assert len(source_reporters) == 31
    multiplicity = Counter(canonical(label) for label in source_reporters)
    reporters = {r["reporter_id"]: r for r in tables["reporter_forms"]}
    slots = {r["slot_id"]: r for r in tables["source_slots"]}
    loads = {r["nominal_load_id"]: r for r in tables["nominal_loads"]}
    decisions = {
        r["measurement_id"]: r
        for r in json.loads((package / "dispositions.json").read_text())
    }
    raw = [r for r in tables["s4_measurements"] if r["representation"] == "raw"]
    assert len(raw) == len(decisions) == 3069
    expected_groups = defaultdict(list)
    coordinates, reason_counts = set(), Counter()
    for row in raw:
        rep, slot = reporters[row["reporter_id"]], slots[row["slot_id"]]
        label = rep["label_raw"]
        seq = canonical(label)
        assert seq == rep["peptide_sequence"] and label in source_reporters
        coordinate = slot["source_column"] + str(source_reporters[label])
        assert row["source_locator"] == "supplementary-data-4.xlsx#" + coordinate
        assert coordinate not in coordinates
        coordinates.add(coordinate)
        col = colno(slot["source_column"])
        assert 6 <= col <= 104
        assert int(slot["site_label"]) == (col - 6) // 9 + 1
        value = v(4, coordinate)
        assert row["value"] == (value if numeric(value) else None)
        status = (
            "absent_source_cell"
            if coordinate not in books[4]
            else "explicit_empty_string"
            if value == ""
            else "reported_positive"
            if numeric(value) and value > 0
            else None
        )
        assert row["value_status"] == status
        decision = decisions[row["measurement_id"]]
        assert (
            decision["source_value_status"] == status
            and decision["source_value"] == row["value"]
        )
        assert (
            decision["source_locator"] == row["source_locator"]
            and decision["source_cell"] == row["source_cell"]
        )
        assert decision["sequence"] == seq and decision["fold_id"] == folds[seq]
        reasons = set()
        if multiplicity[seq] > 1:
            reasons.add("multiple_raw_reporters_for_nominal_standard")
        if re.search(r"\.sum\d+$", label):
            reasons.add("unresolved_reporter_aggregation")
        if re.findall(r"\[\+\d+\]", label) != re.findall(r"\[\+\d+\]", panel[seq][0]):
            reasons.add("unresolved_nominal_isotope_annotation")
        if not numeric(value) or value <= 0:
            reasons.add("response_not_reported_positive_finite")
        assert set(decision["reasons"]) == reasons
        assert decision["status"] == ("excluded" if reasons else "admitted")
        reason_counts.update(reasons)
        load = loads[row["nominal_load_id"]]
        assert load["unit"] == "fmol_on_column" and load["sample_label"] == "S4"
        assert load["value"] == amounts[panel[seq][1], "S4"]
        if not reasons:
            expected_groups[seq, slot["site_label"]].append((row, math.log10(value)))
        else:
            assert decision["example_id"] is None
    examples = json.loads((package / "examples.json").read_text())
    contributions = {
        r["measurement_id"]: r
        for r in json.loads((package / "contributors.json").read_text())
    }
    assert len(examples) == len(expected_groups)
    keys = set()
    all_used = set()
    for example in examples:
        key = (example["sequence"], example["site_label"])
        assert key not in keys
        keys.add(key)
        members = expected_groups[key]
        vals = sorted(value for _, value in members)
        median = (vals[(len(vals) - 1) // 2] + vals[len(vals) // 2]) / 2
        assert math.isclose(example["target_log10_response"], median, abs_tol=1e-12)
        assert example["min_log10_response"] == min(vals) and example[
            "max_log10_response"
        ] == max(vals)
        assert example["n_positive_source_slots"] == len(vals)
        assert set(example["measurement_ids"]) == {
            r["measurement_id"] for r, _ in members
        }
        assert example["nominal_load_fmol"] == amounts[panel[key[0]][1], "S4"]
        assert math.isclose(
            example["nominal_log10_load"],
            math.log10(example["nominal_load_fmol"]),
            abs_tol=1e-12,
        )
        assert example["fold_id"] == folds[key[0]]
        for row, value in members:
            mid = row["measurement_id"]
            assert mid not in all_used
            all_used.add(mid)
            link = contributions[mid]
            assert (
                link["example_id"]
                == example["example_id"]
                == decisions[mid]["example_id"]
            )
            assert (
                link["source_cell"] == row["source_cell"]
                and link["source_value"] == row["value"]
            )
            assert math.isclose(link["log10_response"], value, abs_tol=1e-12)
    assert all_used == contributions.keys()
    assert len({e["sequence"] for e in examples}) == 23
    summary = json.loads((package / "summary.json").read_text())
    assert summary["n_admitted_measurements"] == len(all_used)
    assert summary["n_examples"] == len(examples) and summary[
        "n_excluded_measurements"
    ] == 3069 - len(all_used)
    assert summary["reason_counts"] == dict(reason_counts)
    report.update(
        passed=True,
        raw_slots_verified=3069,
        n_admitted_measurements=len(all_used),
        n_examples=len(examples),
        n_sequences=23,
        reason_counts=dict(reason_counts),
        sequence_folds={
            seq: folds[seq] for seq in sorted({e["sequence"] for e in examples})
        },
        recipe_manifest_sha256=hashlib.sha256(
            (package / "manifest.json").read_bytes()
        ).hexdigest(),
    )


if __name__ == "__main__":
    start = time.perf_counter()
    report = {"passed": False}
    try:
        main(report)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - start
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (R / "results/independent-admission.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report), flush=True)
