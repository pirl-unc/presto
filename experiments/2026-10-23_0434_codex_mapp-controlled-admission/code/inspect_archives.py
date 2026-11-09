"""Inventory original processed archives; no model-ready labels are inferred."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/archive-inspection.json").read_text())
ASSETS = Path(CONFIG["asset_root"])
START = time.perf_counter()
results = []
for project in CONFIG["projects"]:
    archive = ASSETS / project / "search.tar.gz"
    source_files = json.loads((ASSETS / project / "files.json").read_text())
    raw_names = {r["fileName"].removesuffix(".raw") for r in source_files if r["fileName"].endswith(".raw")}
    result = {"project": project, "members": [], "tables": {}, "run_assignments": []}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            entry = {"name": member.name, "size_bytes": member.size, "type": member.type.decode("ascii")}
            result["members"].append(entry)
            path = PurePosixPath(member.name)
            if path.name not in CONFIG["extract_leaves"]:
                continue
            assert not path.is_absolute() and ".." not in path.parts and member.isfile()
            dest = ASSETS / project / "extracted" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tar.extractfile(member) as source, dest.open("xb") as output:
                while data := source.read(1024 * 1024):
                    output.write(data)
                    digest.update(data)
            entry.update(extracted_path=str(dest), sha256=digest.hexdigest())
            if path.suffix != ".txt":
                continue
            with dest.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                headers = reader.fieldnames or []
                experiment_counts, raw_counts, sequence_hits = Counter(), Counter(), []
                intensity_columns = [h for h in headers if h.startswith("Intensity ")]
                positive_counts = Counter()
                rows = 0
                source_rows = []
                for row in reader:
                    rows += 1
                    if path.name in CONFIG["retain_all_rows_leaves"]:
                        source_rows.append(row)
                    if row.get("Experiment"):
                        experiment_counts[row["Experiment"]] += 1
                    if row.get("Raw file"):
                        raw_counts[row["Raw file"]] += 1
                    if path.name == "summary.txt" and row.get("Raw file") in raw_names:
                        result["run_assignments"].append({
                            "raw_file": row["Raw file"], "experiment_literal": row.get("Experiment"),
                            "fraction_literal": row.get("Fraction"), "source_member": member.name,
                            "source_line": rows + 1, "matches_deposited_raw_name": True,
                        })
                    if row.get("Sequence") in CONFIG["figure_reporter_sequences"]:
                        sequence_hits.append({"source_line": rows + 1, "row": row})
                    for column in intensity_columns:
                        if row.get(column) and float(row[column]) > 0:
                            positive_counts[column] += 1
                result["tables"][member.name] = {
                    "rows": rows, "headers": headers, "experiment_counts": dict(experiment_counts),
                    "raw_file_counts": dict(raw_counts), "positive_intensity_rows": dict(positive_counts),
                    "figure_reporter_sequence_hits": sequence_hits,
                    "source_rows": source_rows,
                }
    result["unmapped_deposited_raw_names"] = sorted(raw_names - {r["raw_file"] for r in result["run_assignments"]})
    results.append(result)
    (ROOT / "results/archive-inspection.json").write_text(json.dumps({
        "wall_seconds": time.perf_counter() - START, "projects": results,
    }, indent=2) + "\n")
    print(json.dumps({"project": project, "members": len(result["members"]),
                      "tables": {k: v["rows"] for k, v in result["tables"].items()},
                      "run_assignments": result["run_assignments"]}), flush=True)
