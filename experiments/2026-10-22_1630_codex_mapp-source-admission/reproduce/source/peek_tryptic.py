"""Identify the recovered candidate's run headers without admitting quantities."""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time


FAMILY = Path(__file__).resolve().parents[2 if Path(__file__).parent.name == "source" else 1]
REPO = FAMILY.parents[1]


def main():
    start = time.perf_counter()
    receipt = json.loads((FAMILY / "retrievals/tryptic_receipt.json").read_text())
    archive = REPO / receipt["immutable_path"]
    sha = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    assert sha.hexdigest() == receipt["asset"]["sha256"]
    tables, hashes = {}, {}
    for name in ["summary", "parameters"]:
        raw = subprocess.check_output(["tar", "-xOzf", str(archive), f"txt/{name}.txt"])
        hashes[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t"))
    runs = [{"source_row": i, **{k: r[k] for k in ["Raw file", "Experiment", "Enzyme", "Enzyme mode"]}}
            for i, r in enumerate(tables["summary"], 2) if r["Experiment"]]
    metadata = json.loads((REPO / "data/availability_sources/snapshots/2026-10-22-mapp-series/PXD009360.files.json").read_text())
    raw_names = {f["fileName"].removesuffix(".raw") for f in metadata if f["fileName"].endswith(".raw")}
    assert {r["Raw file"] for r in runs} == raw_names and len(runs) == len(raw_names)
    result = {"n_source_summary_rows": len(tables["summary"]), "runs": runs,
              "selected_parameters": [r for r in tables["parameters"] if r["Parameter"] in
                  {"Version", "PSM FDR", "Protein FDR", "Match between runs", "Fasta file"}],
              "member_hashes": hashes, "seconds": time.perf_counter() - start,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "condition_map_status": "unresolved; source experiments are bare numeric labels",
              "quantity_audit_status": "not performed"}
    (FAMILY / "retrievals/tryptic_headers.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
