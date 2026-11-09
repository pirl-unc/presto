"""Independent bsdtar extraction and grouped-evidence replay of source audit."""

import gzip
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/quantity-audit.json").read_text())
ASSETS = Path(CONFIG["asset_root"])
START = time.perf_counter()
inspection = json.loads((ROOT / "results/archive-inspection.json").read_text())
reported = json.loads((ROOT / "results/quantity-audit.json").read_text())
results = []
for project in CONFIG["projects"]:
    inv = next(x for x in inspection["projects"] if x["project"] == project)
    stats = next(x for x in reported["projects"] if x["project"] == project)
    dest = ASSETS / "independent-replay" / project
    dest.mkdir(parents=True, exist_ok=False)
    members = [x for x in inv["members"] if "extracted_path" in x]
    archive = ASSETS / project / "search.tar.gz"
    metadata = json.loads((ASSETS / project / "files.json").read_text())
    original = next(x for x in metadata if x["fileName"] == "search.tar.gz")
    digest = hashlib.sha1()
    with archive.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    assert digest.hexdigest() == original["checksum"]
    assert archive.stat().st_size == original["fileSizeBytes"]
    subprocess.run(["/usr/bin/tar", "-xf", str(archive), "-C", str(dest), "--", *[m["name"] for m in members]], check=True)
    for member in members:
        source = dest / member["name"]
        assert source.stat().st_size == member["size_bytes"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == member["sha256"]
    def read(name):
        return pd.read_csv(dest / "txt" / name, sep="\t", dtype=str, keep_default_na=False)
    pep, ev, summary, groups = [read(name) for name in ("peptides.txt", "evidence.txt", "summary.txt", "proteinGroups.txt")]
    assert len(pep) == stats["peptide_rows"] and len(ev) == stats["evidence_rows"] and len(groups) == stats["protein_group_rows"]
    assert (groups["id"] == "").sum() == stats["blank_group_ids"]
    assert ev["id"].is_unique and pep["id"].is_unique
    peps = pep.set_index("id")
    assert (ev["Sequence"].to_numpy() == ev["Peptide ID"].map(peps["Sequence"]).to_numpy()).all()
    grouped_ids = ev.groupby("Peptide ID")["id"].agg(lambda x: frozenset(x)).to_dict()
    for _, row in pep.iterrows():
        assert frozenset(filter(None, row["Evidence IDs"].split(";"))) == grouped_ids.get(row["id"], frozenset())
    ev["value"] = pd.to_numeric(ev["Intensity"].replace("", "0"))
    sums = ev.groupby(["Peptide ID", "Experiment"])["value"].sum()
    experiments = sorted(ev["Experiment"].unique())
    strip = str.maketrans("", "", "_-")
    aliases = {c: [x for x in experiments if x.translate(strip) == c.removeprefix("Intensity ").translate(strip)] for c in pep if c.startswith("Intensity ")}
    assert all(len(x) == 1 for x in aliases.values())
    with gzip.open(ROOT / f"results/{project}-quantity-comparisons.jsonl.gz", "rt") as stream:
        dump = pd.DataFrame(json.loads(line) for line in stream)
    assert not dump.duplicated(["peptide_id", "source_column"]).any()
    assert len(dump) == len(pep) * len(aliases) == stats["comparisons"]
    expected = np.array([sums.get((p, aliases[c][0]), 0.0) for p, c in zip(dump.peptide_id, dump.source_column)])
    literal = [peps.at[p, c] for p, c in zip(dump.peptide_id, dump.source_column)]
    assert literal == dump.reported_literal.tolist()
    np.testing.assert_allclose(dump.evidence_sum, expected, rtol=1e-12, atol=1e-5)
    values = np.array([float(x) if x else np.nan for x in literal])
    states = np.full(len(dump), "positive_disagreement", dtype=object)
    states[np.isclose(values, expected, rtol=CONFIG["relative_tolerance"], atol=CONFIG["absolute_tolerance"]) & (values > 0) & (expected > 0)] = "positive_agree_within_rounding"
    states[(values == 0) & (expected == 0)] = "both_zero"
    states[(values == 0) & (expected > 0)] = "reported_zero_positive_evidence"
    states[(values > 0) & (expected == 0)] = "reported_positive_zero_evidence"
    states[np.isnan(values)] = "unreported"
    assert states.tolist() == dump.state.tolist()
    assert pd.Series(states).value_counts().to_dict() == stats["states"]
    for crosswalk in stats["run_crosswalk"]:
        subset = ev.loc[ev["Experiment"] == crosswalk["evidence_experiment"], "Raw file"]
        assert set(subset) == {crosswalk["raw_file"]}
        assert summary.loc[summary["Raw file"] == crosswalk["raw_file"], "Experiment"].tolist() == [crosswalk["summary_experiment"]]
    results.append({"project": project, "passed": True, "verified_member_files": len(members), "quantity_comparisons": len(dump), "max_sum_difference": float(np.max(np.abs(dump.evidence_sum - expected)))})
receipt = {"passed": True, "projects": results, "wall_seconds": time.perf_counter() - START,
           "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(ROOT / "results/independent-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
