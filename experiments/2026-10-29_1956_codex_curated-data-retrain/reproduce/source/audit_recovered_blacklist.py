"""Count actual old-label intersections before revising the positive exclusion pool."""

from collections import Counter
from pathlib import Path
import csv
import gzip
import hashlib
import json
import shutil
import sqlite3
import time
import modal

FID = "2026-10-29_1956_codex_curated-data-retrain"
PARENT = "2026-10-28_1540_codex_temporal-multi-output"
CURATION = "2026-10-29_1653_codex_measurement-curation-recovery"
app = modal.App("presto-recovered-positive-intersection-audit")
volume = modal.Volume.from_name("presto-checkpoints")
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    source = Path("/Users/iskander/code/presto/artifacts") / CURATION / "recovered/strazar_2023_recovered_unmodified.csv"
    image = image.add_local_file(source, "/bundle/recovered.csv")


@app.function(image=image, cpu=4, memory=8192, timeout=7200, max_containers=1,
              volumes={"/volume": volume})
def run():
    started = time.monotonic()
    with open("/bundle/recovered.csv") as stream:
        reviewed = {r["peptide"] for r in csv.DictReader(stream)}
    source = Path("/volume") / CURATION / "recovered-admission/train.jsonl.gz"
    with gzip.open(source, "rt") as stream:
        admitted = [json.loads(line) for line in stream]
    eligible = {r["peptide"] for r in admitted}
    db = sqlite3.connect("file:/volume/" + PARENT + "/admission/admitted.sqlite?mode=ro", uri=True)
    missing = set()
    for peptide in reviewed:
        if db.execute("SELECT 1 FROM positive_peptides WHERE peptide=?", (peptide,)).fetchone() is None:
            missing.add(peptide)
    db.close()
    base = Path("/volume") / PARENT / "revisions/kinetic-review-v2/prepared"
    manifest = json.loads((base / "wide-multi-s42.json").read_text())
    intersections, heldout, hashes = [], set(), {}
    for spec in manifest["sources"]:
        path = base / Path(spec["path"]).name
        is_background = spec["source"] == "hitlist_ligand_background"
        is_heldout = path.name.endswith(("-validation.jsonl", "-test.jsonl"))
        if not (is_background or is_heldout):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                if is_heldout:
                    heldout.add(row["peptide"])
                if row["peptide"] in reviewed:
                    intersections.append(dict(observation_id=row["observation_id"], split=row["split"],
                        generated=is_background, peptide_sha256=hashlib.sha256(row["peptide"].encode()).hexdigest(),
                        recovered_eligible=row["peptide"] in eligible, previously_blacklisted=row["peptide"] not in missing))
        assert digest.hexdigest() == spec["sha256"]
        hashes[path.name] = digest.hexdigest()
        print("Scanned", path.name, flush=True)
    result = dict(status="audited", reviewed_unique_sequences=len(reviewed),
        eligible_unique_sequences=len(eligible), missing_reviewed_from_old_blacklist=len(missing),
        missing_eligible_from_old_blacklist=len(missing & eligible),
        new_heldout_embargo_observations=sum(r["peptide"] in heldout for r in admitted),
        counts=dict(Counter(r["split"] + (":generated" if r["generated"] else ":original") for r in intersections)),
        intersections=intersections, source_hashes=hashes, seconds=time.monotonic()-started)
    return result


@app.local_entrypoint()
def main():
    call = run.spawn()
    shutil.copyfile(__file__, F / "reproduce/source/audit_recovered_blacklist.py")
    (F / "reproduce/blacklist-audit-call.json").write_text(json.dumps(dict(call_id=call.object_id)) + "\n")
    result = call.get()
    (F / "results/blacklist-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("intersections", "source_hashes")}), flush=True)
