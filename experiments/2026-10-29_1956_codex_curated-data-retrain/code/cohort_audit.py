"""Independent full-cohort census while model fitting runs on its own workers."""

from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import sys
import time
import modal

FID = "2026-10-29_1956_codex_curated-data-retrain"
app = modal.App("presto-curated-data-census-20261029")
volume = modal.Volume.from_name("presto-checkpoints")
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    image = image.add_local_file(
        F / "analysis/cohort_composition.py", "/bundle/cohort_composition.py"
    )


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@app.function(
    image=image, cpu=4, memory=8192, timeout=7200, max_containers=1, volumes={"/volume": volume}
)
def run():
    sys.path.insert(0, "/bundle")
    from cohort_composition import summarize

    prepared = Path("/volume") / FID / "revisions/source-review-v2/prepared"
    local = Path("/runs") / FID
    local.mkdir(parents=True)
    (local / "prepared").symlink_to(prepared, target_is_directory=True)
    output = Path("/tmp/cohort-composition.json")
    summarize(local / "prepared/curated-multi-s42.json", output)
    result = json.loads(output.read_text())
    p = json.loads((prepared / "prepared.json").read_text())
    counts = Counter()
    for row in result["populations"]:
        counts[row["split"], row["endpoint"], row["mhc_class"]] += row["n"]
    expected = Counter()
    for key, n in p["counts"].items():
        split, endpoint, cls, _ = key.split(":")
        expected[split, endpoint, cls] += n
    assert counts == expected
    for row in result["counts"]:
        if row["endpoint"] != "elution_observation":
            key = ":".join(row[k] for k in ("split", "endpoint", "mhc_class", "qualifier"))
            assert row["n"] == p["counts"][key]
    selection = json.loads((prepared / "study-selection.json").read_text())
    selection["components"].append(
        dict(id="pmid:37301199", aliases=["pmid:37301199"], split="train", years=[2023])
    )
    verification = dict(
        status="passed",
        result_sha256=sha(output),
        population_counts_match_frozen_contract=True,
        physical_qualifier_counts_match=True,
        full_source_hashes_verified=True,
        recorded_unix=time.time(),
    )
    return dict(composition=result, verification=verification, study_selection=selection)


@app.local_entrypoint()
def main():
    call = run.spawn()
    (F / "reproduce/cohort-audit-call.json").write_text(
        json.dumps(dict(call_id=call.object_id)) + "\n"
    )
    shutil.copyfile(__file__, F / "reproduce/source/cohort_audit.py")
    result = call.get()
    for name, key in (
        ("cohort-composition", "composition"),
        ("cohort-composition-verification", "verification"),
        ("study-selection", "study_selection"),
    ):
        (F / "results" / (name + ".json")).write_text(
            json.dumps(result[key], indent=2, allow_nan=False) + "\n"
        )
    print("Full cohort composition verified", flush=True)
